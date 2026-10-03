"""
experiments/run_vwa_adv_curated_matrix.py
----------------------------------------------
VWA-Adv matrix over agent-attack's OWN curated adversarial tasks
(Wu et al., ICLR 2025), using their published evaluators for ASR/UA
rather than anything invented here.

HOW ASR AND UA ARE MEASURED (this is the important part):
  ASR : evaluator_router(agent_adv/<task>/config.json) -- the paper's own
        adversarial evaluator, scored against real Postmill end-state
        (e.g. program_html + reddit_get_latest_comment_content_by_username
        with must_exclude on the correct answer). Nothing custom.
  UA  : evaluator_router(agent_clean/<base>/config.json) -- the benign
        evaluator for the SAME user task, scored on the SAME final page
        state after the SAME attacked episode. One episode, two scorings,
        giving the full ASR x UA 2x2.

  UA IS DELIBERATELY SKIPPED where the adv config's `intent` differs from
  the clean config's. That is not laziness: the _1/_2 task variants carry
  genuinely DIFFERENT user intents (e.g. "release month in mm format" vs
  "release date in mm-dd format"), so the base task's benign evaluator is
  checking a task the agent was never asked to do. Scoring it anyway
  would manufacture false UA=False. Those rows get utility_success=None
  with a recorded reason.

ATTACK DELIVERY: keyed on the victim image's absolute URL, read from
agent-attack's own recorded obs_text.txt (already rewritten to the live
host by their scripts/process_data.py). Verified against a live instance:
the recorded URL matches the live DOM src exactly, both channels land --
adv_url2caption appears in obs['text'] as ", description: <payload>",
adv_url2image replaces node.src with a data: URL the screenshot picks up.

CWD: agent-attack's config.json files use storage_state paths relative to
the AGENT-ATTACK repo root ("../visualwebarena/./.auth/..."), NOT the
visualwebarena root. This driver therefore chdirs to agent-attack, unlike
run_vwa_adv_matrix.py which chdirs to visualwebarena. Getting this wrong
surfaces as a Playwright FileNotFoundError on the auth file.

STATUS: never run end-to-end. The pieces it composes are each verified
(episode loop, both injection channels, attack generation on real
weights), but their composition is not.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AGENT_ATTACK_REPO = Path(os.environ.get("AGENT_ATTACK_REPO", "/datasets/nmahomed/vwa-adv/agent-attack"))
VWA_ADV_REPO = Path(os.environ.get("VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
sys.path.insert(0, str(VWA_ADV_REPO))

import requests
import torch
from PIL import Image

from attacks.base import BaseAttack
from attacks.template_ipi import TemplateIPIAttack
from attacks.gcg import GCGAttack
from attacks.pixel_perturbation import PixelPerturbationAttack
from attacks.cross_modal_injection import CrossModalInjectionAttack
from attacks.typographic import TypographicAttack
from attacks.adversarial_patch import AdversarialPatchAttack
from attacks.steganographic import SteganographicAttack
from attacks.comet import CometAttack
from attacks.obfuscated_activation import ObfuscatedActivationAttack
from attacks.shipped_transfer import ShippedImageAttack
from extraction.hooks import ActivationExtractor
from environments.self_captioner import (SelfCaptioner, clear_caption_cache, seed_caption,
                                         CAPTION_PROMPT, FALLBACK_CAPTION, MAX_SIDE)
from experiments.code_version import (code_version, check_resume_compatible, attack_code_sha,
                                      dir_sha, files_sha, spotlight_seed_sha, write_manifest)
from experiments.episode_utils import (_cuda_memory_guard, _run_episode_safely, _build_defense_registry,
                                       atomic_write_json, atomic_write_text, atomic_torch_save,
                                       atomic_save_image)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"url:\s*(\S+?)\]?$")


class _ReusePayload(Exception):
    """Internal signal: a persisted payload exists; skip generation."""


class CuratedTask:
    """One curated (base task, adversarial goal) variant."""

    def __init__(self, adv_dir: Path, exp_data: Path):
        self.adv_dir = adv_dir
        self.name = adv_dir.name
        self.strict_asr = False
        self.clean_ua_authored = False
        self.base = "_".join(self.name.split("_")[:2])           # reddit_35_upvote_cap_1 -> reddit_35
        self.data = json.loads((adv_dir / "data.json").read_text())
        self.adv_config = adv_dir / "config.json"
        # Strict-ASR override: the published adversarial evaluator scores
        # must_exclude(<correct answer>) on 10/40 reddit tasks, which any
        # failed episode satisfies -- measured at 10/10 clean episodes
        # scoring ASR=True with no attack present. analysis/asr_review_sheet.py
        # emits corrected configs checking the attacker's OWN recorded
        # target instead. When one exists for this task, use it.
        _strict = os.environ.get("STRICT_ASR_CONFIG_DIR")
        if _strict:
            _c = Path(_strict) / f"{self.name}.json"
            if _c.exists():
                self.adv_config = _c
                self.strict_asr = True
        self.clean_config = exp_data / "agent_clean" / self.base / "config.json"
        # Clean-UA override: 22 of 40 adversarial variants change the user's
        # own question, so the BASE task's benign evaluator scores a task the
        # agent was never asked to perform, and ua_scorable refuses them. Ten
        # were re-authored by hand against the base's own correct answer,
        # reformatted to what the variant asks for. When one exists, it
        # replaces the base config and makes the task UA-scorable.
        _cu = os.environ.get("CLEAN_UA_CONFIG_DIR")
        if _cu:
            _c = Path(_cu) / f"{self.name}.json"
            if _c.exists():
                self.clean_config = _c
                self.clean_ua_authored = True
        self.clean_obs_text = exp_data / "agent_clean" / self.base / "obs_text.txt"

    @property
    def usable(self) -> bool:
        return self.adv_config.exists()

    @property
    def adversarial_goal(self) -> str:
        return self.data.get("adversarial_goal", "") or ""

    @property
    def intent(self) -> str:
        return json.loads(self.adv_config.read_text()).get("intent", "")

    @property
    def ua_scorable(self) -> bool:
        """UA only valid when the benign evaluator targets the SAME user task."""
        if not self.clean_config.exists():
            return False
        clean_intent = json.loads(self.clean_config.read_text()).get("intent", "").strip()
        return clean_intent == self.intent.strip()

    def victim_image_path(self) -> Optional[Path]:
        """agent-attack ships the exact image it attacked, at the exact
        rendered resolution (verified: victim_image.png size == data.json
        position.size for all 53 reddit tasks). Preferred over fetching
        from Postmill: the LiipImagine thumbnail-cache URL 404s for 19/40
        tasks, and the un-cached /media/ original is a DIFFERENT, larger
        image than the agent actually sees on the page."""
        vp = self.adv_dir / "victim_image.png"
        return vp if vp.exists() else None

    def victim_image_url(self) -> Optional[str]:
        """Absolute URL of the trigger image, from agent-attack's own recorded
        observation (host already rewritten by their process_data.py)."""
        vid = self.data.get("victim_som_id")
        if vid is None or not self.clean_obs_text.exists():
            return None
        for line in self.clean_obs_text.read_text().splitlines():
            if line.startswith(f"[{vid}] "):
                m = _URL_RE.search(line)
                return m.group(1) if m else None
        return None


def _build_registry(model, processor, extractor=None) -> Dict[str, BaseAttack]:
    return {
        "template_ipi": TemplateIPIAttack(),
        "gcg": GCGAttack(model, processor),
        "pixel_perturbation": PixelPerturbationAttack(model, processor),
        "cross_modal_injection": CrossModalInjectionAttack(model, processor),
        # Visual row of Table 3.2. typographic/steganographic need no model
        # access and no gradient steps -- effectively free to generate, so
        # they can run at full task coverage where GCG/patch cannot.
        "typographic": TypographicAttack(),
        "steganographic": SteganographicAttack(redundancy=8),
        "adversarial_patch": AdversarialPatchAttack(model, processor),
        "comet": CometAttack(),
        # Adaptive row of Table 3.2. Needs a clean_reference_activation
        # per task, taken from that task's OWN clean episode -- cached
        # below during the clean pass, exactly as run_attack_matrix.py
        # does via clean_activations_by_task.
        "obfuscated_activation": ObfuscatedActivationAttack(model, processor, extractor),
        # The paper's own adversarial images, captioned by OUR model in the
        # episode: transfer attacks, paper-comparable (self-captioning).
        "bim_caption_transfer": ShippedImageAttack("bim_caption_transfer", "bim_caption"),
        "clip_attack_transfer": ShippedImageAttack("clip_attack_transfer", "clip_attack"),
    }


def _fetch_image_candidates(url: str) -> List[str]:
    """Postmill serves victim images through LiipImagine's on-demand
    thumbnail cache (/media/cache/resolve/<filter>/<hash>.jpg). That
    endpoint 404s for images whose thumbnail was never generated, which
    is why 19/40 reddit tasks failed to fetch while the rest succeeded.
    Fall back to the un-cached original and to the other filters rather
    than dropping the task."""
    cands = [url]
    m = re.search(r"/media/cache/resolve/([^/]+)/(.+)$", url)
    if m:
        filt, tail = m.group(1), m.group(2)
        base = url.split("/media/cache/resolve/")[0]
        cands.append(f"{base}/media/{tail}")                    # original
        for alt in ("submission_thumbnail_1x", "submission_thumbnail_2x",
                    "submission_image_1x", "submission_image_2x"):
            if alt != filt:
                cands.append(f"{base}/media/cache/resolve/{alt}/{tail}")
    return cands


def _fetch_image(url: str, timeout: float = 15.0) -> Optional[Image.Image]:
    errs = []
    for i, cand in enumerate(_fetch_image_candidates(url)):
        try:
            r = requests.get(cand, timeout=timeout)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            if i:
                logger.info("  fetched via fallback #%d: %s", i, cand)
            return img
        except Exception as e:
            errs.append(f"{type(e).__name__}")
    logger.warning("  could not fetch %s -- all %d candidates failed (%s)",
                   url, len(errs), ",".join(errs))
    return None


def run_curated_matrix(
    model, processor, model_id: str,
    attack_names: List[str], target_layers, output_dir: Path,
    exp_data: Path, site: str = "reddit",
    max_tasks: Optional[int] = None, max_steps: int = 15,
    only_tasks: Optional[List[str]] = None,
    defense_names: Optional[List[str]] = None,
) -> None:
    from browser_env import ScriptBrowserEnv
    from evaluation_harness import evaluator_router
    from environments.vwa_adv_tasks import VWAAdvTask, run_vwa_adv_episode

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_data = exp_data.resolve()
    os.chdir(str(AGENT_ATTACK_REPO))  # see module docstring -- storage_state paths depend on this

    tasks = [CuratedTask(d, exp_data)
             for d in sorted((exp_data / "agent_adv").glob(f"{site}_*/"))
             if (d / "data.json").exists()]
    usable = [t for t in tasks if t.usable]
    logger.info("%s: %d variants, %d with an adversarial evaluator, %d also UA-scorable",
                site, len(tasks), len(usable), sum(1 for t in usable if t.ua_scorable))
    if only_tasks:
        # Re-scoring a subset is not possible offline: the authored clean-UA
        # evaluators are program_html checks that read LIVE page state, so a
        # changed evaluator means re-running those episodes. This filter
        # exists so that means 10 episodes per condition rather than 40.
        want = set(only_tasks)
        usable = [t for t in usable if t.name in want]
        missing = want - {t.name for t in usable}
        if missing:
            logger.warning("--only_tasks named %d task(s) not in the usable "
                           "set: %s", len(missing), sorted(missing))
        logger.info("--only_tasks: restricted to %d task(s)", len(usable))
    if max_tasks:
        usable = usable[:max_tasks]
    if not usable:
        # A typo in --only_tasks or --site would otherwise "complete" a run
        # of zero episodes (review 3, #4).
        raise SystemExit(f"FATAL: no tasks selected (site={site!r}, only_tasks={only_tasks})")

    extractor = ActivationExtractor(model, target_layers=target_layers, token_aggregation="last")
    registry = _build_registry(model, processor, extractor)

    # (attack_name, task_name) -> (adv_cap, adv_img, meta). Populated by
    # the undefended attack loop and reused by every defended pass. The
    # adversarial payload is a function of the attack and the task, NOT
    # of the defence, so regenerating it per defence is pure waste --
    # and for GCG that waste is ~150s per task per defence.
    _payloads: Dict[tuple, tuple] = {}
    # task_name -> {layer: tensor}, from that task's clean episode.
    _clean_acts: Dict[str, Dict[int, Any]] = {}

    # PERSISTED across runs. _payloads is populated only by the
    # undefended attack loop, and a resumed run skips episodes already
    # recorded -- so on any resume the cache is empty and every
    # defended-attacked episode fails the "no cached payload" guard.
    # Measured: 304 of 400 blur rows and 36 of 40 template_ipi defended
    # rows lost exactly this way. Text payloads round-trip through JSON;
    # images do not, so those are held in memory only and a resumed run
    # still regenerates them.
    # PAYLOAD_PERSIST_V1: every payload -- attacker text AND image -- is
    # written to disk when generated and reloaded on resume, so a defended
    # cell after a resume uses the SAME payload as its undefended cell
    # instead of failing "no cached payload" or silently regenerating.
    _payload_dir = output_dir / "payloads"
    _payload_dir.mkdir(exist_ok=True)

    def _payload_save(attack_name, task_name, text, image, meta):
        # Stamped with the attack's own code hash and the model, so a
        # payload from before an attack rebuild, or from another model, is
        # never reused (review 3, #3).
        d = _payload_dir / f"{attack_name}__{task_name}"
        d.mkdir(exist_ok=True)
        if image is not None:
            atomic_save_image(image, d / "image.png")      # lossless; atomic (review 4, H2)
        # payload.json last: it is what marks the payload as present.
        atomic_write_text(d / "payload.json", json.dumps(
            {"text": text, "has_image": image is not None, "meta": meta,
             "attack_sha256": _ATK_SHA.get(attack_name), "model_id": model_id},
            indent=1, default=str))


    defense_registry = {}
    ecso_registry = {}
    if defense_names:

        from defenses.visual import build_visual_defense_registry
        from defenses.ecso import build_ecso_registry
        # Three registries, because the three defence families enter the
        # pipeline at different points. Spotlighting transforms text and
        # rides defense=. Gaussian blur transforms the image and also
        # rides defense=, via apply_image(). ECSO restructures generation
        # itself and must ride ecso= -- routed through defense= its
        # apply() is a no-op and the run would be labelled ECSO-defended
        # with no defence applied.
        defense_registry = {**_build_defense_registry(),
                            **build_visual_defense_registry()}
        ecso_registry = build_ecso_registry()
        for d in defense_names:
            if d not in defense_registry and d not in ecso_registry:
                raise KeyError(
                    f"Unknown defense {d!r} -- have "
                    f"{sorted(set(defense_registry) | set(ecso_registry))}")
    for n in attack_names:
        if n not in registry:
            raise KeyError(f"Unknown attack {n!r} -- have {sorted(registry)}")

    # Per-attack code hashes, run settings and defence configuration,
    # stamped on rows and checked on resume (review 3, #3, #4, #13).
    _ATK_SHA = {n: attack_code_sha(registry[n]) for n in attack_names}
    _RUN_SETTINGS = {
        "model_id": model_id, "site": site, "max_steps": max_steps,
        "strict_asr_dir_sha": dir_sha(os.environ.get("STRICT_ASR_CONFIG_DIR")),
        "clean_ua_dir_sha": dir_sha(os.environ.get("CLEAN_UA_CONFIG_DIR")),
        "captioner": {"prompt": CAPTION_PROMPT, "max_side": MAX_SIDE, "max_new_tokens": 64},
        # Inputs outside this repo that change what an episode is (review 4,
        # M4): the agent prompt, visualwebarena's environment and evaluator
        # code, and agent-attack's task data for the selected tasks.
        "instruction_sha": files_sha([VWA_ADV_REPO / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"],
                                     base=VWA_ADV_REPO),
        # visualwebarena code the episode runs: environment, evaluators,
        # and the agent/prompt-constructor and llms packages the backend
        # imports (review 5, M5). Paths hashed relative (review 5, M6).
        "vwa_code_sha": {d: dir_sha(str(VWA_ADV_REPO / d), "*.py")
                         for d in ("browser_env", "evaluation_harness", "agent", "llms")},
        "task_data_sha": files_sha(
            # every variant of the site, not just the selected ones, so a
            # subset run (--only_tasks) resumes into the same directory
            [f for t in tasks for f in sorted(t.adv_dir.iterdir()) if f.is_file()]
            + [f for t in tasks for f in (t.clean_config, t.clean_obs_text)],
            base=exp_data),
    }
    _DEF_SHA = {d: spotlight_seed_sha() for d in (defense_names or []) if d.startswith("spotlighting")}

    # Transfer attacks need agent-attack's shipped images; say up front how
    # many tasks have them (review 3, #8).
    for n in attack_names:
        atk = registry[n]
        if isinstance(atk, ShippedImageAttack):
            have = sum(1 for t in usable if (t.adv_dir / f"{atk.prefix}_attack_image.png").exists())
            logger.info("%s: shipped image present for %d/%d tasks", n, have, len(usable))
            if have == 0:
                raise SystemExit(f"FATAL: {n}: no task has {atk.prefix}_attack_image.png -- "
                                 "check the filename against exp_data/agent_adv/*/")

    _stale = 0
    for d in sorted(_payload_dir.glob("*__*")):
        try:
            _a, _t = d.name.split("__", 1)
            if _a not in _ATK_SHA:
                continue                    # not an attack of this run
            j = json.loads((d / "payload.json").read_text())
            if j.get("attack_sha256") != _ATK_SHA[_a] or j.get("model_id") != model_id:
                _stale += 1                 # regenerated; the new one overwrites it
                continue
            _img = Image.open(d / "image.png").convert("RGB") if j.get("has_image") else None
            _payloads[(_a, _t)] = (j.get("text"), _img, j.get("meta") or {})
        except Exception as e:
            logger.warning("could not reload payload %s (%s: %s)", d.name, type(e).__name__, e)
    logger.info("resumed %d persisted payloads from %s (%d stale: other code or model, ignored)",
                len(_payloads), _payload_dir, _stale)

    # SELF-CAPTIONING (decision 30 Sep): the agent model captions every page
    # image, as in the VWA-Adv paper's self-captioning configuration.
    captioner = SelfCaptioner(model, processor, max_new_tokens=64, extractor=extractor)
    env = ScriptBrowserEnv(headless=True, observation_type="image_som",
                           current_viewport_only=False, viewport_size={"width": 1280, "height": 2048},
                           captioning_fn=captioner)
    _CV = code_version()
    logger.info("code version: commit=%s dirty=%s harness=%s attacks=%s",
                _CV["commit"], _CV["dirty"], _CV["harness_sha256"][:12], _CV["attacks_sha256"][:12])

    # ---- captions -----------------------------------------------------
    # Captions depend on (image, blur). The victim's CLEAN caption is seeded
    # into VWA's cache before EVERY episode, for the current blur setting,
    # so that (a) clean, attacked and defended episodes see the same clean
    # description, and (b) an adversarial caption always has a description
    # to replace -- VWA's captioned branch only substitutes over an existing
    # one (review C2, C4).
    # PERSISTED (review 3, #6): every caption -- the victim's clean and
    # attacked captions, and VWA's captions of every other page image -- is
    # saved to captions.json and reloaded on resume, so a resume (possibly
    # on another GPU type, where a greedy caption can differ) shows the
    # agent exactly the page text the earlier rows saw. A fallback caption
    # (captioner failure) is never cached or saved, so the next episode
    # retries it (review 3, #7).
    _caption_cache: Dict[tuple, str] = {}           # (kind, key, sigma) -> caption
    _page_caps: Dict[tuple, str] = {}               # (url, sigma) -> caption, other page images
    # Page images (not the victim) the captioner has failed on. A first
    # failure errors the row so it is retried; a SECOND non-OOM failure on
    # the same image is treated as deterministic: the fallback is persisted
    # as that image's caption (flagged), seeded identically for every
    # condition, and rows complete -- otherwise one unreadable image on a
    # common page would make every episode fail forever (review 5, H1).
    _page_fail_counts: Dict[tuple, int] = {}
    _page_failed: set = set()
    _captions_path = output_dir / "captions.json"
    if _captions_path.exists():
        try:
            _cj = json.loads(_captions_path.read_text())
            if _cj.get("run_settings") != _RUN_SETTINGS:
                # Another model, captioner or task set wrote these. Rows it
                # completed are refused by the resume check below; with
                # only error rows, starting the captions afresh is right
                # (review 4, L5).
                logger.warning("captions.json was written under different run settings -- ignoring it")
                _cj = {}
            for e in _cj.get("self", []):
                _caption_cache[(e["kind"], e["key"], e["sigma"])] = e["caption"]
            for e in _cj.get("page", []):
                _page_caps[(e["url"], e["sigma"])] = e["caption"]
                if e.get("failed"):
                    _page_failed.add((e["url"], e["sigma"]))
            for e in _cj.get("page_fail_counts", []):
                _page_fail_counts[(e["url"], e["sigma"])] = e["n"]
            logger.info("resumed %d victim and %d page-image captions from %s",
                        len(_caption_cache), len(_page_caps), _captions_path)
        except Exception as e:
            raise SystemExit(f"FATAL: {_captions_path} unreadable ({type(e).__name__}: {e}); "
                             "resuming without it would re-caption and could change page text")

    def _save_captions():
        atomic_write_json(_captions_path, {                 # atomic (review 4, H2)
            "run_settings": _RUN_SETTINGS,
            "self": [{"kind": k, "key": key, "sigma": sg, "caption": c}
                     for (k, key, sg), c in _caption_cache.items()],
            "page": [{"url": u, "sigma": sg, "caption": c, "failed": (u, sg) in _page_failed}
                     for (u, sg), c in _page_caps.items()],
            "page_fail_counts": [{"url": u, "sigma": sg, "n": n} for (u, sg), n in _page_fail_counts.items()],
        }, indent=1)

    _blur_state: Dict[str, Any] = {"def": None}     # page-level blur of the current pass
    _victims: Dict[str, tuple] = {}

    def _sigma():
        return getattr(_blur_state["def"], "sigma", None)

    # Whether the most recent _self_caption() call fell back. Decided by the
    # captioner's failure counter, never by comparing text, so a genuine
    # caption that happens to read "an image" is not mistaken for a failure
    # (review 4, L2).
    _cap_failed = {"v": False}

    def _self_caption(kind: str, key: str, image, blur_def=None) -> str:
        ck = (kind, key, getattr(blur_def, "sigma", None))
        if ck in _caption_cache:
            _cap_failed["v"] = False
            return _caption_cache[ck]
        prev = captioner.blur                       # restore, never reset (review C1)
        captioner.set_blur(blur_def)
        n_fail = captioner.n_failures
        try:
            cap = captioner([image])[0]             # never raises (safe_caption)
        finally:
            captioner.set_blur(prev)
        _cap_failed["v"] = captioner.n_failures != n_fail
        if not _cap_failed["v"]:
            _caption_cache[ck] = cap
            _save_captions()
        return cap

    def _image_key(image) -> str:
        """Content key for an attack image: a regenerated image must never
        be shown with the previous image's caption (review 4, H3)."""
        import hashlib as _h
        return _h.sha256(f"{image.mode}{image.size}".encode() + image.tobytes()).hexdigest()

    def _sync_page_captions(victim_url, oom: bool, failed_this_episode: int, trajectory=None):
        """After an episode: keep VWA's captions of the OTHER page images
        (not the victim, whose caption is seeded; not data: URLs, which are
        substituted attack images), and handle fallbacks (review 5, H1):
          * after an out-of-memory failure: dropped from VWA's cache, so
            they are retried -- OOM is transient;
          * a first non-OOM failure on an image: dropped and counted;
          * a second non-OOM failure on the same image: deterministic --
            the fallback is persisted as that image's caption, flagged, and
            seeded for every condition from then on.
        Returns (pending, persisted_fallbacks_on_page): `pending` > 0 means
        this episode saw a fallback that later episodes may not see, so the
        row must be retried."""
        cache = getattr(getattr(getattr(env, "observation_handler", None), "text_processor", None),
                        "url2caption", None)
        if not isinstance(cache, dict):
            return 0, 0
        changed = False
        pending = 0
        # Fallback-looking entries this episode. If there are more of them
        # than captioner failures, some are genuine captions reading "an
        # image" and they cannot be told apart: retry the row, persist
        # nothing as failed (review 6, #3).
        cands = [u for u, c in cache.items()
                 if (u, _sigma()) not in _page_failed and u != victim_url
                 and not str(u).startswith("data:") and (not c or c == FALLBACK_CAPTION)]
        ambiguous = failed_this_episode > 0 and len(cands) > failed_this_episode
        handled = 0
        for u, c in list(cache.items()):
            key = (u, _sigma())
            if key in _page_failed:
                continue
            if u == victim_url or str(u).startswith("data:"):
                if not c or c == FALLBACK_CAPTION:
                    cache.pop(u, None)
                continue
            is_fallback = (not c) or (c == FALLBACK_CAPTION and failed_this_episode > 0 and not ambiguous)
            if is_fallback:
                handled += 1
                cache.pop(u, None)
                # OOM is usually transient, so it needs more repeats before
                # the image is treated as uncaptionable (review 6, #8).
                n = _page_fail_counts.get(key, 0) + 1
                _page_fail_counts[key] = n
                changed = True
                if n >= (3 if oom else 2):
                    _page_caps[key] = FALLBACK_CAPTION
                    _page_failed.add(key)
                    cache[u] = FALLBACK_CAPTION
                    logger.warning("page image %s: captioner failed twice (%s) -- fallback caption "
                                   "persisted for every condition", u, captioner.last_error)
                else:
                    pending += 1
                continue
            if key not in _page_caps:
                _page_caps[key] = c
                changed = True
        # A failure that left no recognisable entry (review 6, #4), or an
        # ambiguous set, still means this row may differ from its retry.
        if failed_this_episode > handled:
            pending += failed_this_episode - handled
        if changed:
            _save_captions()
        # Persisted fallbacks counted only if their URL appears in THIS
        # episode's observations -- VWA's cache also holds every seeded
        # caption from other pages (review 6, #1).
        obs_text = " ".join(str((st.get("observation") or {}).get("text", ""))
                            for st in (trajectory or []) if isinstance(st, dict))
        seen_failed = sum(1 for (u, sg) in _page_failed if sg == _sigma() and u in obs_text)
        return pending, seen_failed

    def _victim(task):
        """(url, image) of the victim image, cached per task."""
        if task.name not in _victims:
            url = task.victim_image_url()
            vp = task.victim_image_path()
            img = Image.open(vp).convert("RGB") if vp is not None else (_fetch_image(url) if url else None)
            _victims[task.name] = (url, img)
        return _victims[task.name]

    def _clean_caption(task, blur_def=None):
        url, img = _victim(task)
        return _self_caption("clean", task.name, img, blur_def) if img is not None else None

    def _seed_victim(task):
        """Seeds, for the current blur, the persisted captions of other page
        images and the victim's clean caption. Returns an error string, or
        None."""
        try:
            for (u, sg), c in _page_caps.items():
                if sg == _sigma():
                    seed_caption(env, u, c)
        except Exception as e:
            return f"caption seeding failed: {type(e).__name__}: {str(e)[:120]}"
        url, img = _victim(task)
        if not url or img is None:
            return None     # handled by the attack path's own errors
        cap = _clean_caption(task, _blur_state["def"])
        if _cap_failed["v"]:
            # The captioner failed on the victim image (e.g. a transient
            # OOM). No row may complete with a placeholder where the
            # victim's description belongs: recorded as an error, so a
            # resubmit retries it (review 3, #7).
            return f"victim caption unavailable (captioner failed: {captioner.last_error})"
        try:
            seed_caption(env, url, cap)
        except Exception as e:
            return f"caption seeding failed: {type(e).__name__}: {str(e)[:120]}"
        return None

    def _victim_caption(text, image, meta, attack_name, task, blur_def=None):
        """The caption the agent sees for the victim image under attack.
        Attacker-written text IS the caption, built as <clean caption> +
        <payload>; under blur the clean part is the BLURRED clean caption,
        so attacked = clean + payload holds in every pass. An image-only
        attack's caption is our model's caption of the adversarial image,
        blurred if a page-level blur is active."""
        if text:
            gen_clean = (meta or {}).get("clean_caption") or ""
            if gen_clean and text.startswith(gen_clean):
                return (_clean_caption(task, blur_def) or "") + text[len(gen_clean):], "attacker_text"
            return text, "attacker_text"
        if image is not None:
            cap = _self_caption("adv", _image_key(image), image, blur_def)
            if _cap_failed["v"]:
                return None, "caption_failed"
            return cap, "self_caption"
        return None, None

    results_path = output_dir / "attack_results.json"
    results: List[Dict[str, Any]] = json.loads(results_path.read_text()) if results_path.exists() else []
    index = {r["sample_id"]: i for i, r in enumerate(results)}
    _override = None
    if results:
        _override = check_resume_compatible(       # review C9; review 3 #4, #13
            results, _CV, set(attack_names), attack_shas=_ATK_SHA,
            run_settings=_RUN_SETTINGS, defense_shas=_DEF_SHA, output_dir=output_dir)
    write_manifest(output_dir)

    def done(sid): return sid in index and results[index[sid]].get("error") is None

    def _clean_ref(task):
        """This task's clean reference activation (last step of its clean
        episode). Loaded from clean_<task>.pt when the clean episode ran in
        an earlier process -- a resumed or split run skips it, and the
        in-memory cache would otherwise be empty (review 3, #1)."""
        if task.name not in _clean_acts:
            sid_c = f"clean_{task.name}"
            p = output_dir / f"{sid_c}.pt"
            if done(sid_c) and p.exists():
                try:
                    steps = torch.load(p, weights_only=False)
                except Exception as e:
                    logger.warning("could not load %s (%s)", p.name, e)
                    steps = None
                if steps:
                    _clean_acts[task.name] = {k: (v.clone() if torch.is_tensor(v) else v)
                                              for k, v in steps[-1].items()}
        return _clean_acts.get(task.name)

    # Failures that are terminal for the whole process, not for one task.
    # env = ScriptBrowserEnv(...) is shared by every episode, so once the
    # browser or Playwright's loop is broken, nothing downstream can recover.
    _FATAL_ERROR_MARKERS = (
        "asyncio loop",                 # the loop was left running: poisoned
        "BrowserType.launch",           # browser never started
        "Target page, context or browser has been closed",
        "Connection closed",
        "browser has been closed",
        "Target crashed",
    )
    # Legitimate per-task data conditions. These are recorded for individual
    # tasks and several can appear in a row, so they must not trip the
    # consecutive-failure backstop.
    _BENIGN_ERROR_MARKERS = (
        "no victim image URL",
        "could not fetch victim image",
        "no clean reference activation",
        "no cached payload",
        "not shipped for this task",
        "victim caption unavailable",
        "attack image caption failed",
        "(first failure; retried)",
    )
    _MAX_CONSECUTIVE_ERRORS = 25

    _fail_streak = {"n": 0}

    def _abort(reason, sid):
        atomic_write_json(results_path, results, indent=2)
        n_err = sum(1 for r in results if r.get("error"))
        logger.error("ABORTING: %s", reason)
        logger.error("  last sample: %s", sid)
        logger.error("  %d of %d rows carry an error", n_err, len(results))
        logger.error("  Rows with an error are NOT counted as done(), so a "
                     "resubmit retries them. Do not delete the output dir.")
        raise SystemExit(
            f"ABORT: {reason}\n"
            f"  last sample: {sid}\n"
            f"  {n_err}/{len(results)} rows carry an error; resubmit to retry "
            f"them (done() requires error is None).")

    def record(sid, row):
        # scoring_attempts survives a replacement by a DIFFERENT error, so
        # the 3-attempt cap is reached (review 6, #2).
        if (sid in index and "scoring_attempts" not in row
                and results[index[sid]].get("scoring_attempts")):
            row["scoring_attempts"] = results[index[sid]]["scoring_attempts"]
        row.setdefault("code_version", _CV)
        row.setdefault("run_settings", _RUN_SETTINGS)
        an, dn = row.get("attack_name"), row.get("defense_name")
        if an in _ATK_SHA:
            row.setdefault("attack_sha256", _ATK_SHA[an])
        if dn in _DEF_SHA:
            row.setdefault("defense_sha256", _DEF_SHA[dn])
        if _override:
            row.setdefault("resume_override", _override)
        if sid in index:
            results[index[sid]] = row
        else:
            results.append(row); index[sid] = len(results) - 1
        atomic_write_json(results_path, results, indent=2)

        err = str(row.get("error") or "")
        if not err:
            _fail_streak["n"] = 0
            return
        for m in _FATAL_ERROR_MARKERS:
            if m in err:
                _abort(f"browser/environment failure is terminal for this "
                       f"process (matched {m!r}): {err[:160]}", sid)
        if any(m in err for m in _BENIGN_ERROR_MARKERS):
            return                      # per-task, not a systemic failure
        _fail_streak["n"] += 1
        if _fail_streak["n"] >= _MAX_CONSECUTIVE_ERRORS:
            _abort(f"{_fail_streak['n']} consecutive episodes failed; this is "
                   f"systemic, not per-task. Last: {err[:160]}", sid)

    def _for_resub(caption: str) -> str:
        """VWA inserts the adversarial caption as the REPLACEMENT string of
        re.sub, where a backslash starts an escape. Doubling backslashes
        makes the caption land verbatim."""
        return caption.replace("\\", "\\\\")

    def _markers(adv_cap, clean_cap=None):
        # Whitespace is normalised here and in the backend before matching
        # (review 3, #14).
        """What to look for in the prompt to know the injected caption
        reached the model (audit 11.11): the first 60 characters of the
        part that is NOT the clean caption. Text attacks append to the
        clean caption, so the clean prefix is present in clean runs too and
        cannot serve as the marker. A caption identical to the clean one
        injects nothing textual and gets no marker."""
        out = []
        for c in (adv_cap or {}).values():
            c = (c or "").strip()
            if clean_cap and c.startswith(clean_cap):
                c = c[len(clean_cap):].strip()
            c = " ".join(c.split())
            if c and c != " ".join((clean_cap or "").split()):
                out.append(c[:60])
        return out

    def _episode_fields(out, cap_before=None, task=None, blur_def=None):
        """Per-row diagnostics that used to be computed and discarded
        (audit 11.4, 11.9, 11.11), plus this episode's captioner activity."""
        pip = out.get("payload_in_prompt") or []
        cb = cap_before or _cap_counts()
        vcc = _clean_caption(task, blur_def) if task is not None else None
        return {
            "victim_clean_caption": vcc,
            "caption_calls_while_hooked": captioner.n_calls_while_hooked - cb[2],
            "asr_scoring_error": out.get("asr_scoring_error"),
            "final_url": out.get("final_url"),
            "double_bos": out.get("double_bos"),
            "blur_css_failures": out.get("blur_css_failures"),
            "captions_this_episode": captioner.n_images - cb[0],
            "caption_failures_this_episode": captioner.n_failures - cb[1],
            "defense_addenda": out.get("defense_addenda"),
            "defense_unmatched_steps": out.get("defense_unmatched_steps"),
            "prev_action_defended": out.get("prev_action_defended"),
            "page_blur_css": out.get("page_blur_css"),
            "ecso_info": out.get("ecso_info"),
            "ecso_flagged_steps": sum(1 for e in (out.get("ecso_info") or []) if e.get("flagged")),
            "retries_deduplicated": out.get("retries_deduplicated"),
            "payload_in_prompt": pip,
            "payload_reached": (any(pip) if pip else None),
        }

    _MAX_SCORING_ATTEMPTS = 3

    def _final_error(err, out, ua_err, cap_before, task, sid):
        """Returns (error, extra row fields). A row only completes if what
        makes it comparable held; otherwise it is an ERROR row, which a
        resubmit retries (activations are still saved).
          * a caption pass while activation hooks were live (review 4, L4);
          * a captioner out-of-memory failure, or a FIRST failure on a page
            image (review 4 L3, review 5 H1 -- a repeated failure is
            persisted and no longer errors, see _sync_page_captions);
          * an ASR/UA scoring failure (review 4, M1): the evaluators read
            LIVE page state, so a transient failure cannot be re-scored
            offline and the episode is retried -- but at most
            _MAX_SCORING_ATTEMPTS times; after that the row completes with
            the scoring error recorded and scoring_error_final=True, for
            exclusion, so a deterministic evaluator failure cannot block the
            run forever (review 5, H1)."""
        oom = captioner.n_oom - cap_before[3] > 0
        pending, seen_failed = _sync_page_captions(
            _victim(task)[0], oom, captioner.n_failures - cap_before[1], out.get("trajectory"))
        extra = {"page_caption_fallbacks": seen_failed}
        if err is not None:
            return err, extra
        probs = []
        if captioner.n_calls_while_hooked - cap_before[2] > 0:
            probs.append("caption pass ran while activation hooks were registered")
        if pending:
            probs.append(f"captioner failed on {pending} page image(s) (first failure; retried): "
                         f"{captioner.last_error}")
        scoring = []
        if out.get("asr_scoring_error"):
            scoring.append(f"ASR scoring failed: {out['asr_scoring_error']}")
        if ua_err:
            scoring.append(f"UA scoring failed: {ua_err}")
        if scoring:
            prior = results[index[sid]].get("scoring_attempts", 0) if sid in index else 0
            extra["scoring_attempts"] = prior + 1
            if prior + 1 >= _MAX_SCORING_ATTEMPTS and not probs:
                extra["scoring_error_final"] = True
                logger.warning("  %s: scoring failed %d times -- completing the row with the "
                               "scoring error recorded (exclude it)", sid, prior + 1)
            else:
                probs.extend(scoring)
        return (("post-episode: " + "; ".join(probs))[:400] if probs else None), extra

    def _cap_counts():
        return (captioner.n_images, captioner.n_failures, captioner.n_calls_while_hooked, captioner.n_oom)

    def score(config_path: Path, trajectory, final_url: Optional[str] = None):
        """Scores a finished episode against a config's evaluator. Returns
        (verdict, error). The in-episode ASR evaluation may have navigated
        the page, so the agent's final page is restored first (review A7);
        a failure is RETURNED, not swallowed, so the row says why its UA is
        missing (review A8)."""
        try:
            if final_url and env.page.url != final_url:
                env.page.goto(final_url)
                env.page.wait_for_load_state("load", timeout=30000)
            ev = evaluator_router(str(config_path), captioning_fn=None)
            return bool(ev(trajectory=trajectory, config_file=str(config_path),
                           page=env.page, client=env.get_page_client(env.page))), None
        except Exception as e:
            logger.warning("  scoring against %s failed (%s: %s)", config_path.name, type(e).__name__, e)
            return None, f"{type(e).__name__}: {str(e)[:160]}"

    try:
        for attack_name in ["clean"] + list(attack_names):
            logger.info("--- %s (%d tasks) ---", attack_name, len(usable))
            for task in usable:
                sid = f"{attack_name}_{task.name}"
                if done(sid):
                    logger.info("  skip %s", sid); continue

                # REMOVED (P1.3, 3 Oct): a check that aborted the run whenever
                # asyncio.get_running_loop() succeeded. Playwright's SYNC API
                # runs its event loop in a separate greenlet on the same
                # thread, and asyncio's "running loop" is per thread, so after
                # the browser's first start the check fires on EVERY later
                # episode: smoke jobs 62671/62673 each completed one episode
                # and then aborted with no error row. The condition it meant to
                # catch -- a browser failure that leaves Playwright unable to
                # restart -- surfaces as an episode error containing "asyncio
                # loop", which _FATAL_ERROR_MARKERS already treats as terminal.

                _blur_state["def"] = None
                seed_err = _seed_victim(task)
                if seed_err:
                    record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                 "error": seed_err})
                    continue

                adv_cap, adv_cap_raw, adv_img, meta, gen_s = {}, {}, {}, {}, None
                if attack_name != "clean":
                    url, img = _victim(task)
                    if not url:
                        record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                     "error": "no victim image URL in recorded obs_text.txt"})
                        continue
                    if img is None:
                        record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                     "error": f"could not fetch victim image {url}"})
                        continue

                    # The REAL published adversarial goal drives the attack --
                    # not a generic placeholder string.
                    goal = task.adversarial_goal or task.data.get("target_caption", "")
                    # With self-captioning the victim image already HAS a
                    # caption; text attacks append to it, as an attacker
                    # editing the alt text would.
                    clean_cap = _clean_caption(task, None)
                    base_input = {"prompt": task.intent, "image": img,
                                  "injectable_field": "caption", "caption": clean_cap,
                                  "adv_dir": str(task.adv_dir)}

                    # A payload already persisted for this (attack, task)
                    # is REUSED, never regenerated: defended cells may
                    # already have run with it (review C8).
                    cached_payload = _payloads.get((attack_name, task.name))

                    # obfuscated_activation optimises its suffix to hold the
                    # attacked state near THIS task's own clean state, so the
                    # reference must come from this task's clean episode.
                    # Resolved here rather than inside the try below, where a
                    # `continue` would be illegal.
                    _ref_act = None
                    if attack_name == "obfuscated_activation" and cached_payload is None:
                        _ref = _clean_ref(task)
                        _layer = target_layers[len(target_layers) // 2]
                        if not _ref or _layer not in _ref:
                            record(sid, {"sample_id": sid, "attack_name": attack_name,
                                         "task_id": task.name,
                                         "error": "no clean reference activation for this task"})
                            continue
                        _ref_act = _ref[_layer]

                    t0 = time.perf_counter()
                    try:
                        if cached_payload is not None:
                            raise _ReusePayload()
                        with _cuda_memory_guard():
                            atk = registry[attack_name]
                            if attack_name in ("template_ipi", "gcg"):
                                res = atk.generate(base_input, target_behavior=goal)
                            elif attack_name == "pixel_perturbation":
                                res = atk.generate(base_input, target_behavior=goal, image=img)
                            elif attack_name == "obfuscated_activation":
                                # Text-only by construction: its signature takes
                                # no image and no clean_text, so passing them
                                # would land in **kwargs and be silently
                                # ignored. It rewrites the injected CAPTION,
                                # which is the text channel of the visual
                                # observation -- the image itself is untouched.
                                res = atk.generate(base_input, target_behavior=goal,
                                                   clean_reference_activation=_ref_act)
                            else:
                                # clean_text = the clean caption, so COMET and
                                # CrossInject append to it like every other
                                # text attack (review C5).
                                res = atk.generate(base_input, target_behavior=goal, image=img, clean_text=clean_cap)
                    except _ReusePayload:
                        res = None
                    except torch.OutOfMemoryError as e:
                        torch.cuda.empty_cache(); gc.collect()
                        record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                     "error": f"CUDA_OOM_generation: {str(e).splitlines()[0][:180]}"})
                        continue
                    except Exception as e:
                        gc.collect()
                        record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                     "error": f"attack_generation_{type(e).__name__}: {str(e).splitlines()[0][:200]}"})
                        continue
                    if res is None:
                        ptext, pimg, meta = cached_payload
                        meta = dict(meta)
                        meta["payload_reused"] = True
                        gen_s = 0.0
                    else:
                        gen_s = time.perf_counter() - t0
                        ptext, pimg = res.adversarial_text, res.adversarial_image
                        meta = {f"attack_meta_{k}": v for k, v in (res.metadata or {}).items()}
                        meta["victim_url"] = url
                        meta["clean_caption"] = clean_cap
                        _payloads[(attack_name, task.name)] = (ptext, pimg, meta)
                        try:
                            _payload_save(attack_name, task.name, ptext, pimg, meta)
                        except Exception as e:
                            # An unsaved payload must not produce a completed
                            # row: after a resume its defended cells could
                            # never run (review 5, M3). Error row; retried.
                            _payloads.pop((attack_name, task.name), None)
                            record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                         "error": f"payload could not be saved: {type(e).__name__}: {str(e)[:160]}"})
                            continue
                    cap_before = _cap_counts()
                    cap, cap_src = _victim_caption(ptext, pimg, meta, attack_name, task, None)
                    if cap_src == "caption_failed":
                        # Captioning the attack image failed: no row may
                        # complete with a placeholder as the victim's
                        # description (review 4, H4). Retried on resubmit.
                        record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                     "error": f"attack image caption failed: {captioner.last_error}"})
                        continue
                    if cap:
                        adv_cap_raw[url] = cap
                        adv_cap[url] = _for_resub(cap)
                    if pimg is not None:
                        adv_img[url] = pimg
                    meta["caption_source"] = cap_src
                    meta["victim_caption"] = cap

                vwa_task = VWAAdvTask(config_file=task.adv_config, task_id=task.name, intent=task.intent)
                if attack_name == "clean":
                    cap_before = _cap_counts()
                t0 = time.perf_counter()
                with _cuda_memory_guard() as mem:
                    out, err = _run_episode_safely(
                        run_vwa_adv_episode, model, processor, extractor, model_id, vwa_task, env,
                        instruction_path=str(VWA_ADV_REPO / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"),
                        max_steps=max_steps, adv_url2caption=adv_cap, adv_url2image=adv_img,
                        payload_markers=_markers(adv_cap_raw, _clean_caption(task, None)),
                    )
                ep_s = time.perf_counter() - t0

                traj = out.get("trajectory")
                # ONE ASR path for every condition: the in-episode evaluation
                # of the adversarial config, run first on the agent's final
                # page (review A7). UA is scored after restoring that page.
                asr = out.get("success") if traj else None
                ua, ua_err = score(task.clean_config, traj, out.get("final_url")) if (traj and task.ua_scorable) else (None, None)

                atomic_torch_save(out.get("step_activations", []), output_dir / f"{sid}.pt")
                fin_err, fin_extra = _final_error(err, out, ua_err, cap_before, task, sid)
                if attack_name == "clean":
                    # obfuscated_activation optimises its suffix to keep the
                    # attacked state close to this task's own clean state, so
                    # the reference must be THIS task's clean activation, not a
                    # pooled baseline. Last step, matching the geometry
                    # analysis: step 0 is pre-injection. Only from a clean
                    # row that COMPLETES (review 5, M4) -- the same rule
                    # _clean_ref applies in a later process.
                    steps = out.get("step_activations") or []
                    if steps and fin_err is None:
                        _clean_acts[task.name] = steps[-1]
                record(sid, {
                    "sample_id": sid, "condition": "clean" if attack_name == "clean" else "attacked",
                    "attack_name": attack_name, "task_id": task.name, "base_task": task.base,
                    "site": site, "intent": task.intent, "adversarial_goal": task.adversarial_goal,
                    "strict_asr": getattr(task, "strict_asr", False),
                    "clean_ua_authored": getattr(task, "clean_ua_authored", False),
                    "asr_success": asr, "utility_success": ua,
                    "ua_note": None if task.ua_scorable else "clean evaluator targets a different intent -- UA not scorable",
                    "num_steps": out.get("num_steps"), "attack_generation_time_s": gen_s,
                    "episode_time_s": ep_s, "peak_cuda_memory_gb": mem["peak_gb"],
                    "error": fin_err, **meta, **fin_extra,
                    "ua_scoring_error": ua_err, **_episode_fields(out, cap_before, task, None),
                })
                logger.info("  done: %s (ASR=%s, UA=%s, gen=%s, ep=%.1fs)",
                            sid, asr, ua, f"{gen_s:.1f}s" if gen_s else "n/a", ep_s)

        # ---------------- defended passes ----------------------------
        # Two per defence. defended_clean measures the defence's own
        # displacement with nothing to defend against; without it,
        # rq2_geometry.py has no control and the restoration-versus-
        # suppression question is unanswerable.
        for defense_name in (defense_names or []):
            defense = defense_registry.get(defense_name)
            ecso_obj = ecso_registry.get(defense_name)
            # Page-level blur also blurs what the captioner reads, so the
            # environment's cached clean captions are invalid for this
            # defence and are recomputed; they are cleared again after it.
            blur_def = defense if getattr(defense, "page_level", False) else None
            _blur_state["def"] = blur_def
            captioner.set_blur(blur_def)
            if blur_def is not None:
                logger.info("blur %s: cleared %d cached captions", defense_name, clear_caption_cache(env))

            logger.info("--- defended CLEAN: %s (%d tasks) ---",
                        defense_name, len(usable))
            for task in usable:
                sid = f"defended_{defense_name}_{task.name}"
                if done(sid):
                    logger.info("  skip %s", sid); continue
                seed_err = _seed_victim(task)
                if seed_err:
                    record(sid, {"sample_id": sid, "attack_name": "clean", "defense_name": defense_name,
                                 "task_id": task.name, "error": seed_err})
                    continue
                vwa_task = VWAAdvTask(config_file=task.adv_config,
                                      task_id=task.name, intent=task.intent)
                cap_before = _cap_counts()
                t0 = time.perf_counter()
                with _cuda_memory_guard() as mem:
                    out, err = _run_episode_safely(
                        run_vwa_adv_episode, model, processor, extractor,
                        model_id, vwa_task, env,
                        instruction_path=str(VWA_ADV_REPO / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"),
                        max_steps=max_steps, adv_url2caption={}, adv_url2image={},
                        defense=defense, ecso=ecso_obj,
                    )
                out.setdefault("payload_in_prompt", [])
                ep_s = time.perf_counter() - t0
                traj = out.get("trajectory")
                asr = out.get("success") if traj else None
                ua, ua_err = score(task.clean_config, traj, out.get("final_url")) if (traj and task.ua_scorable) else (None, None)
                atomic_torch_save(out.get("step_activations", []), output_dir / f"{sid}.pt")
                fin_err, fin_extra = _final_error(err, out, ua_err, cap_before, task, sid)
                record(sid, {
                    "sample_id": sid, "condition": "defended_clean",
                    "attack_name": "clean", "defense_name": defense_name,
                    "task_id": task.name, "base_task": task.base, "site": site,
                    "intent": task.intent, "adversarial_goal": task.adversarial_goal,
                    "strict_asr": getattr(task, "strict_asr", False),
                    "clean_ua_authored": getattr(task, "clean_ua_authored", False),
                    "asr_success": asr, "utility_success": ua,
                    "ua_note": None if task.ua_scorable else "clean evaluator targets a different intent -- UA not scorable",
                    "num_steps": out.get("num_steps"), "attack_generation_time_s": None,
                    "episode_time_s": ep_s, "peak_cuda_memory_gb": mem["peak_gb"],
                    "error": fin_err, **fin_extra, "ua_scoring_error": ua_err,
                    **_episode_fields(out, cap_before, task, blur_def),
                })
                logger.info("  done: %s (ASR=%s, UA=%s, ep=%.1fs)", sid, asr, ua, ep_s)

            for attack_name in attack_names:
                logger.info("--- defended %s: %s (%d tasks) ---",
                            attack_name, defense_name, len(usable))
                for task in usable:
                    sid = f"{attack_name}_defended_{defense_name}_{task.name}"
                    if done(sid):
                        logger.info("  skip %s", sid); continue
                    cached = _payloads.get((attack_name, task.name))
                    if cached is None:
                        # The undefended pass for this (attack, task) did
                        # not produce a payload -- generation failed, or
                        # this is a resumed run whose attack loop was
                        # skipped. Either way, defending nothing would
                        # silently produce a row labelled defended that
                        # is really a clean episode.
                        record(sid, {"sample_id": sid, "attack_name": attack_name,
                                     "defense_name": defense_name, "task_id": task.name,
                                     "error": "no cached payload from the undefended pass"})
                        continue
                    text, image, meta = cached
                    meta = dict(meta)
                    url = meta.get("victim_url") or task.victim_image_url()
                    seed_err = _seed_victim(task)
                    if seed_err:
                        record(sid, {"sample_id": sid, "attack_name": attack_name,
                                     "defense_name": defense_name, "task_id": task.name,
                                     "error": seed_err})
                        continue
                    cap_before = _cap_counts()
                    cap, cap_src = _victim_caption(text, image, meta, attack_name, task, blur_def)
                    if cap_src == "caption_failed":
                        record(sid, {"sample_id": sid, "attack_name": attack_name,
                                     "defense_name": defense_name, "task_id": task.name,
                                     "error": f"attack image caption failed: {captioner.last_error}"})
                        continue
                    adv_cap_raw = {url: cap} if cap else {}
                    adv_cap = {url: _for_resub(cap)} if cap else {}
                    adv_img = {url: image} if image is not None else {}
                    meta["caption_source"], meta["victim_caption"] = cap_src, cap
                    vwa_task = VWAAdvTask(config_file=task.adv_config,
                                          task_id=task.name, intent=task.intent)
                    t0 = time.perf_counter()
                    with _cuda_memory_guard() as mem:
                        out, err = _run_episode_safely(
                            run_vwa_adv_episode, model, processor, extractor,
                            model_id, vwa_task, env,
                            instruction_path=str(VWA_ADV_REPO / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"),
                            max_steps=max_steps, adv_url2caption=adv_cap,
                            adv_url2image=adv_img, defense=defense, ecso=ecso_obj,
                            payload_markers=_markers(adv_cap_raw, _clean_caption(task, blur_def)),
                        )
                    ep_s = time.perf_counter() - t0
                    traj = out.get("trajectory")
                    asr = out.get("success") if traj else None
                    ua, ua_err = score(task.clean_config, traj, out.get("final_url")) if (traj and task.ua_scorable) else (None, None)
                    atomic_torch_save(out.get("step_activations", []), output_dir / f"{sid}.pt")
                    fin_err, fin_extra = _final_error(err, out, ua_err, cap_before, task, sid)
                    record(sid, {
                        "sample_id": sid, "condition": "defended_attacked",
                        "attack_name": attack_name, "defense_name": defense_name,
                        "task_id": task.name, "base_task": task.base, "site": site,
                        "intent": task.intent, "adversarial_goal": task.adversarial_goal,
                        "strict_asr": getattr(task, "strict_asr", False),
                    "clean_ua_authored": getattr(task, "clean_ua_authored", False),
                        "asr_success": asr, "utility_success": ua,
                        "ua_note": None if task.ua_scorable else "clean evaluator targets a different intent -- UA not scorable",
                        "num_steps": out.get("num_steps"),
                        "attack_generation_time_s": 0.0,  # reused, not regenerated
                        "episode_time_s": ep_s, "peak_cuda_memory_gb": mem["peak_gb"],
                        "error": fin_err, **meta, **fin_extra, "ua_scoring_error": ua_err,
                        **_episode_fields(out, cap_before, task, blur_def),
                    })
                    logger.info("  done: %s (ASR=%s, UA=%s, ep=%.1fs)", sid, asr, ua, ep_s)
            _blur_state["def"] = None
            if blur_def is not None:
                captioner.set_blur(None)
                clear_caption_cache(env)
    finally:
        env.close()

    logger.info("Curated matrix complete: %d rows -> %s", len(results), results_path)


if __name__ == "__main__":
    import argparse
    from extraction.hooks import _resolve_target_layers
    from extraction.model_loader import LoadConfig, VLMLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    ap.add_argument("--attack_names", type=str, nargs="+",
                    default=["template_ipi", "cross_modal_injection"])
    ap.add_argument("--site", type=str, default="reddit")
    ap.add_argument("--exp_data", type=str, default=str(AGENT_ATTACK_REPO / "exp_data"))
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--only_tasks", type=str, nargs="*", default=None,
                    help="restrict to these curated task names, e.g. the 10 "
                         "whose clean-UA evaluators were authored")
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--defense_names", type=str, nargs="*", default=None,
                    help="spotlighting_delimiting spotlighting_datamarking "
                         "spotlighting_encoding; runs defended-clean AND "
                         "defended-attacked passes per defence")
    ap.add_argument("--output_dir", type=str, default="vwa_adv_curated")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: no CUDA device visible. device_map='auto' would silently place the model on "
            "CPU and run at roughly 100x slowdown without erroring -- which is exactly what "
            "happened on mscluster46, where Postmill served fine (no GPU needed) and the dead GPU "
            "only surfaced much later. bigbatch has GRES=(null), so SLURM does not prevent this. "
            "Move to a node with a working GPU and add this one to the exclude lists."
        )

    # PRECISION MUST MATCH THE AGENTDOJO PATH. This driver hardcoded
    # dtype=torch.float16 while model_loader's load_qwen3vl_8b /
    # load_internvl_8b set bfloat16 -- so the SAME model ran at
    # different precision in the two environments (logs: AgentDojo job
    # 44510 "dtype=torch.bfloat16", VWA-Adv job 48183
    # "dtype=torch.float16"). fp16 has a fraction of bf16's dynamic
    # range, the Qwen-VL family is known to overflow in it, and this
    # project already hit a NaN-in-probability crash serious enough to
    # warrant experiments/qwen3vl_nan_repro.py. Cross-environment
    # comparison across that gap is confounded by numerics.
    from extraction.model_loader import (
        load_llavanext_7b, load_internvl_8b, load_qwen3vl_8b, load_gemma4_12b)
    _LOADERS = {"Qwen/Qwen3-VL-8B-Instruct": load_qwen3vl_8b,
                "OpenGVLab/InternVL3-8B-hf": load_internvl_8b,
                "llava-hf/llava-v1.6-mistral-7b-hf": load_llavanext_7b,
                "google/gemma-4-12B-it": load_gemma4_12b}
    if a.model_id in _LOADERS:
        model, processor = _LOADERS[a.model_id](hpc=True)
    else:
        logger.warning("no named loader for %s -- using LoadConfig defaults. "
                       "Verify dtype matches the AgentDojo runs before "
                       "comparing across environments.", a.model_id)
        model, processor = VLMLoader(LoadConfig(
            model_id=a.model_id, device_map="auto",
            attn_implementation="eager")).load()
    run_curated_matrix(model, processor, a.model_id, a.attack_names,
                       _resolve_target_layers(model, None), Path(a.output_dir),
                       Path(a.exp_data), a.site, a.max_tasks, a.max_steps,
                       only_tasks=a.only_tasks,
                       defense_names=a.defense_names)
    print(f"Done. See {a.output_dir}/attack_results.json")
