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
from extraction.hooks import ActivationExtractor
from experiments.run_attack_matrix import _cuda_memory_guard, _run_episode_safely

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"url:\s*(\S+?)\]?$")


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
    if max_tasks:
        usable = usable[:max_tasks]

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

    defense_registry = {}
    ecso_registry = {}
    if defense_names:
        from experiments.run_attack_matrix import _build_defense_registry
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

    env = ScriptBrowserEnv(headless=True, observation_type="image_som",
                           current_viewport_only=False, viewport_size={"width": 1280, "height": 2048})

    results_path = output_dir / "attack_results.json"
    results: List[Dict[str, Any]] = json.loads(results_path.read_text()) if results_path.exists() else []
    index = {r["sample_id"]: i for i, r in enumerate(results)}

    def done(sid): return sid in index and results[index[sid]].get("error") is None

    def record(sid, row):
        if sid in index:
            results[index[sid]] = row
        else:
            results.append(row); index[sid] = len(results) - 1
        results_path.write_text(json.dumps(results, indent=2))

    def score(config_path: Path, trajectory) -> Optional[bool]:
        """Scores an already-finished episode against a config's evaluator."""
        try:
            ev = evaluator_router(str(config_path), captioning_fn=None)
            return bool(ev(trajectory=trajectory, config_file=str(config_path),
                           page=env.page, client=env.get_page_client(env.page)))
        except Exception as e:
            logger.warning("  scoring against %s failed (%s: %s)", config_path.name, type(e).__name__, e)
            return None

    try:
        for attack_name in ["clean"] + list(attack_names):
            logger.info("--- %s (%d tasks) ---", attack_name, len(usable))
            for task in usable:
                sid = f"{attack_name}_{task.name}"
                if done(sid):
                    logger.info("  skip %s", sid); continue

                adv_cap, adv_img, meta, gen_s = {}, {}, {}, None
                if attack_name != "clean":
                    url = task.victim_image_url()
                    if not url:
                        record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                     "error": "no victim image URL in recorded obs_text.txt"})
                        continue
                    vp = task.victim_image_path()
                    if vp is not None:
                        img = Image.open(vp).convert("RGB")
                    else:
                        img = _fetch_image(url)
                    if img is None:
                        record(sid, {"sample_id": sid, "attack_name": attack_name, "task_id": task.name,
                                     "error": f"could not fetch victim image {url}"})
                        continue

                    # The REAL published adversarial goal drives the attack --
                    # not a generic placeholder string.
                    goal = task.adversarial_goal or task.data.get("target_caption", "")
                    base_input = {"prompt": task.intent, "image": img,
                                  "injectable_field": "caption", "caption": ""}

                    # obfuscated_activation optimises its suffix to hold the
                    # attacked state near THIS task's own clean state, so the
                    # reference must come from this task's clean episode.
                    # Resolved here rather than inside the try below, where a
                    # `continue` would be illegal.
                    _ref_act = None
                    if attack_name == "obfuscated_activation":
                        _ref = _clean_acts.get(task.name)
                        _layer = target_layers[len(target_layers) // 2]
                        if not _ref or _layer not in _ref:
                            record(sid, {"sample_id": sid, "attack_name": attack_name,
                                         "task_id": task.name,
                                         "error": "no clean reference activation for this task"})
                            continue
                        _ref_act = _ref[_layer]

                    t0 = time.perf_counter()
                    try:
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
                                res = atk.generate(base_input, target_behavior=goal, image=img, clean_text="")
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
                    gen_s = time.perf_counter() - t0
                    if res.adversarial_text:
                        adv_cap[url] = res.adversarial_text
                    if res.adversarial_image:
                        adv_img[url] = res.adversarial_image
                    meta = {f"attack_meta_{k}": v for k, v in (res.metadata or {}).items()}

                if attack_name != "clean":
                    _payloads[(attack_name, task.name)] = (adv_cap, adv_img, meta)

                vwa_task = VWAAdvTask(config_file=task.adv_config, task_id=task.name, intent=task.intent)
                t0 = time.perf_counter()
                with _cuda_memory_guard() as mem:
                    out, err = _run_episode_safely(
                        run_vwa_adv_episode, model, processor, extractor, model_id, vwa_task, env,
                        instruction_path=str(VWA_ADV_REPO / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"),
                        max_steps=max_steps, adv_url2caption=adv_cap, adv_url2image=adv_img,
                    )
                ep_s = time.perf_counter() - t0

                traj = out.get("trajectory")
                asr = out.get("success") if attack_name != "clean" else (score(task.adv_config, traj) if traj else None)
                ua = score(task.clean_config, traj) if (traj and task.ua_scorable) else None

                torch.save(out.get("step_activations", []), output_dir / f"{sid}.pt")
                if attack_name == "clean":
                    # obfuscated_activation optimises its suffix to keep the
                    # attacked state close to this task's own clean state, so
                    # the reference must be THIS task's clean activation, not a
                    # pooled baseline. Last step, matching the geometry
                    # analysis: step 0 is pre-injection.
                    steps = out.get("step_activations") or []
                    if steps:
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
                    "episode_time_s": ep_s, "peak_cuda_memory_gb": mem["peak_gb"], "error": err, **meta,
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

            logger.info("--- defended CLEAN: %s (%d tasks) ---",
                        defense_name, len(usable))
            for task in usable:
                sid = f"defended_{defense_name}_{task.name}"
                if done(sid):
                    logger.info("  skip %s", sid); continue
                vwa_task = VWAAdvTask(config_file=task.adv_config,
                                      task_id=task.name, intent=task.intent)
                t0 = time.perf_counter()
                with _cuda_memory_guard() as mem:
                    out, err = _run_episode_safely(
                        run_vwa_adv_episode, model, processor, extractor,
                        model_id, vwa_task, env,
                        instruction_path=str(VWA_ADV_REPO / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"),
                        max_steps=max_steps, adv_url2caption={}, adv_url2image={},
                        defense=defense, ecso=ecso_obj,
                    )
                ep_s = time.perf_counter() - t0
                traj = out.get("trajectory")
                asr = score(task.adv_config, traj) if traj else None
                ua = score(task.clean_config, traj) if (traj and task.ua_scorable) else None
                torch.save(out.get("step_activations", []), output_dir / f"{sid}.pt")
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
                    "error": err,
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
                    adv_cap, adv_img, meta = cached
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
                        )
                    ep_s = time.perf_counter() - t0
                    traj = out.get("trajectory")
                    asr = out.get("success")
                    ua = score(task.clean_config, traj) if (traj and task.ua_scorable) else None
                    torch.save(out.get("step_activations", []), output_dir / f"{sid}.pt")
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
                        "error": err, **meta,
                    })
                    logger.info("  done: %s (ASR=%s, UA=%s, ep=%.1fs)", sid, asr, ua, ep_s)
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
                       defense_names=a.defense_names)
    print(f"Done. See {a.output_dir}/attack_results.json")
