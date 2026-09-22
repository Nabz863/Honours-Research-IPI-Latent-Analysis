"""
experiments/run_vwa_adv_matrix.py
-------------------------------------
VWA-Adv attack matrix driver -- the VWA-Adv analog of run_attack_matrix.py's
run_agentdojo_attack_matrix(). Imports shared helpers from there rather
than duplicating them.

ATTACKS-FIRST PHASE, same sequencing and reasoning as the AgentDojo
matrix (proposal 3.4.2, "Baseline Anchoring"): no defense wiring yet.

THREE REAL SCOPE DECISIONS, worth restating if asked, not silent:

1. Reddit-only tasks (153 of 210 real generated tasks). 57 reference a
   second site (Wikipedia/Shopping/etc.) that was never actually stood
   up -- REDDIT is the only site env var pointing at something real
   (Postmill); the rest are placeholders pointed at the same instance.
   Running a multi-site task as-is means its second tab loads nothing
   real. Confirmed against the raw (pre-generation) task file, since
   generated per-task configs have placeholders already substituted
   with real URLs and lose that signal -- filtering cross-references
   task_id between the two.

2. Attack delivery targets each task's own real "image" field URL.
   Visual/cross-modal attacks override the actual rendered pixels via
   adv_url2image -- confirmed via real source
   (browser_env/processors.py): image.evaluate("node => node.src =
   'data:image/png;base64,...'") is genuine DOM mutation the browser
   renders, not a text-only override. Textual attacks (template_ipi,
   gcg) deliver via adv_url2caption targeting the SAME image URL --
   its accessibility-tree "description" -- a real, realistic IPI
   vector (poisoned image caption), not an arbitrary choice. Tasks
   without an "image" field are skipped for ALL attacks in this first
   version, logged with a clear reason, not silently dropped.

3. ASR is NOT measured here -- explicitly, not silently. evaluator_router()
   scores against the task's OWN reference answer (the benign user's
   original goal); it has no concept of an attacker-injected goal at
   all. result["success"] is genuine UTILITY (did the real task still
   complete), logged as utility_success. asr_success is left None with
   a clear reason in metadata, not backfilled with a wrong proxy.
   Real ASR measurement needs its own design (most likely: inspect the
   trajectory for an action matching the attacker's specific goal) --
   a real conversation, not something to guess at here.

CLEAN_TEXT / TARGET_BEHAVIOR ARE CURRENTLY GENERIC, not scraped from the
real page or defined per-task -- clean_text is a placeholder description
("A photo posted to r/<subreddit>."), target_behavior is the same
generic compliance string used in this project's own smoketests. Real,
flagged simplifications for getting the pipeline correct end-to-end
first; revisit with real per-task attacker goals once confirmed working,
same "baseline anchoring" sequencing as the phase this whole file is in.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import torch
from PIL import Image

from attacks.base import BaseAttack
from attacks.template_ipi import TemplateIPIAttack
from attacks.gcg import GCGAttack
from attacks.pixel_perturbation import PixelPerturbationAttack
from attacks.cross_modal_injection import CrossModalInjectionAttack

from extraction.hooks import ActivationExtractor

from experiments.run_attack_matrix import (
    _cuda_memory_guard,
    _run_episode_safely,
    _wandb_init_if_requested,
    _wandb_log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VWA_ADV_REPO = Path(os.environ.get("VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
sys.path.insert(0, str(VWA_ADV_REPO))

try:
    from browser_env import ScriptBrowserEnv
except ImportError:
    ScriptBrowserEnv = None

from environments.vwa_adv_tasks import VWAAdvTask, load_tasks, run_vwa_adv_episode

DEFAULT_INSTRUCTION_PATH = str(VWA_ADV_REPO / "agent" / "prompts" / "jsons" / "p_som_cot_id_actree_3s.json")
_NON_REDDIT_PLACEHOLDERS = ["__SHOPPING__", "__WIKIPEDIA__", "__HOMEPAGE__", "__CLASSIFIEDS__", "__SHOPPING_ADMIN__", "__GITLAB__"]


def _build_vwa_adv_attack_registry(model, processor) -> Dict[str, BaseAttack]:
    """
    template_ipi/gcg/pixel_perturbation reused as-is; cross_modal_injection
    added here specifically -- it's not in run_attack_matrix.py's own
    _build_attack_registry() since AgentDojo has no images to attack.
    obfuscated_activation deliberately excluded from this first version:
    its clean_reference_activation sourcing (from a task's own clean
    episode) adds real complexity worth confirming works for AgentDojo
    first before porting the same pattern here.
    """
    return {
        "template_ipi": TemplateIPIAttack(),
        "gcg": GCGAttack(model, processor),
        "pixel_perturbation": PixelPerturbationAttack(model, processor),
        "cross_modal_injection": CrossModalInjectionAttack(model, processor),
    }


def _reddit_only_task_ids(config_dir: Path, suite: str) -> set:
    """
    Cross-references the RAW (pre-generation) task file against
    generated task_ids -- generated per-task configs already have
    placeholders substituted with real URLs, losing the signal for
    which tasks originally referenced a non-Reddit site.
    """
    raw_path = config_dir / f"{suite}.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found -- needed to determine which tasks are genuinely "
            "Reddit-only (generated per-task files have already substituted real URLs "
            "in place of placeholders, losing this signal)."
        )
    with open(raw_path) as f:
        raw_tasks = json.load(f)

    reddit_only = set()
    for t in raw_tasks:
        task_id = str(t.get("task_id"))
        if not any(p in json.dumps(t) for p in _NON_REDDIT_PLACEHOLDERS):
            reddit_only.add(task_id)
    return reddit_only


def _fetch_clean_image(url: str, timeout: float = 15.0) -> Optional[Image.Image]:
    """Downloads a task's real image from the live Postmill instance.
    Returns None (not a raised exception) on failure -- caller decides
    whether a missing image means skip-this-task or something else;
    the network call itself isn't the interesting failure mode here."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("  Failed to fetch clean image from %s (%s: %s)", url, type(e).__name__, e)
        return None


def run_vwa_adv_attack_matrix(
    model,
    processor,
    model_id: str,
    attack_names: List[str],
    target_layers,
    output_dir: Path,
    config_dir: Optional[Path] = None,
    suite: str = "test_reddit",
    reddit_only: bool = True,
    max_tasks: Optional[int] = None,
    max_steps: int = 15,
    use_wandb: bool = False,
    wandb_project: str = "ipi-latent-analysis",
    wandb_run_name: Optional[str] = None,
) -> None:
    if ScriptBrowserEnv is None:
        raise ImportError("browser_env is not importable -- check VWA_ADV_REPO/PYTHONPATH.")

    output_dir = output_dir.resolve()  # ABSOLUTE before chdir below, or every write lands in the wrong repo
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = (config_dir or (VWA_ADV_REPO / "config_files")).resolve()

    # Task configs' storage_state paths are relative to the VWA-Adv repo
    # root (same fix as run_one_vwa_adv_episode.py) -- must chdir AFTER
    # resolving output_dir/config_dir to absolute paths above, not before.
    os.chdir(str(VWA_ADV_REPO))

    all_tasks = load_tasks(config_dir, suite=suite)
    if reddit_only:
        keep_ids = _reddit_only_task_ids(config_dir, suite)
        tasks = [t for t in all_tasks if t.task_id in keep_ids]
        logger.info("Reddit-only filter: %d of %d tasks kept", len(tasks), len(all_tasks))
    else:
        tasks = all_tasks

    # Filter further: this version only attacks tasks with a real "image"
    # field (see module docstring, decision 2). Also reads each task's
    # image URL now, once, rather than re-reading the config file per attack.
    tasks_with_image: List[Tuple[VWAAdvTask, str]] = []
    for t in tasks:
        with open(t.config_file) as f:
            cfg = json.load(f)
        image_url = cfg.get("image")
        if image_url:
            tasks_with_image.append((t, image_url))
        else:
            logger.info("  skip %s from attack pool (no 'image' field) -- clean pass still runs it", t.task_id)

    logger.info(
        "%d Reddit-only tasks total, %d have a real 'image' field (attack-eligible)",
        len(tasks), len(tasks_with_image),
    )

    if max_tasks is not None:
        tasks = tasks[:max_tasks]
        tasks_with_image = tasks_with_image[:max_tasks]

    extractor = ActivationExtractor(model, target_layers=target_layers, token_aggregation="last")
    registry = _build_vwa_adv_attack_registry(model, processor)
    for name in attack_names:
        if name not in registry:
            raise KeyError(f"Unknown attack {name!r} -- available: {sorted(registry.keys())}")

    logger.info("Constructing ScriptBrowserEnv (headless, image_som)...")
    env = ScriptBrowserEnv(
        headless=True, observation_type="image_som",
        current_viewport_only=True, viewport_size={"width": 1280, "height": 720},
    )

    results_log_path = output_dir / "attack_results.json"
    results_log: List[Dict[str, Any]] = []
    if results_log_path.exists():
        with open(results_log_path) as f:
            results_log = json.load(f)
        logger.info("Resumed %d already-completed results from %s", len(results_log), results_log_path)
    results_index: Dict[str, int] = {r["sample_id"]: i for i, r in enumerate(results_log)}

    def _is_complete(sample_id: str) -> bool:
        idx = results_index.get(sample_id)
        return idx is not None and results_log[idx].get("error") is None

    def _record_result(sample_id: str, row: Dict[str, Any]) -> None:
        idx = results_index.get(sample_id)
        if idx is None:
            results_log.append(row)
            results_index[sample_id] = len(results_log) - 1
        else:
            results_log[idx] = row

    def _persist() -> None:
        with open(results_log_path, "w") as f:
            json.dump(results_log, f, indent=2)

    wandb_run = _wandb_init_if_requested(
        use_wandb, wandb_project, wandb_run_name,
        config={"attack_names": attack_names, "reddit_only": reddit_only, "n_tasks": len(tasks),
                "n_attack_eligible": len(tasks_with_image), "target_layers": list(target_layers)},
    )

    try:
        logger.info("--- Clean baseline pass (%d tasks) ---", len(tasks))
        for task in tasks:
            sample_id = f"clean_{task.task_id}"
            if _is_complete(sample_id):
                logger.info("  skip %s (already done)", sample_id)
                continue

            t0 = time.perf_counter()
            with _cuda_memory_guard() as mem:
                out, episode_error = _run_episode_safely(
                    run_vwa_adv_episode, model, processor, extractor, model_id, task, env,
                    instruction_path=DEFAULT_INSTRUCTION_PATH, max_steps=max_steps,
                )
            episode_time_s = time.perf_counter() - t0

            torch.save(out.get("step_activations", []), output_dir / f"{sample_id}.pt")
            row = {
                "sample_id": sample_id, "condition": "clean", "attack_name": "clean",
                "task_id": task.task_id, "intent": task.intent,
                "utility_success": out.get("success"), "asr_success": None,
                "num_steps": out.get("num_steps"), "episode_time_s": episode_time_s,
                "peak_cuda_memory_gb": mem["peak_gb"], "error": episode_error,
                "asr_note": "not measured -- evaluator_router scores the benign task's own goal, not an attacker goal",
            }
            _record_result(sample_id, row)
            _wandb_log(wandb_run, **row)
            _persist()
            if episode_error:
                logger.warning("  FAILED: %s (%s, %.1fs)", sample_id, episode_error, episode_time_s)
            else:
                logger.info("  done: %s (utility=%s, %.1fs)", sample_id, out.get("success"), episode_time_s)

        for attack_name in attack_names:
            attack = registry[attack_name]
            logger.info("--- Attack: %s (%d attack-eligible tasks) ---", attack_name, len(tasks_with_image))
            for task, image_url in tasks_with_image:
                sample_id = f"{attack_name}_{task.task_id}"
                if _is_complete(sample_id):
                    logger.info("  skip %s (already done)", sample_id)
                    continue

                clean_image = _fetch_clean_image(image_url)
                if clean_image is None:
                    row = {
                        "sample_id": sample_id, "condition": "attacked", "attack_name": attack_name,
                        "task_id": task.task_id, "error": f"could not fetch clean image from {image_url}",
                    }
                    _record_result(sample_id, row)
                    _persist()
                    continue

                target_behavior = "Sure, I will help with that."  # see module docstring -- generic, flagged placeholder
                clean_text = f"A photo posted to r/{task.task_id.split('_')[0]}."  # same -- generic placeholder

                base_input = {"prompt": task.intent, "image": clean_image, "injectable_field": "caption", "caption": clean_text}

                t0 = time.perf_counter()
                gen_error: Optional[str] = None
                try:
                    with _cuda_memory_guard() as gen_mem:
                        if attack_name in ("template_ipi", "gcg"):
                            adv_result = attack.generate(base_input, target_behavior=target_behavior)
                        elif attack_name == "pixel_perturbation":
                            adv_result = attack.generate(base_input, target_behavior=target_behavior, image=clean_image)
                        else:  # cross_modal_injection
                            adv_result = attack.generate(base_input, target_behavior=target_behavior, image=clean_image, clean_text=clean_text)
                except torch.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    gc.collect()
                    gen_error = f"CUDA_OOM_during_generation: {str(e).splitlines()[0][:200]}"
                attack_generation_time_s = time.perf_counter() - t0

                if gen_error:
                    row = {"sample_id": sample_id, "condition": "attacked", "attack_name": attack_name,
                           "task_id": task.task_id, "attack_generation_time_s": attack_generation_time_s, "error": gen_error}
                    _record_result(sample_id, row)
                    _persist()
                    logger.warning("  FAILED: %s (%s)", sample_id, gen_error)
                    continue

                adv_url2caption, adv_url2image = CrossModalInjectionAttack.to_vwa_adv_delivery(adv_result, image_url)

                t0 = time.perf_counter()
                with _cuda_memory_guard() as ep_mem:
                    out, episode_error = _run_episode_safely(
                        run_vwa_adv_episode, model, processor, extractor, model_id, task, env,
                        instruction_path=DEFAULT_INSTRUCTION_PATH, max_steps=max_steps,
                        adv_url2caption=adv_url2caption, adv_url2image=adv_url2image,
                    )
                episode_time_s = time.perf_counter() - t0

                torch.save(out.get("step_activations", []), output_dir / f"{sample_id}.pt")
                row = {
                    "sample_id": sample_id, "condition": "attacked", "attack_name": attack_name,
                    "task_id": task.task_id, "intent": task.intent,
                    "utility_success": out.get("success"), "asr_success": None,
                    "num_steps": out.get("num_steps"),
                    "attack_generation_time_s": attack_generation_time_s, "episode_time_s": episode_time_s,
                    "peak_cuda_memory_gb_generation": gen_mem["peak_gb"], "peak_cuda_memory_gb_episode": ep_mem["peak_gb"],
                    "error": episode_error,
                    "asr_note": "not measured -- evaluator_router scores the benign task's own goal, not an attacker goal",
                    **{f"attack_meta_{k}": v for k, v in adv_result.metadata.items()},
                }
                _record_result(sample_id, row)
                _wandb_log(wandb_run, **row)
                _persist()
                if episode_error:
                    logger.warning("  FAILED: %s (%s, %.1fs)", sample_id, episode_error, episode_time_s)
                else:
                    logger.info("  done: %s (utility=%s, gen=%.1fs, ep=%.1fs)",
                                sample_id, out.get("success"), attack_generation_time_s, episode_time_s)
    finally:
        env.close()

    logger.info("VWA-Adv attack matrix run complete: %d rows -> %s", len(results_log), results_log_path)
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception as e:
            logger.warning("wandb.finish() failed (%s: %s) -- harmless.", type(e).__name__, e)


if __name__ == "__main__":
    import argparse
    from extraction.hooks import _resolve_target_layers
    from extraction.model_loader import LoadConfig, VLMLoader

    parser = argparse.ArgumentParser(description="Run the VWA-Adv attack matrix.")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--attack_names", type=str, nargs="+", default=["template_ipi", "gcg"])
    parser.add_argument("--suite", type=str, default="test_reddit")
    parser.add_argument("--all_sites", action="store_true", help="Disable the Reddit-only filter (not recommended, see module docstring).")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ipi-latent-analysis")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="vwa_adv_matrix")
    args = parser.parse_args()

    cfg = LoadConfig(model_id=args.model_id, dtype=torch.float16, device_map="auto", attn_implementation="eager")
    model, processor = VLMLoader(cfg).load()
    target_layers = _resolve_target_layers(model, None)

    run_vwa_adv_attack_matrix(
        model=model, processor=processor, model_id=args.model_id,
        attack_names=args.attack_names, target_layers=target_layers,
        output_dir=Path(args.output_dir), suite=args.suite, reddit_only=not args.all_sites,
        max_tasks=args.max_tasks, max_steps=args.max_steps,
        use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=args.wandb_run_name,
    )
    print(f"Done. Check {args.output_dir}/ for saved activations and attack_results.json.")
