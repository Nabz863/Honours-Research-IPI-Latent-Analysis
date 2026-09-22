"""
experiments/run_one_vwa_adv_episode.py
------------------------------------------
The actual next concrete step for VWA-Adv, flagged since the earliest
handoff and unchanged in intent since: not the full 210-task suite --
ONE real episode through run_vwa_adv_episode(), checking specifically:
does login/session work, does SoM action parsing produce something
LocalVLMAgent can parse, does a click/type action land, does
evaluator_router produce a real, sane score.

Every signature this depends on has now been checked against the real,
installed repo source directly (not assumed by analogy):
  - ScriptBrowserEnv.reset()/step()'s adv_url2caption/adv_url2image kwargs
  - EvaluatorComb.__call__(trajectory, config_file, page, client)
  - env.get_page_client(page) -> CDPSession
This is the first attempt to actually exercise the whole chain together.

Needs BOTH repos on sys.path: this project's own root (environments/,
extraction/) and the real VWA-Adv repo (browser_env, evaluation_harness,
agent) -- the latter isn't installed as a package, so its path is added
explicitly below via the VWA_ADV_REPO env var rather than assumed.

TWO REAL, UNVERIFIED RISKS this script cannot resolve for you, printed
explicitly below so a failure is diagnosable instead of a mystery:

1. config_files/test_reddit/*.json's baked-in start_url may not point at
   THIS live Postmill instance. generate_test_data.py needs no live
   server to run ("even a placeholder" REDDIT value per earlier project
   notes) -- meaning these 210 configs may have a stale/placeholder host
   baked into their URLs from whenever they were generated, not the real
   host:8080 this run actually points REDDIT at now. This script prints
   the task's own start_url next to the REDDIT env var so a mismatch is
   visible immediately rather than surfacing as an unexplained navigation
   timeout deep inside the episode.

2. run_vwa_adv_episode() does not renew the task's storage_state/login
   cookie the way the real run.py does (auto_login.py subprocess call,
   see run.py's test() function) before calling env.reset(). If this
   task needs authentication and the referenced storage_state is stale
   or was never generated against a genuinely fresh Postmill instance,
   actions requiring login may silently fail. This script prints
   storage_state below too -- if the episode result looks wrong in a way
   that smells like "not actually logged in," this is the first place
   to look, and porting run.py's auto_login call into run_vwa_adv_episode()
   is the fix, not something to guess at blind here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VWA_ADV_REPO = Path(os.environ.get("VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
sys.path.insert(0, str(VWA_ADV_REPO))

# Task configs store storage_state as a RELATIVE path ("./.auth/...")
# assuming the process cwd is the VWA-Adv repo root -- matching how the
# original benchmark's own run.py is always invoked (`cd visualwebarena
# && python run.py ...`). This script (and vwa_adv_full_smoketest.slurm,
# which always cd's to THIS repo's root first) doesn't match that
# assumption otherwise, so Playwright looks for the auth file relative
# to the wrong directory and finds nothing, even though it genuinely
# exists.
os.chdir(str(VWA_ADV_REPO))

import torch

# Beartype's numpy NDArray[dtype] introspection is incompatible with the
# installed numpy (2.x) regardless of beartype version tried (0.12.0
# pinned in requirements.txt, 0.22.9 latest -- both failed on the same
# function, same root cause: numpy 2.x changed dtype/generic internals
# beartype's numpy support was written against). Real fix would be
# pinning numpy back to requirements.txt's 1.25.2, not done here since
# that's shared with currently-running AgentDojo jobs in this same
# venv and a major downgrade risks their torch bindings. Beartype is
# pure runtime validation, not functional logic -- no-opping it changes
# nothing about what this code does.
import beartype as _beartype_module
def _noop_beartype(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda func: func
_beartype_module.beartype = _noop_beartype

from browser_env import ScriptBrowserEnv  # noqa: E402  (needs VWA_ADV_REPO on sys.path first)

from extraction.hooks import ActivationExtractor  # noqa: E402
from extraction.model_loader import load_llavanext_7b  # noqa: E402
from environments.vwa_adv_tasks import load_tasks, run_vwa_adv_episode  # noqa: E402

DEFAULT_INSTRUCTION_PATH = VWA_ADV_REPO / "agent" / "prompts" / "jsons" / "p_som_cot_id_actree_3s.json"


def _print_task_context(config_file: Path) -> None:
    with open(config_file) as f:
        config = json.load(f)
    print(f"Task config: {config_file}")
    print(f"  intent: {config.get('intent')}")
    print(f"  start_url: {config.get('start_url', '<not present in config>')}")
    print(f"  storage_state: {config.get('storage_state', '<not present in config>')}")
    print(f"  REDDIT env var (what this run is actually pointing at): {os.environ.get('REDDIT', '<not set>')}")
    print("  >>> If start_url's host doesn't match REDDIT above, this episode will fail")
    print("      to navigate anywhere real. If storage_state looks present but this run")
    print("      still can't do logged-in actions, see risk #2 in this file's docstring. <<<")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=str, default="0", help="Which config_files/test_reddit/<id>.json to run.")
    parser.add_argument("--config-dir", type=str, default=str(VWA_ADV_REPO / "config_files"))
    parser.add_argument("--suite", type=str, default="test_reddit")
    parser.add_argument("--max-steps", type=int, default=15, help="Kept short for a first real smoketest.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "ERROR: no CUDA device visible. bigbatch has GRES=(null) -- no GPU resource "
            "tracking -- check against the known-bad exclude list before assuming this node "
            "is simply out of GPUs.",
            file=sys.stderr,
        )
        sys.exit(1)

    tasks = load_tasks(Path(args.config_dir), suite=args.suite)
    task = next((t for t in tasks if t.task_id == args.task_id), None)
    if task is None:
        print(f"ERROR: task_id={args.task_id!r} not found in {args.config_dir}/{args.suite}/", file=sys.stderr)
        sys.exit(1)

    _print_task_context(task.config_file)

    print("\nLoading LLaVA-NeXT 7B (hpc=True)...")
    model, processor = load_llavanext_7b(hpc=True)
    print(f"Model loaded on {next(model.parameters()).device}.")

    # capture_step defaults to "first" in the fixed hooks.py -- captures
    # the prefill (post-prompt, pre-generation) state, not the final
    # decode step. Confirmed via the earlier fake-layer test.
    extractor = ActivationExtractor(model, token_aggregation="last")
    print(f"ActivationExtractor targeting layers: {extractor.target_layers}")

    print("\nConstructing ScriptBrowserEnv (headless, image_som)...")
    env = ScriptBrowserEnv(
        headless=True,
        observation_type="image_som",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 720},
    )

    try:
        print(f"\nRunning episode for task {task.task_id}: {task.intent!r}")
        result = run_vwa_adv_episode(
            model=model,
            processor=processor,
            extractor=extractor,
            model_id="llava-hf/llava-v1.6-mistral-7b-hf",
            task=task,
            env=env,
            instruction_path=str(DEFAULT_INSTRUCTION_PATH),
            max_steps=args.max_steps,
        )
    finally:
        env.close()

    print("\n=== RESULT ===")
    print(f"success: {result['success']}")
    print(f"num_steps: {result['num_steps']}")
    print(f"trajectory length: {len(result['trajectory'])}")
    print(f"step_activations: {len(result['step_activations'])} steps captured")
    if result["step_activations"]:
        first_step = result["step_activations"][0]
        print(f"  first step's captured layers: {list(first_step.keys())}")
        first_layer = next(iter(first_step))
        print(f"  shape at layer {first_layer}: {tuple(first_step[first_layer].shape)}")

    actions = [s for s in result["trajectory"] if isinstance(s, dict) and "action_type" in s]
    if actions:
        print(f"action sequence: {[a.get('action_type') for a in actions]}")
        print(f"final action (full dict, exact fields unverified -- dumping whole thing "
              f"rather than guessing a key name): {actions[-1]}")

    save_path = Path("logs") / f"episode_result_task{args.task_id}_{int(time.time())}.pt"
    torch.save({
        "task_id": task.task_id,
        "success": result["success"],
        "num_steps": result["num_steps"],
        "step_activations": result["step_activations"],
        "actions": actions,
    }, save_path)
    print(f"Saved full result to {save_path}")


if __name__ == "__main__":
    main()
