"""
experiments/determinism_check.py
------------------------------------
Runs ONE task's clean episode TWICE, back to back, same model instance,
same code, same everything -- to answer a specific question raised by
comparing pre-fix and post-fix LLaVA-NeXT behavior on task_14 (a
correct email search vs. a hallucinated calendar event on two different
invocations): is generation actually deterministic, or is every
single-sample comparison in this project (before/after any fix, model A
vs model B, attacked vs clean) potentially just sampling noise dressed
up as a real effect?

If completion_1 == completion_2 exactly: generation IS deterministic
(or at least consistently seeded) -- the original run's different
behavior on this task came from something else changing (prompt
construction, model version, etc.), not sampling, and that's worth
tracking down specifically.

If they differ: generation is genuinely stochastic (the model's own
generation_config.json almost certainly has do_sample=true with no
temperature=0 override anywhere in this pipeline) -- meaning every
single-sample UA/ASR/behavior comparison collected so far, in either
direction, needs to be treated as one noisy draw, not a stable fact,
until generation is made deterministic and re-run.

Usage:
    python3 experiments/determinism_check.py --model llava
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.hooks import ActivationExtractor
from extraction.model_loader import load_llavanext_7b, load_qwen2vl_7b, load_internvl_8b, load_qwen3vl_8b
from environments.agentdojo_tasks import load_tasks, run_agentdojo_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MODEL_LOADERS = {
    "llava": load_llavanext_7b,
    "qwen2vl": load_qwen2vl_7b,
    "internvl": load_internvl_8b,
    "qwen3vl": load_qwen3vl_8b,
}


def main(model_name: str, user_task_id: str):
    logger.info("Loading %s...", model_name)
    model, processor = _MODEL_LOADERS[model_name](hpc=True)
    extractor = ActivationExtractor(model, target_layers=None, token_aggregation="last")

    tasks, context = load_tasks(suite_names=["workspace"], pair_with_injections=True)
    task = next(t for t in tasks if t.user_task_id == user_task_id)
    logger.info("Running %s TWICE, back to back, same model instance...", task.task_id)

    out1 = run_agentdojo_episode(model, processor, extractor, task, context, injections={})
    out2 = run_agentdojo_episode(model, processor, extractor, task, context, injections={})

    completions_1 = out1["turn_raw_completions"]
    completions_2 = out2["turn_raw_completions"]
    tool_calls_1 = out1["turn_tool_calls"]
    tool_calls_2 = out2["turn_tool_calls"]

    identical = completions_1 == completions_2

    print()
    print("=" * 70)
    print(f"DETERMINISM CHECK -- {model_name} on {task.task_id}, run twice")
    print("=" * 70)
    print(f"Run 1: num_turns={len(completions_1)}, UA={out1['utility_success']}, tool_calls={tool_calls_1}")
    print(f"Run 2: num_turns={len(completions_2)}, UA={out2['utility_success']}, tool_calls={tool_calls_2}")
    print()
    print(f"Completions IDENTICAL across both runs: {identical}")
    if not identical:
        print()
        print("Generation is NOT deterministic. Every single-sample comparison")
        print("collected so far (before/after fixes, model vs model, attacked vs")
        print("clean) needs to be treated as one noisy draw, not a stable fact,")
        print("until do_sample=False (or equivalent) is set and data is re-collected.")
    else:
        print()
        print("Generation IS deterministic (at least for this task, this model).")
        print("Whatever changed task_14's behavior between the original matrix run")
        print("and the post-fix recheck was NOT sampling noise -- worth tracking")
        print("down what else differed (prompt construction, code changes made")
        print("between those two runs, etc.) rather than attributing it to chance.")
    return 0 if identical else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="llava", choices=list(_MODEL_LOADERS))
    parser.add_argument(
        "--user_task_id", type=str, default="user_task_14",
        help="Which workspace user_task_id to check -- default is task_14, the one "
             "that showed different behavior between the original matrix run and "
             "the post-fix recheck.",
    )
    args = parser.parse_args()
    sys.exit(main(args.model, args.user_task_id))
