"""
experiments/qwen3vl_diagnostic.py
------------------------------------
Standalone, small-scale diagnostic -- NOT part of the main attack
matrix. Answers one specific question: does Qwen3-VL-8B engage more
deeply (more turns, more/different tool calls) than LLaVA-NeXT 7B did
on the SAME 5 workspace tasks that LLaVA-NeXT already handled well
(UA=True, ASR=False, but capped at exactly 2 turns every time)?

Deliberately does NOT reuse run_attack_matrix.py's generic matrix
loop -- that loop iterates "first N tasks" in whatever order the suite
dict happens to produce, which is NOT the same as "these 5 specific
user_task_ids" (confirmed earlier: max_tasks=9 was needed just to reach
user_task_1). This script filters directly to the 5 tasks that matter
for this comparison, keeping the run small and fast.

Runs clean + template_ipi (cheap, no GCG optimization needed for this
question) for each task, using the SAME injection-vector-selection
pattern as the main matrix (first vector from
suite.get_injection_vector_defaults(), consistent across all injection
pairings within a suite).

Usage (on the cluster, inside an sbatch job or interactive session):
    python3 experiments/qwen3vl_diagnostic.py

Prints a compact, directly-pasteable comparison table -- no need to
paste raw output back, same as the main matrix's summary script.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import argparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.hooks import ActivationExtractor
from extraction.model_loader import load_llavanext_7b, load_qwen2vl_7b, load_internvl_8b, load_qwen3vl_8b
from environments.agentdojo_tasks import load_tasks, run_agentdojo_episode
from attacks.template_ipi import TemplateIPIAttack

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_USER_TASK_IDS = {"user_task_14", "user_task_17", "user_task_24", "user_task_26", "user_task_28"}

# Real, direct function references -- not a name-based dispatch through
# getattr like run_attack_matrix.py's _load_model_by_name, since this
# script only ever needs to call exactly one of these, once, per run.
_MODEL_LOADERS = {
    "llava": load_llavanext_7b,
    "qwen2vl": load_qwen2vl_7b,
    "internvl": load_internvl_8b,
    "qwen3vl": load_qwen3vl_8b,
}

# What LLaVA-NeXT's original completed matrix run showed on these same 5
# tasks BEFORE the query()/​_parse_model_output fixes -- printed for
# comparison when checking a DIFFERENT model. Not printed when
# re-checking LLaVA-NeXT itself against those same fixes -- comparing a
# model's fixed run against its own pre-fix numbers is the actual point
# of that specific recheck, not a fixed comparison table.
LLAVA_NEXT_PRE_FIX_REFERENCE = {
    "condition": "clean AND template_ipi, all 5 tasks, BEFORE the query()/_parse_model_output fixes",
    "utility_success": True,
    "asr_success": False,
    "num_turns": 2,
    "note": "identical num_turns=2 for clean vs template_ipi on every one of these "
            "5 tasks -- the injected suffix never measurably changed behavior. "
            "Whether this holds AFTER the parsing fixes is exactly what a --model llava "
            "run of this script now checks directly, rather than assuming either way.",
}


def main(model_name: str):
    logger.info("Loading %s...", model_name)
    model, processor = _MODEL_LOADERS[model_name](hpc=True)

    extractor = ActivationExtractor(model, target_layers=None, token_aggregation="last")
    logger.info(
        "Resolved target_layers=%s (%d layers hooked) -- sanity-check this looks "
        "reasonable before trusting anything else from this run.",
        extractor.target_layers, len(extractor.target_layers),
    )

    tasks, context = load_tasks(suite_names=["workspace"], pair_with_injections=True)
    selected = [t for t in tasks if t.user_task_id in TARGET_USER_TASK_IDS]
    # One injection_task per user_task is enough for this diagnostic --
    # the question is about ENGAGEMENT DEPTH (turns/tools), which the
    # main run already showed doesn't vary by injection_task for a given
    # user_task under LLaVA-NeXT. Keep it to the first pairing per task.
    seen_user_tasks = set()
    filtered = []
    for t in selected:
        if t.user_task_id not in seen_user_tasks:
            filtered.append(t)
            seen_user_tasks.add(t.user_task_id)

    logger.info("Selected %d tasks: %s", len(filtered), [t.user_task_id for t in filtered])

    attack = TemplateIPIAttack()
    results = []

    for task in filtered:
        suite = context["suites"][task.suite_name]
        injection_defaults = suite.get_injection_vector_defaults()
        vector_id = next(iter(injection_defaults))
        clean_vector_text = injection_defaults[vector_id]

        logger.info("--- %s ---", task.task_id)

        clean_out = run_agentdojo_episode(model, processor, extractor, task, context, injections={})
        logger.info(
            "  clean: UA=%s num_turns=%d tool_calls=%s",
            clean_out["utility_success"], len(clean_out["turn_activations"]), clean_out["turn_tool_calls"],
        )

        adv_result = attack.generate(
            {"page_text": clean_vector_text, "injectable_field": "page_text"},
            target_behavior=task.target_behavior,
        )
        attacked_out = run_agentdojo_episode(
            model, processor, extractor, task, context, injections={vector_id: adv_result.adversarial_text},
        )
        logger.info(
            "  template_ipi: ASR=%s UA=%s num_turns=%d tool_calls=%s",
            attacked_out["asr_success"], attacked_out["utility_success"],
            len(attacked_out["turn_activations"]), attacked_out["turn_tool_calls"],
        )

        results.append({
            "task_id": task.task_id,
            "clean_utility_success": clean_out["utility_success"],
            "clean_num_turns": len(clean_out["turn_activations"]),
            "clean_tool_calls": clean_out["turn_tool_calls"],
            "clean_raw_completions": clean_out["turn_raw_completions"],
            "attacked_asr_success": attacked_out["asr_success"],
            "attacked_utility_success": attacked_out["utility_success"],
            "attacked_num_turns": len(attacked_out["turn_activations"]),
            "attacked_tool_calls": attacked_out["turn_tool_calls"],
            "attacked_raw_completions": attacked_out["turn_raw_completions"],
        })

    out_path = Path(f"{model_name}_diagnostic_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved full results to %s", out_path)

    print()
    print("=" * 70)
    print(f"SUMMARY -- {model_name} on the 5 tasks LLaVA-NeXT originally capped at 2 turns")
    print("=" * 70)
    if model_name != "llava":
        print(f"LLaVA-NeXT pre-fix reference: {LLAVA_NEXT_PRE_FIX_REFERENCE}")
        print()
    for r in results:
        turns_changed = r["clean_num_turns"] != r["attacked_num_turns"]
        exceeded_two = max(r["clean_num_turns"], r["attacked_num_turns"]) > 2
        print(
            f"{r['task_id']:35s} clean_turns={r['clean_num_turns']} "
            f"attacked_turns={r['attacked_num_turns']} "
            f"{'<<< EXCEEDS 2 TURNS' if exceeded_two else ''}"
            f"{'  <<< TURNS DIFFER CLEAN VS ATTACKED' if turns_changed else ''}"
        )
    any_exceeded = any(max(r["clean_num_turns"], r["attacked_num_turns"]) > 2 for r in results)
    any_turns_differ = any(r["clean_num_turns"] != r["attacked_num_turns"] for r in results)
    print()
    print(f"Any task exceeded 2 turns: {any_exceeded}")
    print(f"Any task showed different turn count clean vs attacked: {any_turns_differ}")
    if model_name == "llava":
        print()
        print(f"Pre-fix reference for comparison: {LLAVA_NEXT_PRE_FIX_REFERENCE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default="qwen3vl", choices=list(_MODEL_LOADERS),
        help="Which model to run this 5-task diagnostic against. Default: qwen3vl "
             "(this script's original purpose) -- pass --model llava to recheck "
             "LLaVA-NeXT's OWN behavior on these same 5 tasks against the "
             "query()/_parse_model_output fixes, using the exact same 5-task, "
             "minutes-not-hours diagnostic rather than a throwaway /tmp script.",
    )
    args = parser.parse_args()
    main(args.model)
