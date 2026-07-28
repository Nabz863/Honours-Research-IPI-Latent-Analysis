"""
experiments/agentdojo_smoketest.py
-------------------------------------
Real end-to-end smoke test: one real loaded model, one real AgentDojo
task, clean vs attacked, through the full LocalVLMPipelineElement +
TaskSuite.run_task_with_pipeline() chain.

Requires agentdojo installed in THIS SAME environment as the main
project's torch/transformers -- see the note in the chat that pointed
you here for why (model_loader.py's loaded model and agentdojo's
pipeline/task-suite code must run in the same process).

Picks the FIRST available injection vector in the suite for simplicity
-- which vector a given attack should target is a real methodological
question for the full experiment (AgentDojo may have its own
convention for this via an `agentdojo.attacks` module we haven't
inspected yet), but for a mechanics smoke test, any valid vector is
fine.

One thing this does NOT verify, and can't until it actually runs: some
LLaVA-family chat templates don't natively support a "system" role turn
(folding it into the first user turn instead, or dropping it). If the
attacked episode's tool-documentation system prompt silently vanishes,
that's the first thing to check.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from attacks.template_ipi import TemplateIPIAttack
from environments.agentdojo_tasks import load_tasks, run_agentdojo_episode
from extraction.hooks import ActivationExtractor, _resolve_target_layers
from extraction.model_loader import load_llavanext_7b


def main():
    print("=== Loading model ===")
    model, processor = load_llavanext_7b(hpc=True)
    target_layers = _resolve_target_layers(model, None)
    print(f"Target layers: {target_layers}")

    print("\n=== Loading one real AgentDojo task ===")
    tasks, context = load_tasks(benchmark_version="v1", suite_names=["workspace"], pair_with_injections=True)
    task = tasks[0]
    print(f"task_id: {task.task_id}")
    print(f"prompt: {task.prompt}")
    print(f"target_behavior (injection GOAL): {task.target_behavior}")

    suite = context["suites"][task.suite_name]
    injection_defaults = suite.get_injection_vector_defaults()
    vector_id = next(iter(injection_defaults))
    clean_vector_text = injection_defaults[vector_id]
    print(f"\nUsing injection vector: {vector_id!r}")
    print(f"Clean default text: {clean_vector_text!r}")

    # Generate the actual attack payload with the already-tested Template
    # IPI attack, rather than hand-writing an injection string here.
    attack = TemplateIPIAttack()
    attack_result = attack.generate(
        {"page_text": clean_vector_text, "injectable_field": "page_text"},
        target_behavior=task.target_behavior,
    )
    print(f"Generated adversarial payload: {attack_result.adversarial_text!r}")

    extractor = ActivationExtractor(model, target_layers=target_layers, token_aggregation="last")

    print("\n=== Running CLEAN episode ===")
    clean_result = run_agentdojo_episode(model, processor, extractor, task, context, injections={})
    print(
        f"utility_success={clean_result['utility_success']}  "
        f"asr_success={clean_result['asr_success']}  "
        f"turns={len(clean_result['turn_activations'])}"
    )

    print("\n=== Running ATTACKED episode ===")
    attacked_result = run_agentdojo_episode(
        model,
        processor,
        extractor,
        task,
        context,
        injections={vector_id: attack_result.adversarial_text},
    )
    print(
        f"utility_success={attacked_result['utility_success']}  "
        f"asr_success={attacked_result['asr_success']}  "
        f"turns={len(attacked_result['turn_activations'])}"
    )

    print("\n=== Saving activations ===")
    out_dir = Path("agentdojo_smoketest_output")
    out_dir.mkdir(exist_ok=True)
    torch.save(clean_result["turn_activations"], out_dir / f"clean_{task.task_id}.pt")
    torch.save(attacked_result["turn_activations"], out_dir / f"attacked_{task.task_id}.pt")

    if clean_result["turn_activations"]:
        first_turn = clean_result["turn_activations"][0]
        one_layer = next(iter(first_turn))
        print(f"Sample activation shape (layer {one_layer}, clean, turn 0): {tuple(first_turn[one_layer].shape)}")

    print("\nDone. Check agentdojo_smoketest_output/ for saved per-turn activations.")


if __name__ == "__main__":
    main()
