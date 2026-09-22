"""
experiments/vwa_determinism_check.py
------------------------------------
Is VWA-Adv episode execution deterministic?

WHY
    Comparing two full curated runs (vwa_adv_qwen3vl vs
    vwa_adv_qwen3vl_strict), 9 ASR verdicts flipped on tasks whose
    evaluator config was NOT rewritten, and 5 of those changed num_steps:
        reddit_36_wrong_object_cap_1  template_ipi  2 -> 1
        reddit_36_wrong_object_cap_2  typographic   1 -> 3
        reddit_41_wrong_object_cap_2  comet         2 -> 4
    Same task, same attack, same model, greedy decoding patched at
    vwa_adv_llm.py:290. The agent took a different path.

    Two candidate causes with very different consequences:
      (a) Environment state. Postmill's DB is rebuilt per job, but
          WITHIN a run, comments from earlier tasks accumulate. The
          affected tasks are all url_match "navigate to the first/top
          post containing X", so page ordering could genuinely differ.
          Benign, but it means url_match tasks are order-dependent and
          are not independent samples.
      (b) Residual non-determinism in generation or observation. SoM
          element ids are known to be unstable across renders in this
          project; if the agent sees different ids it acts differently
          even under greedy decoding.

    determinism_check.py only ever covered AgentDojo. This is the
    VWA-Adv equivalent.

DESIGN
    Runs ONE task twice back to back in the same process, same model
    instance, same browser env. Defaults to a NAVIGATION task
    (url_match) rather than a comment task, so the first repeat cannot
    change the database the second repeat reads -- which isolates model
    and observation determinism from environment state.

    Compares, in order of diagnostic value:
      1. the raw action sequence (action_type + element/text)
      2. num_steps
      3. the evaluator verdict

    Identical actions with differing verdicts implicates the evaluator
    or page state, not generation. Differing actions implicates
    generation or the observation pipeline.

Run INSIDE the job serving Postmill.

    python3 experiments/vwa_determinism_check.py --task reddit_36_wrong_object_cap_1
    python3 experiments/vwa_determinism_check.py --task ... --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VWA_ADV_REPO = Path(os.environ.get(
    "VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
AGENT_ATTACK_REPO = Path(os.environ.get(
    "AGENT_ATTACK_REPO", "/datasets/nmahomed/vwa-adv/agent-attack"))
sys.path.insert(0, str(VWA_ADV_REPO))


def summarise_action(act) -> str:
    """Stable, comparable one-line form of a browser action."""
    if not isinstance(act, dict):
        return f"<{type(act).__name__}>"
    at = act.get("action_type")
    name = getattr(at, "name", str(at))
    bits = [name]
    for k in ("element_id", "element_role", "element_name", "text",
              "url", "answer", "direction", "key_comb"):
        v = act.get(k)
        if v not in (None, "", [], 0):
            bits.append(f"{k}={v}")
    return " ".join(str(b) for b in bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="reddit_36_wrong_object_cap_1",
                    help="curated task dir name; default is a url_match "
                         "NAVIGATION task so repeats cannot mutate the DB")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=15)
    a = ap.parse_args()

    import torch  # noqa: F401
    from browser_env import ScriptBrowserEnv
    from extraction.hooks import ActivationExtractor, _resolve_target_layers
    from extraction.model_loader import (
        load_llavanext_7b, load_internvl_8b, load_qwen3vl_8b)
    _LOADERS = {"Qwen/Qwen3-VL-8B-Instruct": load_qwen3vl_8b,
                "OpenGVLab/InternVL3-8B-hf": load_internvl_8b,
                "llava-hf/llava-v1.6-mistral-7b-hf": load_llavanext_7b}
    from environments.vwa_adv_tasks import VWAAdvTask, run_vwa_adv_episode
    from experiments.run_vwa_adv_curated_matrix import CuratedTask

    exp = AGENT_ATTACK_REPO / "exp_data"
    task = CuratedTask(exp / "agent_adv" / a.task, exp)
    print(f"task   : {a.task}")
    print(f"config : {task.adv_config}")
    print(f"intent : {task.intent}")
    print(f"repeats: {a.repeats}\n")

    if a.model_id not in _LOADERS:
        raise SystemExit(f"no loader for {a.model_id}; have {list(_LOADERS)}")
    model, processor = _LOADERS[a.model_id](hpc=True)
    model.eval()
    extractor = ActivationExtractor(
        model, target_layers=_resolve_target_layers(model, None),
        token_aggregation="last")

    os.chdir(str(AGENT_ATTACK_REPO))
    env = ScriptBrowserEnv(headless=True, observation_type="image_som",
                           current_viewport_only=False,
                           viewport_size={"width": 1280, "height": 2048})

    runs = []
    try:
        for i in range(a.repeats):
            vt = VWAAdvTask(config_file=task.adv_config,
                            task_id=task.name, intent=task.intent)
            out = run_vwa_adv_episode(
                model, processor, extractor, a.model_id, vt, env,
                instruction_path=str(VWA_ADV_REPO /
                                     "agent/prompts/jsons/p_som_cot_id_actree_3s.json"),
                max_steps=a.max_steps, adv_url2caption={}, adv_url2image={})
            acts = [summarise_action(s) for s in out.get("trajectory", [])
                    if isinstance(s, dict) and "action_type" in s]
            runs.append({"actions": acts,
                         "num_steps": out.get("num_steps"),
                         "success": out.get("success")})
            print(f"--- repeat {i + 1}: num_steps={out.get('num_steps')} "
                  f"score={out.get('success')}  ({len(acts)} actions)")
            for j, s in enumerate(acts):
                print(f"      [{j}] {s}")
    finally:
        env.close()

    base = runs[0]
    same_actions = all(r["actions"] == base["actions"] for r in runs[1:])
    same_steps = all(r["num_steps"] == base["num_steps"] for r in runs[1:])
    same_score = all(r["success"] == base["success"] for r in runs[1:])

    print("\n" + "=" * 64)
    print(f"identical action sequences : {same_actions}")
    print(f"identical num_steps        : {same_steps}")
    print(f"identical evaluator verdict: {same_score}")
    if same_actions and not same_score:
        print("\n-> Generation IS deterministic. The verdict moved without the")
        print("   agent moving, so the variation is in the evaluator or in live")
        print("   page state, not the model. url_match tasks are then")
        print("   order-dependent and are not independent samples.")
    elif not same_actions:
        print("\n-> Generation or the observation pipeline is NOT deterministic")
        print("   in this environment, despite do_sample=False at")
        print("   vwa_adv_llm.py:290. First place to look: SoM element ids,")
        print("   which this project has already shown are unstable across")
        print("   renders. Compare the diverging step's element_id above.")
        for j in range(max(len(r["actions"]) for r in runs)):
            vals = {tuple(r["actions"][j:j + 1]) for r in runs}
            if len(vals) > 1:
                print(f"\n   first divergence at step {j}:")
                for i, r in enumerate(runs):
                    v = r["actions"][j] if j < len(r["actions"]) else "<no action>"
                    print(f"     repeat {i + 1}: {v}")
                break
    else:
        print("\n-> Fully deterministic on this task.")
    print("=" * 64)

    Path("analysis_out").mkdir(exist_ok=True)
    out_p = Path("analysis_out") / f"vwa_determinism_{a.task}.json"
    json.dump({"task": a.task, "model": a.model_id, "runs": runs,
               "same_actions": same_actions, "same_steps": same_steps,
               "same_score": same_score},
              open(PROJECT_ROOT / out_p, "w"), indent=2)
    print(f"wrote {out_p}")
    sys.exit(0 if same_actions else 1)


if __name__ == "__main__":
    main()
