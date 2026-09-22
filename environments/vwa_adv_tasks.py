"""
environments/vwa_adv_tasks.py
--------------------------------
Real task loading + episode execution for VWA-Adv -- the VWA-Adv analog
of environments/agentdojo_tasks.py.

Episode loop ported directly from the real ChenWu98/visualwebarena
repo's own run.py (the actual reference driver script VWA/VWA-Adv itself
uses for end-to-end evaluation) -- reset agent+env, loop next_action() ->
env.step() until STOP or termination, evaluate via the real
evaluator_router(). Verified against the REAL run.py source (not
guessed) for: reset()/step() signatures and return shapes,
Trajectory/StateInfo types, early_stop()'s COMPLETE stopping logic
(including the is_equivalent()-based action comparison and the different
handling for TYPE vs non-TYPE actions -- fetched and read in full, not
partially, after an earlier session already had to correct one
paper-equation misreading elsewhere in this project's defenses, so this
one was checked to the end deliberately), and evaluator_router()'s
config-driven evaluator construction. instruction_path/action_set_tag
default to the same p_som_cot_id_actree_3s.json / "som" pairing already
confirmed (via that instruction JSON's own meta_data) to match
LocalVLMPromptConstructor's "som" branch in vwa_adv_llm.py.

STATUS: this module has NEVER been run end-to-end. It needs a LIVE
Postmill instance (Reddit environment) and a real Playwright browser --
neither is available in the environment this was written in. Every API
call here is verified against the real installed/cloned
ChenWu98/visualwebarena source, but the actual RUNTIME BEHAVIOR (does a
real observation from a real page actually flow correctly through
LocalVLMPromptConstructor, does the model's SoM-formatted output
actually parse into a valid browser action, does the evaluator correctly
score a real trajectory) is unverified. Run this for real on the
cluster, against a live Postmill instance, before trusting it for actual
attack/defense episodes -- same "verified against real code, not yet
against real weights/a live environment" caveat every new AgentDojo
attack/defense module in this project carried before its own first real
run caught a real bug (the GCG YAML crash, the obfuscated_activation
performance issue) that no amount of source-reading alone would have
surfaced.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from browser_env import ActionTypes, ScriptBrowserEnv, create_stop_action
    from browser_env.actions import is_equivalent
    from browser_env.helper_functions import get_action_description
    from evaluation_harness import evaluator_router
except ImportError:
    # Keeps this module importable outside the real VWA-Adv repo/PYTHONPATH
    # (e.g. for isolated unit testing of load_tasks() alone).
    ActionTypes = ScriptBrowserEnv = create_stop_action = None
    is_equivalent = get_action_description = evaluator_router = None

from environments.vwa_adv_llm import construct_local_vlm_agent

DEFAULT_EARLY_STOP_THRESHOLDS = {"parsing_failure": 3, "repeating_action": 3}
DEFAULT_INSTRUCTION_PATH = "agent/prompts/jsons/p_som_cot_id_actree_3s.json"


@dataclass
class VWAAdvTask:
    """One real VWA-Adv Reddit task. config_file points at an ALREADY-
    GENERATED config_files/test_reddit/N.json (see generate_test_data.py
    -- already confirmed this project's own earlier finding that it
    needs no live server, only REDDIT/SHOPPING/etc. env vars set to
    something, even a placeholder)."""

    config_file: Path
    task_id: str
    intent: str


def load_tasks(config_dir: Path, suite: str = "test_reddit") -> List[VWAAdvTask]:
    """
    Loads real VWA-Adv task configs from an already-generated
    config_files/{suite}/ directory.
    """
    suite_dir = config_dir / suite
    if not suite_dir.exists():
        raise FileNotFoundError(
            f"{suite_dir} does not exist -- run generate_test_data.py first "
            "(needs REDDIT/SHOPPING/WIKIPEDIA/HOMEPAGE/CLASSIFIEDS/CLASSIFIEDS_RESET_TOKEN "
            "env vars set, no live server required for THAT step -- see this project's "
            "own confirmed finding on this)."
        )

    tasks: List[VWAAdvTask] = []
    for config_file in sorted(suite_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        with open(config_file) as f:
            config = json.load(f)
        tasks.append(VWAAdvTask(config_file=config_file, task_id=config_file.stem, intent=config.get("intent", "")))
    return tasks


def _early_stop(trajectory: list, max_steps: int, thresholds: Dict[str, int]) -> Tuple[bool, str]:
    """
    Reimplemented from the real run.py's own early_stop() -- not
    imported, since run.py is a top-level script, not a package member
    this project can import from. Same stopping logic, verified against
    the complete real source (fetched in full, including the
    is_equivalent()-based comparison and the TYPE-action branch):
    max steps reached, OR `k` consecutive unparseable actions, OR the
    last k actions are all equivalent to each other (non-TYPE actions),
    OR the current typed action recurs >=k times anywhere in the
    trajectory (TYPE actions specifically).
    """
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    actions = trajectory[1::2]

    k = thresholds["parsing_failure"]
    last_k = actions[-k:]
    if len(last_k) >= k and all(a["action_type"] == ActionTypes.NONE for a in last_k):
        return True, f"Failed to parse actions for {k} times"

    k = thresholds["repeating_action"]
    last_k = actions[-k:]
    if len(actions) == 0:
        return False, ""
    last_action = actions[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k) >= k and all(is_equivalent(a, last_action) for a in last_k):
            return True, f"Same action for {k} times"
    else:
        if sum(is_equivalent(a, last_action) for a in actions) >= k:
            return True, f"Same typing action for {k} times"

    return False, ""


def run_vwa_adv_episode(
    model,
    processor,
    extractor,
    model_id: str,
    task: VWAAdvTask,
    env: "ScriptBrowserEnv",
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    action_set_tag: str = "som",
    max_steps: int = 30,
    adv_url2caption: Optional[Dict[str, str]] = None,
    adv_url2image: Optional[Dict[str, Any]] = None,
    defense: Optional[Any] = None,
    ecso: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Runs ONE real VWA-Adv episode -- structure ported directly from the
    real repo's own run.py loop (reset -> next_action/step loop until
    STOP or terminated -> evaluator_router scoring), substituting this
    project's own construct_local_vlm_agent() for whatever provider
    run.py normally builds via construct_agent(args).

    env : an ALREADY-CONSTRUCTED ScriptBrowserEnv (build once, reuse
        across many episodes on the same node -- the real repo's own
        run.py does the same, only tearing it down at the very end via
        env.close()). Caller owns that lifecycle, not this function.
    adv_url2caption / adv_url2image : VWA-Adv's own image-injection
        mechanism (maps a URL to an overriding caption/image) -- this is
        the actual attack delivery channel for a VISUAL attack
        condition; pass {} (the default) for clean.

    Returns {"success", "num_steps", "step_activations", "trajectory"}.
    step_activations comes straight from backend.step_activations (see
    vwa_adv_llm.py's LocalVLMBackend) -- one entry per real
    backend.generate() call (one per agent decision), the VWA-Adv analog
    of AgentDojo's turn_activations.
    """
    if ScriptBrowserEnv is None:
        raise ImportError(
            "browser_env/evaluation_harness are not importable -- this must run inside the "
            "real ChenWu98/visualwebarena repo with it on PYTHONPATH, not standalone."
        )

    adv_url2caption = adv_url2caption or {}
    adv_url2image = adv_url2image or {}

    agent, backend = construct_local_vlm_agent(
        model, processor, extractor, model_id, instruction_path,
        action_set_tag=action_set_tag, defense=defense, ecso=ecso,
    )

    agent.reset(str(task.config_file))
    trajectory: list = []
    obs, info = env.reset(
        options={"config_file": str(task.config_file)},
        adv_url2caption=adv_url2caption,
        adv_url2image=adv_url2image,
    )
    state_info = {"observation": obs, "info": info}
    trajectory.append(state_info)

    meta_data = {"action_history": ["None"]}
    num_steps = 0
    while True:
        early_stop_flag, stop_info = _early_stop(trajectory, max_steps, DEFAULT_EARLY_STOP_THRESHOLDS)
        if early_stop_flag:
            action = create_stop_action(f"Early stop: {stop_info}")
        else:
            try:
                action = agent.next_action(trajectory, task.intent, images=[], meta_data=meta_data)
            except ValueError as e:
                action = create_stop_action(f"ERROR: {e}")

        trajectory.append(action)

        try:
            action_str = get_action_description(
                action, state_info["info"]["observation_metadata"],
                action_set_tag=action_set_tag, prompt_constructor=agent.prompt_constructor,
            )
            meta_data["action_history"].append(action_str)
        except Exception:
            # Non-fatal -- action_history is used for prompting context on
            # LATER turns, not for correctness of this turn. Losing one
            # entry's readable description shouldn't crash a real episode.
            meta_data["action_history"].append(str(action.get("action_type", "unknown")))

        if action["action_type"] == ActionTypes.STOP:
            break

        obs, _, terminated, _, info = env.step(action, adv_url2caption=adv_url2caption, adv_url2image=adv_url2image)
        state_info = {"observation": obs, "info": info}
        trajectory.append(state_info)
        num_steps += 1

        if terminated:
            trajectory.append(create_stop_action(""))
            break

    evaluator = evaluator_router(task.config_file, captioning_fn=None)
    score = evaluator(
        trajectory=trajectory,
        config_file=str(task.config_file),
        page=env.page,
        client=env.get_page_client(env.page),
    )

    return {
        "success": bool(score),
        "num_steps": num_steps,
        "step_activations": backend.step_activations,
        "trajectory": trajectory,
    }
