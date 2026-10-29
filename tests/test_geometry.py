"""analysis/geometry.py on a tiny fixture with known activations: the join,
the per-condition displacement, and the RQ1 failed-vs-successful split."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import geometry as g  # noqa: E402


def _ep(d: Path, sid, vec_by_layer):
    torch.save([{L: torch.tensor(v, dtype=torch.float32) for L, v in vec_by_layer.items()}],
               d / f"{sid}.pt")


def _ep_turns(d: Path, sid, turns):
    """turns: list of {layer: vec}. Saves a multi-turn .pt."""
    torch.save([{L: torch.tensor(v, dtype=torch.float32) for L, v in t.items()} for t in turns],
               d / f"{sid}.pt")


def _umsg(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _transcript(d: Path, sid, steps, messages):
    """steps: list of dicts with at least prompt_sha256 (+ input, completion,
    tool_calls); messages: the shared message table. Shape matches what
    environments/agentdojo_llm.py's TranscriptRecorder writes, so
    attention_replay.check_arrival can read it."""
    tr = {"format": 1, "messages": messages, "steps": steps}
    (d / f"{sid}.transcript.json.gz").write_bytes(gzip.compress(json.dumps(tr).encode()))


def make(tmp: Path):
    d = tmp / "final_agentdojo_fix"
    d.mkdir()
    rows, L = [], 31
    # two user tasks, clean + important_instructions each; II succeeds on one,
    # fails on the other. Attacked state is displaced from clean in BOTH cases.
    specs = [
        ("user_task_0", True,  [0.0, 0.0]),
        ("user_task_1", False, [0.0, 0.0]),
    ]
    for ut, succ, clean_vec in specs:
        tid = f"workspace_{ut}_injection_task_0"
        rows.append({"sample_id": f"clean_{tid}", "condition": "clean", "attack_name": "clean",
                     "task_id": tid, "suite_name": "workspace", "asr_success": False, "error": None})
        _ep(d, f"clean_{tid}", {L: clean_vec})
        # failed attack still displaced by 3.0; successful one by 4.0
        shift = 4.0 if succ else 3.0
        rows.append({"sample_id": f"important_instructions_{tid}", "condition": "attacked",
                     "attack_name": "important_instructions", "task_id": tid, "suite_name": "workspace",
                     "asr_success": succ, "error": None})
        _ep(d, f"important_instructions_{tid}", {L: [clean_vec[0] + shift, clean_vec[1]]})
        # a defended row must be ignored by the undefended join
        rows.append({"sample_id": f"important_instructions_defended_icon_{tid}", "condition": "defended_attacked",
                     "attack_name": "important_instructions", "defense_name": "icon", "task_id": tid,
                     "suite_name": "workspace", "asr_success": False, "error": None})
        _ep(d, f"important_instructions_defended_icon_{tid}", {L: [9.0, 9.0]})
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_join_excludes_defended(tmp_path):
    d = make(tmp_path)
    paired = g.build_dataset([str(d)], "first", verbose=False)
    assert len(paired) == 2
    for conds in paired.values():
        assert set(conds) == {"clean", "important_instructions"}   # icon row excluded


def test_by_outcome_split(tmp_path):
    d = make(tmp_path)
    paired = g.build_dataset([str(d)], "first", verbose=False)
    oc = g.per_condition_geometry_by_outcome(paired)
    ii = oc["important_instructions"]
    assert set(ii) == {"success", "failure"}
    assert ii["success"][31]["n"] == 1 and ii["failure"][31]["n"] == 1
    # the key RQ1 fact: the FAILED attack still displaces, well above zero
    assert ii["failure"][31]["delta_mean"] == pytest.approx(3.0, abs=1e-4)
    assert ii["success"][31]["delta_mean"] == pytest.approx(4.0, abs=1e-4)


def test_geometry_matches_manual(tmp_path):
    d = make(tmp_path)
    paired = g.build_dataset([str(d)], "first", verbose=False)
    geom = g.per_condition_geometry(paired)
    m = geom["important_instructions"][31]
    assert m["n"] == 2
    assert m["delta_mean"] == pytest.approx(3.5, abs=1e-4)      # (3+4)/2


def make_payload_turn(tmp: Path):
    """A 2-turn IPI fixture: turn 0 identical clean vs attacked (same sha),
    the payload arrives at turn 1 (sha differs). Displacement must be read at
    turn 1, not turn 0 (which would be zero)."""
    d = tmp / "final_agentdojo_pt"
    d.mkdir()
    rows, L = [], 31
    for ut, succ, shift in [("user_task_0", True, 4.0), ("user_task_1", False, 3.0)]:
        tid = f"workspace_{ut}_injection_task_0"
        c, a = f"clean_{tid}", f"important_instructions_{tid}"
        rows += [
            {"sample_id": c, "condition": "clean", "attack_name": "clean", "task_id": tid,
             "suite_name": "workspace", "asr_success": False, "error": None},
            {"sample_id": a, "condition": "attacked", "attack_name": "important_instructions",
             "task_id": tid, "suite_name": "workspace", "asr_success": succ, "error": None},
        ]
        # turn 0 identical (both [1,1]); turn 1 clean=[2,2], attacked shifted on x
        _ep_turns(d, c, [{L: [1.0, 1.0]}, {L: [2.0, 2.0]}])
        _ep_turns(d, a, [{L: [1.0, 1.0]}, {L: [2.0 + shift, 2.0]}])
        # shared prefix (sys 0, user 1), identical step 0; step 1 adds a tool
        # result that differs only by the payload (msg 2 clean vs 3 attacked)
        sys_u = [{"role": "system", "content": [{"type": "text", "text": "s"}]}, _umsg("task")]
        tool_c = _umsg("[TOOL RESULT] {\"result\": \"ok\"}")
        tool_a = _umsg("[TOOL RESULT] {\"result\": \"ok INJECT\"}")
        st0 = {"prompt_sha256": "same", "input": [0, 1], "completion": "call f", "tool_calls": [{"function": "f"}]}
        _transcript(d, c, [st0, {"prompt_sha256": "cleanB", "input": [0, 1, 2], "completion": "", "tool_calls": []}],
                    sys_u + [tool_c])
        _transcript(d, a, [st0, {"prompt_sha256": "advB", "input": [0, 1, 2], "completion": "", "tool_calls": []}],
                    sys_u + [tool_a])
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_payload_turn_displacement(tmp_path):
    d = make_payload_turn(tmp_path)
    summ, exp = g.displacement_by_outcome_payload(str(d))
    ii = summ["important_instructions"]
    # measured at turn 1: failure shifted by 3, success by 4 -- NOT zero
    assert ii["failure"][31]["delta_mean"] == pytest.approx(3.0, abs=1e-4)
    assert ii["success"][31]["delta_mean"] == pytest.approx(4.0, abs=1e-4)
    assert exp["important_instructions"]["exposed"] == 2
    assert exp["important_instructions"]["not_exposed"] == 0


def test_payload_turn_not_exposed(tmp_path):
    """If the prompts never diverge, the episode is counted as not-exposed,
    not bucketed with a spurious zero displacement."""
    d = make_payload_turn(tmp_path)
    sid = "important_instructions_workspace_user_task_0_injection_task_0"
    sys_u = [{"role": "system", "content": [{"type": "text", "text": "s"}]}, _umsg("task")]
    st0 = {"prompt_sha256": "same", "input": [0, 1], "completion": "call f", "tool_calls": [{"function": "f"}]}
    # identical to the clean transcript -> never diverges
    _transcript(d, sid, [st0, {"prompt_sha256": "cleanB", "input": [0, 1, 2], "completion": "", "tool_calls": []}],
                sys_u + [_umsg("[TOOL RESULT] {\"result\": \"ok\"}")])
    # identical prompts under greedy decoding give identical activations
    _ep_turns(d, sid, [{31: [1.0, 1.0]}, {31: [2.0, 2.0]}])
    summ, exp = g.displacement_by_outcome_payload(str(d))
    assert exp["important_instructions"]["not_exposed"] == 1
    assert exp["important_instructions"]["exposed"] == 1


def test_payload_turn_identical_prompts_but_different_activations(tmp_path):
    """Text-identical prompts whose activations nevertheless differ cannot
    happen on AgentDojo under greedy decoding: counted as an anomaly
    (prompt_same_acts_differ), neither exposed nor silently unexposed."""
    d = make_payload_turn(tmp_path)
    sid = "important_instructions_workspace_user_task_0_injection_task_0"
    sys_u = [{"role": "system", "content": [{"type": "text", "text": "s"}]}, _umsg("task")]
    st0 = {"prompt_sha256": "same", "input": [0, 1], "completion": "call f", "tool_calls": [{"function": "f"}]}
    _transcript(d, sid, [st0, {"prompt_sha256": "cleanB", "input": [0, 1, 2], "completion": "", "tool_calls": []}],
                sys_u + [_umsg("[TOOL RESULT] {\"result\": \"ok\"}")])     # .pt still displaced
    summ, exp = g.displacement_by_outcome_payload(str(d))
    assert exp["important_instructions"]["prompt_same_acts_differ"] == 1
    assert exp["important_instructions"]["exposed"] == 1
    assert exp["important_instructions"]["not_exposed"] == 0


def test_payload_turn_rejects_pre_payload_divergence(tmp_path):
    """A divergence at step 0 (not a tool result) is NOT the payload arriving;
    it must be counted not_payload_arrival, never measured at turn 0."""
    d = make_payload_turn(tmp_path)
    sid = "important_instructions_workspace_user_task_1_injection_task_0"
    sys_u = [{"role": "system", "content": [{"type": "text", "text": "s"}]}, _umsg("task PERTURBED")]
    # step 0 prompt already differs from the clean pair's "same"
    st0 = {"prompt_sha256": "DIFFERENT0", "input": [0, 1], "completion": "call f", "tool_calls": [{"function": "f"}]}
    _transcript(d, sid, [st0, {"prompt_sha256": "advB", "input": [0, 1, 2], "completion": "", "tool_calls": []}],
                sys_u + [_umsg("[TOOL RESULT] {\"result\": \"ok INJECT\"}")])
    summ, exp = g.displacement_by_outcome_payload(str(d))
    # user_task_1 (the failure case) is now rejected; only user_task_0 remains
    assert exp["important_instructions"]["not_payload_arrival"] == 1
    assert "failure" not in summ.get("important_instructions", {})
