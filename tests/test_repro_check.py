"""analysis/repro_check.py compare: activations are compared only on turns whose
prompts are identical (after divergence the inputs differ), and a non-finite
value is never reported as bitwise equal."""
import gzip
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import repro_check as rc  # noqa: E402


def _t(vals):
    return [{31: torch.tensor(v, dtype=torch.float32)} for v in vals]


def test_act_diff_nonfinite_is_not_bitwise():
    a = _t([[1.0, 2.0]])
    b = _t([[float("nan"), 2.0]])
    r = rc._act_diff(a, b)
    assert r["comparable"] is False and r.get("nonfinite")


def test_act_diff_only_prompt_identical_turns():
    a = _t([[1.0, 2.0], [1.0, 2.0], [9.0, 9.0]])
    b = _t([[1.0, 2.0], [1.0, 2.0001], [0.0, 0.0]])
    r = rc._act_diff(a, b, n_turns=2)
    assert r["comparable"] and r["turns_compared"] == 2
    assert r["max_abs"] == pytest.approx(1e-4, rel=1e-2)
    assert rc._act_diff(a, b, n_turns=0)["comparable"] is False


def _episode(d, sid, prompts, comps, acts, gpu):
    tr = {"steps": [{"prompt_sha256": p, "completion": c} for p, c in zip(prompts, comps)]}
    (d / f"{sid}.transcript.json.gz").write_bytes(gzip.compress(json.dumps(tr).encode()))
    torch.save(_t(acts), d / f"{sid}.pt")
    return {"sample_id": sid, "attack_name": "clean", "task_id": "t", "asr_success": False,
            "utility_success": True, "tool_calls": [], "code_version": {"harness_sha256": "h"},
            "attack_sha256": "a", "run_settings": {"gpu_name": gpu}}


def test_compare_cross_hardware_divergent_episode(tmp_path):
    o, r = tmp_path / "orig", tmp_path / "rep"
    o.mkdir(); r.mkdir()
    # identical prompts for 2 steps, then the trajectories diverge
    ro = _episode(o, "s1", ["p0", "p1", "p2"], ["c0", "c1", "c2"],
                  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "RTX PRO 6000 Blackwell")
    rr = _episode(r, "s1", ["p0", "p1", "pX"], ["c0", "cY", "c2"],
                  [[1.0, 2.0], [3.0, 4.001], [50.0, 60.0]], "Quadro RTX 8000")
    (o / "attack_results.json").write_text(json.dumps([ro]))
    (r / "attack_results.json").write_text(json.dumps([rr]))
    s = rc.compare(o, r, tmp_path / "out", ["s1"])
    assert s["activations_compared"] == 1
    assert s["activations_max_abs"] == pytest.approx(1e-3, rel=1e-2)    # not the 54 after divergence
    ep = json.loads((tmp_path / "out" / "numbers.json").read_text())["episodes"][0]
    assert ep["prompt_identical_turns"] == 2 and ep["activations"]["turns_compared"] == 2
    assert "CROSS-HARDWARE" in (tmp_path / "out" / "report.md").read_text().splitlines()[0]



def test_compare_without_transcript_does_not_compare_activations(tmp_path):
    o, r = tmp_path / "o", tmp_path / "r"
    o.mkdir(); r.mkdir()
    row = {"sample_id": "s", "attack_name": "clean", "task_id": "t", "asr_success": False,
           "utility_success": True, "tool_calls": [], "run_settings": {}}
    for d in (o, r):
        (d / "attack_results.json").write_text(json.dumps([row]))
        torch.save(_t([[1.0], [2.0]]), d / "s.pt")
    rc.compare(o, r, tmp_path / "out", ["s"])
    ep = json.loads((tmp_path / "out" / "numbers.json").read_text())["episodes"][0]
    assert ep["activations"] == {"comparable": False, "reason": "no transcript"}
