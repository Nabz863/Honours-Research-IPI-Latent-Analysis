"""analysis/repro_check.py on a fixture: selection is fixed and eligible-only;
compare finds identical and differing episodes."""
import gzip
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import repro_check as rc  # noqa: E402


def make(d: Path, perturb=None, users=None, gpu="RTX PRO 6000"):
    d.mkdir()
    rows = []
    users = users or {"workspace": 4, "travel": 3, "banking": 3, "slack": 4}
    for suite, n in users.items():
        for u in range(n):
            for i in range(2):
                tid = f"{suite}_user_task_{u}_injection_task_{i}"
                for s in rc.sids_for(tid):
                    cond = "clean" if s.startswith("clean_") else "attacked"
                    r = {"sample_id": s, "condition": cond, "task_id": tid, "suite_name": suite,
                         "user_task_id": f"user_task_{u}", "injection_task_id": f"injection_task_{i}",
                         "attack_name": s.split("_" + suite)[0], "asr_success": False, "utility_success": True,
                         "tool_calls": [["f"]], "error": None, "code_version": {"harness_sha256": "h"},
                         "run_settings": {"gpu_name": gpu}}
                    rows.append(r)
                    acts = [{0: torch.ones(4), 1: torch.arange(4.0)}]
                    tr = {"steps": [{"prompt_sha256": "p", "completion": "c"}]}
                    if perturb and s == perturb:
                        r["asr_success"] = True
                        acts = [{0: torch.ones(4) * 1.001, 1: torch.arange(4.0)}]
                        tr = {"steps": [{"prompt_sha256": "p", "completion": "other"}]}
                    torch.save(acts, d / f"{s}.pt")
                    (d / f"{s}.transcript.json.gz").write_bytes(gzip.compress(json.dumps(tr).encode()))
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_select_fixed_and_eligible(tmp_path):
    d = make(tmp_path / "o")
    rows = json.loads((d / "attack_results.json").read_text())
    # make one workspace pair ineligible (errored ICON row) and one excluded (clean succeeds)
    for r in rows:
        if r["sample_id"] == "important_instructions_defended_icon_workspace_user_task_0_injection_task_0":
            r["error"] = "x"
        if r["sample_id"] == "clean_workspace_user_task_0_injection_task_1":
            r["asr_success"] = True
    (d / "attack_results.json").write_text(json.dumps(rows))
    a, b = rc.select(d), rc.select(d)
    assert a == b and len(a) == 50
    assert not any("workspace_user_task_0_" in s for s in a)        # no eligible pair left for user_task_0


def test_compare(tmp_path):
    o = make(tmp_path / "o")
    sids = rc.select(o)
    r = make(tmp_path / "r", perturb=sids[1])
    s = rc.compare(o, r, tmp_path / "out", sids)
    assert s["compared"] == 50 and s["asr_same"] == 49 and s["completions_same"] == 49
    assert s["activations_bitwise"] == 49 and abs(s["activations_max_rel"] - 0.001) < 1e-4
    assert s["prompts_same"] == 50 and s["code_hash_same"] == 50 and s["run_settings_differ"] == []
    r2 = make(tmp_path / "r2", gpu="RTX 8000")
    s2 = rc.compare(o, r2, tmp_path / "out2", sids)
    assert s2["run_settings_differ"] == ["gpu_name"]
