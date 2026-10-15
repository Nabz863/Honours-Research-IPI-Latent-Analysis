"""Hand-computed checks of analysis/vwa_behaviour.py on a small fixture."""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

import vwa_behaviour as vb  # noqa: E402


def row(cond, attack, task, asr, ua, d=None, reached=None, err=None, strict=False, scorable=True, **kw):
    r = {"sample_id": f"{attack}_{d}_{task}", "condition": cond, "attack_name": attack, "task_id": task,
         "asr_success": asr, "utility_success": ua, "error": err, "strict_asr": strict,
         "ua_note": None if scorable else "clean evaluator targets a different intent -- UA not scorable",
         "payload_reached": reached, "run_settings": {"model_id": "m"}}
    if d:
        r["defense_name"] = d
    r.update(kw)
    return r


def fixture(tmp: Path, name: str, rows):
    d = tmp / name
    d.mkdir()
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_numbers(tmp_path):
    A = [
        # clean: t0 meets the attacker goal with no attack -> excluded everywhere
        row("clean", "clean", "t0", True, True),
        row("clean", "clean", "t1", False, True),
        row("clean", "clean", "t2", False, False),
        row("clean", "clean", "t3", False, None, scorable=False),
        # undefended attack
        row("attacked", "typographic", "t0", True, True, reached=True),
        row("attacked", "typographic", "t1", True, False, reached=True, strict=True),
        row("attacked", "typographic", "t2", False, False, reached=False),
        row("attacked", "typographic", "t3", True, None, reached=None, scorable=False),
        # defended: prevents t1, enables t2; t3 out of memory (unscored)
        row("defended_attacked", "typographic", "t1", False, True, d="icon", reached=True),
        row("defended_attacked", "typographic", "t2", True, False, d="icon", reached=True),
        row("defended_attacked", "typographic", "t3", None, None, d="icon",
            err="CUDA_OOM: CUDA out of memory. Tried to allocate 24.53 GiB", scorable=False),
        row("defended_clean", "clean", "t1", False, None, d="icon",
            err="post-episode: UA scoring failed: Error: Page.goto: net::ERR_ABORTED"),
        row("defended_clean", "clean", "t2", True, True, d="icon"),
        # not run: the transfer image does not exist
        {"sample_id": "x", "attack_name": "clip_attack_transfer", "task_id": "t1",
         "error": "no cached payload from the undefended pass", "defense_name": "icon"},
        # an unexpected error: not a result
        row("attacked", "comet", "t1", True, True, err="post-episode: captioner failed on 1 page image(s)"),
    ]
    B = [row("clean", "clean", "t1", True, True, run_settings={"model_id": "m2"})]
    data = vb.analyse([fixture(tmp_path, "A", A), fixture(tmp_path, "B", B)], tmp_path / "out")
    # union over models: t0 from A, t1 from B
    assert data["excluded_tasks"] == ["t0", "t1"]
    data = vb.analyse([fixture(tmp_path, "A2", A)], tmp_path / "out2")
    assert data["excluded_tasks"] == ["t0"]
    M = data["models"]["m"]
    assert M["row_kinds"] == {"result": 11, "oom": 1, "ua_failed": 1, "not_run": 1, "error": 1}
    assert M["oom_tasks"] == ["t3"]
    c = M["cells"]
    assert c["typographic|none"] == {"asr": [2, 3], "ua": [0, 2]}     # t0 excluded; t3 UA unscorable
    assert c["clean|none"] == {"asr": [0, 3], "ua": [1, 2]}
    assert c["typographic|icon"] == {"asr": [1, 2], "ua": [1, 2]}     # OOM row unscored
    assert c["clean|icon"] == {"asr": [1, 2], "ua": [1, 2]}           # UA-scoring failure = UA failure
    assert "comet|none" not in c                                       # unexpected error: not a result
    u = M["undefended"]["typographic"]
    assert u["asr_published"] == [1, 2] and u["asr_strict"] == [1, 1]
    d = M["defences"]["icon"]
    assert (d["prevented"], d["enabled"]) == (1, 1) and math.isclose(d["mcnemar_p"], 1.0)
    assert d["matched"] == {"n": 2, "undefended": 1, "defended": 1}   # t1, t2; t3 unscored
    e = M["exposure"]["typographic"]
    assert (e["exposed"], e["not_exposed"], e["no_text"], e["n"]) == (1, 1, 1, 3)
    assert (e["succ_exposed"], e["succ_unexposed"]) == (1, 1)
    # bounds on exposed episodes of UA-scorable tasks: t1 only, clean UA 1 -> c=1
    assert e["bounds_n"] == 1 and e["lower"] == 1.0 and e["upper"] == 1.0
    assert M["exposure_defended"]["icon"] == [2, 2]


def test_kind():
    assert vb.kind({"condition": "attacked", "error": None}) == "result"
    assert vb.kind({"error": "no cached payload"}) == "not_run"
    assert vb.kind({"condition": "attacked", "error": "CUDA_OOM: x"}) == "oom"
    # captioner OOM is a post-episode problem, not an episode that never ran
    assert vb.kind({"condition": "attacked", "error": "post-episode: captioner failed on 1 page image(s) "
                    "(first failure; retried): CUDA out of memory"}) == "error"
    # rows the runner completed after repeated scoring failures
    assert vb.kind({"condition": "attacked", "error": None, "scoring_error_final": True,
                    "ua_scoring_error": "Page.goto", "asr_success": True}) == "ua_failed"
    assert vb.kind({"condition": "attacked", "error": None, "scoring_error_final": True,
                    "asr_scoring_error": "x", "asr_success": None}) == "scoring_failed"
    assert vb.kind({"condition": "attacked", "error": "post-episode: UA scoring failed: y"}) == "ua_failed"
    # a scoring failure AFTER another problem is not a clean UA failure
    assert vb.kind({"condition": "attacked",
                    "error": "post-episode: captioner failed; UA scoring failed: y"}) == "error"
    assert vb.kind({"condition": "attacked", "error": "post-episode: ASR scoring failed: z"}) == "error"


def test_mcnemar():
    assert vb.mcnemar_p(0, 0) == 1.0
    assert math.isclose(vb.mcnemar_p(0, 5), 2 / 32)
    assert math.isclose(vb.mcnemar_p(5, 0), 2 / 32)
    assert math.isclose(vb.mcnemar_p(1, 1), 1.0)
