"""Tests for compliance_bootstrap --common (cross-model comparison on identical pairs)."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import compliance_bootstrap as cb  # noqa: E402


def ep(user, atk, exposed, success, cu, inj, cond="attacked", sid="0"):
    return ((("s", user)), atk, exposed, success, cu, ("s", user, inj), cond, sid)


# model A: exposed on u1,u2,u3 (all pairs); model B: not exposed on u3
A = [ep("u1", "ii", True, True, True, "i1"), ep("u1", "ii", True, False, True, "i2"),
     ep("u2", "ii", True, True, False, "i1"), ep("u2", "ii", True, True, False, "i2"),
     ep("u3", "ii", True, True, True, "i1"), ep("u3", "ii", True, True, True, "i2")]
B = [ep("u1", "ii", True, False, True, "i1"), ep("u1", "ii", True, False, True, "i2"),
     ep("u2", "ii", True, True, False, "i1"), ep("u2", "ii", True, False, False, "i2"),
     ep("u3", "ii", False, False, True, "i1"), ep("u3", "ii", False, False, True, "i2")]


def test_restrict_common_keeps_only_pairs_exposed_for_all():
    exp, blind, info = cb.restrict_common({"A": A, "B": B})
    assert info["n_common_exposed"] == {"ii": 4}
    assert info["n_common_scored"] == {"ii": 6}
    assert {e[5] for e in exp["A"]} == {e[5] for e in exp["B"]}
    assert all(e[0][1] != "u3" for e in exp["A"])
    assert len(blind["A"]) == len(blind["B"]) == 6


def test_restrict_common_drops_and_counts_duplicates():
    exp, blind, info = cb.restrict_common({"A": A + [A[0]], "B": B})
    assert info["duplicates_dropped"] == {"A": 1}
    assert len(blind["A"]) == 6


def test_restrict_common_scored_intersection():
    # B lost one pair (e.g. out of memory): blind ASR must drop it for A too
    exp, blind, info = cb.restrict_common({"A": A, "B": B[:-1]})
    assert info["n_common_scored"] == {"ii": 5}
    assert ("s", "u3", "i2") not in {e[5] for e in blind["A"]}


def run_main(monkeypatch, tmp_path, eps_by_name, extra):
    monkeypatch.setattr(cb.ab, "load_dir", lambda d: {"name": d.name, "rows": []})
    monkeypatch.setattr(cb.ab, "injection_defaults", lambda: {})
    monkeypatch.setattr(cb, "episodes", lambda run, d, x: eps_by_name[run["name"]])
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["cb", "A", "B", "--B", "50", "--out", str(out)] + extra)
    cb.main()
    return json.loads(out.read_text())


def test_main_common_bounds_on_identical_pairs(monkeypatch, tmp_path):
    o = run_main(monkeypatch, tmp_path, {"A": A, "B": B}, ["--common"])
    a, b = o["cells"]["A|ii"], o["cells"]["B|ii"]
    assert a["n_exposed"] == b["n_exposed"] == 4
    # A over u1,u2: successes 1+2 of 4; c = mean clean utility over those 4 episodes = 0.5
    assert a["asr_e"] == pytest.approx(0.75) and a["c"] == pytest.approx(0.5)
    assert a["upper"] == pytest.approx(1.0)
    assert b["asr_e"] == pytest.approx(0.25) and b["upper"] == pytest.approx(0.75)
    bl = o["blind_common"]
    assert bl["A|ii"]["asr"] == pytest.approx(5 / 6) and bl["B|ii"]["asr"] == pytest.approx(1 / 6)
    assert bl["A - B|ii"]["diff"] == pytest.approx(4 / 6)


def test_main_default_mode_unchanged(monkeypatch, tmp_path):
    o = run_main(monkeypatch, tmp_path, {"A": A, "B": B}, [])
    assert o["cells"]["A|ii"]["n_exposed"] == 6 and o["cells"]["B|ii"]["n_exposed"] == 4
    assert "blind_common" not in o and o["common"] is False


def test_common_needs_two_models(monkeypatch, tmp_path):
    monkeypatch.setattr(cb.ab, "load_dir", lambda d: {"name": d.name, "rows": []})
    monkeypatch.setattr(cb.ab, "injection_defaults", lambda: {})
    monkeypatch.setattr(sys, "argv", ["cb", "A", "--common", "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        cb.main()


def test_intersection_is_per_attack():
    # pair (u1,i1): A exposed only under a1, B exposed only under a2 -> excluded from both
    a = [ep("u1", "a1", True, True, True, "i1"), ep("u1", "a2", False, False, True, "i1"),
         ep("u2", "a1", True, True, True, "i1"), ep("u2", "a2", True, True, True, "i1")]
    b = [ep("u1", "a1", False, False, True, "i1"), ep("u1", "a2", True, True, True, "i1"),
         ep("u2", "a1", True, False, True, "i1"), ep("u2", "a2", True, False, True, "i1")]
    exp, _, info = cb.restrict_common({"A": a, "B": b})
    assert info["n_common_exposed"] == {"a1": 1, "a2": 1}
    assert ("s", "u1", "i1") not in {e[5] for e in exp["A"]}


def test_dedup_independent_of_row_order():
    unexp = ep("u1", "ii", False, False, True, "i1", cond="defended_attacked", sid="1")
    expo = ep("u1", "ii", True, True, True, "i1", cond="attacked", sid="2")
    other = [ep("u1", "ii", True, True, True, "i1")]
    r1 = cb.restrict_common({"A": [unexp, expo], "B": other})
    r2 = cb.restrict_common({"A": [expo, unexp], "B": other})
    assert r1[2]["n_common_exposed"] == r2[2]["n_common_exposed"] == {"ii": 1}
    assert r1[0]["A"][0][6] == "attacked"


def test_requires_clean_utility_in_intersection():
    a = [ep("u1", "ii", True, True, None, "i1"), ep("u2", "ii", True, True, True, "i1")]
    b = [ep("u1", "ii", True, True, True, "i1"), ep("u2", "ii", True, True, True, "i1")]
    _, _, info = cb.restrict_common({"A": a, "B": b})
    assert info["n_common_exposed"] == {"ii": 1}


def test_empty_attack_reported():
    a = [ep("u1", "ii", True, True, True, "i1"), ep("u1", "t", True, True, True, "i1")]
    b = [ep("u1", "ii", True, True, True, "i1")]
    _, _, info = cb.restrict_common({"A": a, "B": b})
    assert info["empty_attacks"] == ["t"]


def test_duplicate_names_rejected_in_default_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(cb.ab, "load_dir", lambda d: {"name": "same", "rows": []})
    monkeypatch.setattr(cb.ab, "injection_defaults", lambda: {})
    monkeypatch.setattr(sys, "argv", ["cb", "x/A", "y/A", "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        cb.main()
