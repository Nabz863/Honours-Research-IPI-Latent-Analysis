"""analysis/rq2_geometry.py on a tiny fixture with known activations:
the four-state load, the along/orth decomposition, the zero-delta
exclusion, the footprint_orth significance test (audit 4.1), and
--attack all discovery across two attacked conditions."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import rq2_geometry as rq2  # noqa: E402

L = 31


def _ep(d: Path, sid, vec):
    """Single-turn episode: one layer-L vector."""
    torch.save([{L: torch.tensor(vec, dtype=torch.float32)}], d / f"{sid}.pt")


def _row(rows, sid, cond, attack, task, defense=None, error=None):
    rows.append({"sample_id": sid, "condition": cond, "attack_name": attack,
                 "task_id": task, "suite_name": "workspace",
                 "defense_name": defense, "asr_success": None, "error": error})


def make(tmp: Path, n_tasks=8):
    """n_tasks tasks, one attack ('template_ipi'), one defence ('datamarking').
    Geometry (2-D, layer L):
        clean c = (0,0)
        attacked a = (10,0)       -> attack axis u = +x
        defended-clean f = (0,1)  -> footprint purely OFF-axis, ||f-c||=1
        defended-attacked d = (2,3) -> along=2 (frac .2), orth=3
    So orth (3) EXCEEDS footprint_orth (1): a suppression signal the
    footprint_orth test must detect, and that the full-norm footprint
    (also 1 here) would equally detect -- the distinction matters when f
    has an on-axis part, covered in make_onaxis()."""
    d = tmp / "final_agentdojo_rq2"
    d.mkdir()
    rows = []
    for i in range(n_tasks):
        tid = f"workspace_user_task_{i}_injection_task_0"
        _row(rows, f"clean_{tid}", "clean", "clean", tid)
        _ep(d, f"clean_{tid}", [0.0, 0.0])
        _row(rows, f"template_ipi_{tid}", "attacked", "template_ipi", tid)
        _ep(d, f"template_ipi_{tid}", [10.0, 0.0])
        _row(rows, f"dc_datamarking_{tid}", "defended_clean", "clean", tid, "datamarking")
        _ep(d, f"dc_datamarking_{tid}", [0.0, 1.0])
        _row(rows, f"da_datamarking_{tid}", "defended_attacked", "template_ipi", tid, "datamarking")
        _ep(d, f"da_datamarking_{tid}", [2.0, 3.0])
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_decompose_known_values():
    c, a = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    f, dd = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    m = rq2.decompose(c, a, dd, f)
    assert m["along"] == pytest.approx(2.0)
    assert m["along_frac"] == pytest.approx(0.2)
    assert m["orth"] == pytest.approx(3.0)
    assert m["footprint"] == pytest.approx(1.0)
    assert m["footprint_orth"] == pytest.approx(1.0)   # f is already off-axis
    assert m["footprint_cos_attack"] == pytest.approx(0.0, abs=1e-9)


def test_zero_delta_excluded(tmp_path):
    d = make(tmp_path, n_tasks=3)
    # add a task whose attacked state equals clean -> no axis, must be excluded
    rows = json.loads((d / "attack_results.json").read_text())
    tid = "workspace_user_task_99_injection_task_0"
    _row(rows, f"clean_{tid}", "clean", "clean", tid)
    _ep(d, f"clean_{tid}", [5.0, 5.0])
    _row(rows, f"template_ipi_{tid}", "attacked", "template_ipi", tid)
    _ep(d, f"template_ipi_{tid}", [5.0, 5.0])        # identical -> ||a-c||=0
    _row(rows, f"da_datamarking_{tid}", "defended_attacked", "template_ipi", tid, "datamarking")
    _ep(d, f"da_datamarking_{tid}", [9.0, 9.0])
    (d / "attack_results.json").write_text(json.dumps(rows))

    states = rq2.load(str(d), L)
    _, n_axis, n_zero, _ = rq2.analyse_attack(states, "template_ipi")
    assert n_zero == 1 and n_axis == 3


def test_footprint_orth_is_the_tested_quantity(tmp_path):
    """The headline of audit 4.1: the Wilcoxon compares orth against
    footprint_orth, not the full footprint norm. Build f with a large
    ON-axis component so the two footprints differ a lot, and confirm the
    reported test statistic is driven by footprint_orth."""
    d = tmp_path / "onaxis"
    d.mkdir()
    rows = []
    for i in range(8):
        tid = f"workspace_user_task_{i}_injection_task_0"
        _row(rows, f"clean_{tid}", "clean", "clean", tid)
        _ep(d, f"clean_{tid}", [0.0, 0.0])
        _row(rows, f"template_ipi_{tid}", "attacked", "template_ipi", tid)
        _ep(d, f"template_ipi_{tid}", [10.0, 0.0])          # axis = +x
        # footprint mostly ON-axis: f=(8,1) -> ||f-c||=8.06 but off-axis part=1
        _row(rows, f"dc_{tid}", "defended_clean", "clean", tid, "enc")
        _ep(d, f"dc_{tid}", [8.0, 1.0])
        # defended-attacked off-axis by 3 -> orth=3
        _row(rows, f"da_{tid}", "defended_attacked", "template_ipi", tid, "enc")
        _ep(d, f"da_{tid}", [2.0, 3.0])
    (d / "attack_results.json").write_text(json.dumps(rows))

    states = rq2.load(str(d), L)
    out, _, _, _ = rq2.analyse_attack(states, "template_ipi")
    e = out[0]
    assert e["defense"] == "enc"
    # orth=3 vs footprint_orth=1 everywhere -> orth strictly greater -> r=+1
    assert e["orth_median"] == pytest.approx(3.0)
    assert e["footprint_median"] == pytest.approx(np.hypot(8.0, 1.0), abs=1e-4)
    assert e["orth_over_footprint_orth_median"] == pytest.approx(3.0, abs=1e-4)
    assert e["rank_biserial"] == pytest.approx(1.0)
    # the full-norm ratio would be ~0.37 (3/8.06) -- a RESTORATION-looking
    # number -- so testing against the full footprint would hide the effect
    assert e["orth_over_footprint_median"] < 0.5


def test_attack_all_discovers_both(tmp_path):
    d = make(tmp_path, n_tasks=6)
    # add a second attacked condition carried into the same defence
    rows = json.loads((d / "attack_results.json").read_text())
    for i in range(6):
        tid = f"workspace_user_task_{i}_injection_task_0"
        _row(rows, f"important_instructions_{tid}", "attacked", "important_instructions", tid)
        _ep(d, f"important_instructions_{tid}", [6.0, 0.0])
        _row(rows, f"da2_{tid}", "defended_attacked", "important_instructions", tid, "datamarking")
        _ep(d, f"da2_{tid}", [1.0, 2.0])
    (d / "attack_results.json").write_text(json.dumps(rows))

    states = rq2.load(str(d), L)
    assert rq2.discover_attacks(states) == ["important_instructions", "template_ipi"]
    # both reuse the SAME defended_clean footprint (keyed by defence alone)
    out_t, _, _, _ = rq2.analyse_attack(states, "template_ipi")
    out_i, _, _, _ = rq2.analyse_attack(states, "important_instructions")
    assert out_t[0]["footprint_median"] == pytest.approx(1.0)
    assert out_i[0]["footprint_median"] == pytest.approx(1.0)
    # important_instructions: axis=+x len6, d-c=(1,2) -> along_frac 1/6, orth 2
    assert out_i[0]["along_frac_median"] == pytest.approx(1.0 / 6.0, abs=1e-4)
    assert out_i[0]["orth_median"] == pytest.approx(2.0)


def test_errored_rows_skipped(tmp_path):
    d = make(tmp_path, n_tasks=4)
    rows = json.loads((d / "attack_results.json").read_text())
    # flip one attacked row to an error with a (still loadable) tensor
    for r in rows:
        if r["condition"] == "attacked":
            r["error"] = "boom"
            break
    (d / "attack_results.json").write_text(json.dumps(rows))
    states = rq2.load(str(d), L)
    _, n_axis, _, _ = rq2.analyse_attack(states, "template_ipi")
    assert n_axis == 3        # the errored attacked row dropped one axis


# ---------------------------------------------------------------------------
# Regression A (audit-review finding 1): a ZERO off-axis footprint is a real
# observation (0.0), not a missing one. A truthiness filter
# (`if x.get("footprint_orth")`) drops it, foot_orth ends up shorter than orth,
# and the length guard silently skips the whole defence's Wilcoxon.
# ---------------------------------------------------------------------------

def test_A_zero_offaxis_footprint_keeps_the_pvalue(tmp_path):
    d = tmp_path / "zero_foot"
    d.mkdir()
    rows = []
    # f per task: six ordinary off-axis footprints, one purely ON-axis
    # (off-axis part exactly 0.0), one zero footprint (f == c -> 0.0 branch)
    fs = [[0.0, 1.0]] * 6 + [[5.0, 0.0], [0.0, 0.0]]
    for i, f in enumerate(fs):
        tid = f"workspace_user_task_{i}_injection_task_0"
        _row(rows, f"clean_{tid}", "clean", "clean", tid)
        _ep(d, f"clean_{tid}", [0.0, 0.0])
        _row(rows, f"atk_{tid}", "attacked", "template_ipi", tid)
        _ep(d, f"atk_{tid}", [10.0, 0.0])
        _row(rows, f"dc_{tid}", "defended_clean", "clean", tid, "enc")
        _ep(d, f"dc_{tid}", f)
        _row(rows, f"da_{tid}", "defended_attacked", "template_ipi", tid, "enc")
        _ep(d, f"da_{tid}", [2.0, 3.0])
    (d / "attack_results.json").write_text(json.dumps(rows))

    states = rq2.load(str(d), L)
    out, _, _, raw = rq2.analyse_attack(states, "template_ipi")
    fo = sorted(x["footprint_orth"] for x in raw["enc"])
    assert fo[:2] == [0.0, 0.0]                      # both zero cases present
    e = out[0]
    assert e["n"] == 8
    assert "p_raw" in e, "zero off-axis footprint dropped -> defence lost its p-value"
    sps = pytest.importorskip("scipy.stats")
    _, p = sps.wilcoxon([3.0] * 8, [1.0] * 6 + [0.0, 0.0],
                        zero_method="wilcox", alternative="two-sided")
    assert e["p_raw"] == pytest.approx(p)
    assert e["rank_biserial"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Regression F (finding 8): in AgentDojo each user task is repeated once per
# injection task, and those repeats are not independent (same user prompt, same
# clean and defended-clean episode). The Wilcoxon must run on ONE observation
# per user task -- grouped by geometry.user_task_group, the key stats.py's
# probe folds use -- not on every (user_task, injection_task) pair.
# ---------------------------------------------------------------------------

def _grouped_dir(tmp, spec):
    """spec: list of (task_id, orth_y[, attack_tilt]). Geometry as make():
    c=(0,0), a=(10,tilt) (tilt 0 -> axis +x), f=(0,1) -> footprint_orth=1 when
    untilted; d=(2, y) -> orth=|y| when untilted. A tilt changes the attack
    axis per repeat, so footprint_orth varies within a user task as it does in
    real AgentDojo data."""
    d = tmp / "grouped"
    d.mkdir()
    rows = []
    for item in spec:
        tid, y = item[0], item[1]
        tilt = item[2] if len(item) > 2 else 0.0
        _row(rows, f"clean_{tid}", "clean", "clean", tid)
        _ep(d, f"clean_{tid}", [0.0, 0.0])
        _row(rows, f"atk_{tid}", "attacked", "template_ipi", tid)
        _ep(d, f"atk_{tid}", [10.0, tilt])
        _row(rows, f"dc_{tid}", "defended_clean", "clean", tid, "enc")
        _ep(d, f"dc_{tid}", [0.0, 1.0])
        _row(rows, f"da_{tid}", "defended_attacked", "template_ipi", tid, "enc")
        _ep(d, f"da_{tid}", [2.0, y])
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_F_wilcoxon_runs_on_one_observation_per_user_task(tmp_path):
    # 6 user tasks x 3 injection tasks = 18 pairs. Five user tasks sit above
    # their footprint (orth 3 vs 1), one sits below (0.5 vs 1). Per user task
    # that is 5 up / 1 down -> exact two-sided p = 0.0625, NOT significant.
    # Treating the 18 repeats as independent gives 15 up / 3 down -> p << 0.05.
    spec = [(f"workspace_user_task_{u}_injection_task_{j}", 3.0 if u < 5 else 0.5)
            for u in range(6) for j in range(3)]
    states = rq2.load(str(_grouped_dir(tmp_path, spec)), L)
    e = rq2.analyse_attack(states, "template_ipi")[0][0]
    sps = pytest.importorskip("scipy.stats")
    _, p_grouped = sps.wilcoxon([3.0] * 5 + [0.5], [1.0] * 6,
                                zero_method="wilcox", alternative="two-sided")
    _, p_pairs = sps.wilcoxon([3.0] * 15 + [0.5] * 3, [1.0] * 18,
                              zero_method="wilcox", alternative="two-sided")
    assert p_pairs < 0.05 < p_grouped            # the fixture separates the two
    assert e["n"] == 18 and e["n_groups"] == 6
    assert e["p_raw"] == pytest.approx(p_grouped)
    # matched-pairs rank-biserial on the 6 group diffs (+2 x5, -0.5 x1):
    # |-0.5| has rank 1, the +2s share ranks 2..6 -> (20 - 1) / 21
    assert e["rank_biserial"] == pytest.approx(19 / 21)


def test_F_too_few_user_tasks_gets_no_pvalue(tmp_path):
    # 3 user tasks x 4 injection tasks = 12 pairs >= 6, but only 3 independent
    # observations: below the test's minimum, so no p-value at all.
    spec = [(f"workspace_user_task_{u}_injection_task_{j}", 3.0)
            for u in range(3) for j in range(4)]
    states = rq2.load(str(_grouped_dir(tmp_path, spec)), L)
    e = rq2.analyse_attack(states, "template_ipi")[0][0]
    assert e["n"] == 12 and e["n_groups"] == 3
    assert "p_raw" not in e and "p_holm" not in e


def test_F_groups_are_suite_scoped(tmp_path):
    # user_task_0 exists in every suite; they are different tasks (the P2a.1
    # dedup bug). 3 user-task numbers x 2 suites = 6 groups, not 3.
    spec = [(f"{s}_user_task_{u}_injection_task_{j}", 3.0)
            for s in ("workspace", "slack") for u in range(3) for j in range(2)]
    states = rq2.load(str(_grouped_dir(tmp_path, spec)), L)
    e = rq2.analyse_attack(states, "template_ipi")[0][0]
    assert e["n_groups"] == 6 and "p_raw" in e


def test_F_collapse_is_the_mean_of_varying_repeats(tmp_path):
    """Repeats differ within a user task, so the group summary is pinned to the
    MEAN: first-repeat, max, or median summaries, or an effect size computed on
    the per-pair diffs, all give a different p or r on this fixture."""
    # chosen (by search) so no |diff| ties, and first-repeat / max / median
    # summaries each give a different p AND r from the mean
    ys = {0: [0.3, 0.6, 1.7], 1: [2.7, 3.9, 3.8], 2: [3.7, 3.2, 2.3],
          3: [0.5, 3.4, 0.3], 4: [2.6, 3.0, 2.2], 5: [2.2, 3.1, 1.0],
          6: [3.5, 2.9, 2.7]}
    spec = [(f"workspace_user_task_{u}_injection_task_{j}", y)
            for u, v in ys.items() for j, y in enumerate(v)]
    states = rq2.load(str(_grouped_dir(tmp_path, spec)), L)
    rows, _, _, raw = rq2.analyse_attack(states, "template_ipi")
    e = rows[0]
    sps = pytest.importorskip("scipy.stats")
    # recompute the per-user-task means independently from the decomposed pairs
    by = {}
    for x in raw["enc"]:
        by.setdefault(x["task_id"].rsplit("_injection_task_", 1)[0], []).append(x)
    go = np.array([np.mean([x["orth"] for x in v]) for v in by.values()])
    gf = np.array([np.mean([x["footprint_orth"] for x in v]) for v in by.values()])
    _, p = sps.wilcoxon(go, gf, zero_method="wilcox", alternative="two-sided")
    assert e["n_groups"] == 7
    assert e["p_raw"] == pytest.approx(p)
    assert e["rank_biserial"] == pytest.approx(rq2.matched_rank_biserial(go - gf))
    means = np.array([np.mean(v) for v in ys.values()])
    # the alternatives really are distinguishable here
    for alt in (np.array([v[0] for v in ys.values()]), np.array([max(v) for v in ys.values()]),
                np.array([np.median(v) for v in ys.values()])):
        _, pa = sps.wilcoxon(alt, np.ones(7), zero_method="wilcox", alternative="two-sided")
        ra = rq2.matched_rank_biserial(alt - 1.0)
        assert not (pa == pytest.approx(p, rel=1e-3) and
                    ra == pytest.approx(e["rank_biserial"], rel=1e-3))
    per_pair = np.array([y for v in ys.values() for y in v]) - 1.0
    assert rq2.matched_rank_biserial(per_pair) != pytest.approx(e["rank_biserial"], rel=1e-3)


# --- review finding 5: the effect size is the matched-pairs rank-biserial ----

def test_matched_rank_biserial_uses_ranks_not_signs():
    # six tiny positive diffs, four large negative ones: the sign statistic
    # says +0.2 (wrong direction); the rank-biserial says (21 - 34)/55
    d = [0.1] * 6 + [-10.0] * 4
    assert rq2.matched_rank_biserial(d) == pytest.approx(-13 / 55)
    assert rq2.matched_rank_biserial([1, 2, 3]) == pytest.approx(1.0)
    assert rq2.matched_rank_biserial([0, 0]) == 0.0
    # ties across signs need AVERAGE ranks: |1|,|-1| share 1.5 -> (4.5-1.5)/6
    assert rq2.matched_rank_biserial([1, -1, 2]) == pytest.approx(0.5)
    # zeros are dropped (not ranked, not counted): ranks 1,2 -> (1-2)/3
    assert rq2.matched_rank_biserial([0, 1, -2]) == pytest.approx(-1 / 3)
    # a non-finite difference gives no effect size rather than a silent number
    assert rq2.matched_rank_biserial([np.nan, 1, -2]) is None


def test_avg_ranks_matches_scipy_rankdata():
    sps = pytest.importorskip("scipy.stats")
    from geometry import _avg_ranks
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.integers(0, 6, size=rng.integers(1, 15)).astype(float)
        assert np.allclose(_avg_ranks(a), sps.rankdata(a))


# --- review finding 3 / cleanup-plan A2: VWA variants group by base task -----

def test_user_task_group_vwa_base_task_and_agentdojo_suite():
    from geometry import user_task_group as utg
    assert utg("reddit_36_wrong_object_cap_1") == "reddit_36"
    assert utg("reddit_36_wrong_object_cap_2") == "reddit_36"
    assert utg("reddit_142_wrong_text_cap") == "reddit_142"
    assert utg("workspace_user_task_3_injection_task_5") == "workspace_user_task_3"
    assert utg("slack_user_task_3_injection_task_5") == "slack_user_task_3"
    assert utg("weird") == "weird"
    assert utg("shopping_admin_12_wrong_text_cap") == "shopping_admin_12"   # multi-word site


def test_F_vwa_variants_collapse(tmp_path):
    # 6 base tasks x 2 variants: 12 pairs, 6 independent observations
    spec = [(f"reddit_{10 + u}_wrong_object_cap_{k}", 3.0) for u in range(6) for k in (1, 2)]
    states = rq2.load(str(_grouped_dir(tmp_path, spec)), L)
    e = rq2.analyse_attack(states, "template_ipi")[0][0]
    assert e["n"] == 12 and e["n_groups"] == 6


# --- review finding 4: a defence with no footprint is not labelled PARTIAL ---

def test_zero_footprint_defence_is_undefined_not_partial(tmp_path):
    d = tmp_path / "nofoot"
    d.mkdir()
    rows = []
    for i in range(8):
        tid = f"workspace_user_task_{i}_injection_task_0"
        _row(rows, f"clean_{tid}", "clean", "clean", tid)
        _ep(d, f"clean_{tid}", [0.0, 0.0])
        _row(rows, f"atk_{tid}", "attacked", "template_ipi", tid)
        _ep(d, f"atk_{tid}", [10.0, 0.0])
        _row(rows, f"dc_{tid}", "defended_clean", "clean", tid, "passthru")
        _ep(d, f"dc_{tid}", [0.0, 0.0])                      # f == c: no footprint
        _row(rows, f"da_{tid}", "defended_attacked", "template_ipi", tid, "passthru")
        _ep(d, f"da_{tid}", [0.5, 0.2])
    (d / "attack_results.json").write_text(json.dumps(rows))
    e = rq2.analyse_attack(rq2.load(str(d), L), "template_ipi")[0][0]
    assert e["orth_over_footprint_median"] is None and e["n_orth_over_footprint"] == 0
    assert e["footprint_cos_attack_median"] is None
    assert e["orth_over_footprint_orth_median"] is None
    assert rq2.verdict(e).startswith("UNDEFINED")
    rq2.print_attack("template_ipi", [e], 8, 8, 0, "x", L)     # must not crash on None


# --- review finding 12: rq2 writes its JSON before the console report --------

def test_rq2_json_written_before_report(tmp_path, monkeypatch):
    d = make(tmp_path, n_tasks=8)
    out = tmp_path / "o" / "rq2.json"
    monkeypatch.setattr(sys, "argv", ["rq2_geometry.py", "--results_dir", str(d),
                                      "--layer", str(L), "--out", str(out)])

    def boom(*a, **k):                     # the FIRST console line already fails
        raise RuntimeError("report failed")
    monkeypatch.setattr(rq2, "print", boom, raising=False)
    with pytest.raises(RuntimeError):
        rq2.main()
    res = json.loads(out.read_text())
    assert res["per_attack"]["template_ipi"]["per_defense"][0]["verdict"]


def test_F_footprint_side_is_also_the_mean(tmp_path):
    """The attack axis changes per injection task, so footprint_orth varies
    within a user task; the footprint side must be averaged the same way as
    the orth side (a median there changes p or r on this fixture)."""
    # per-repeat attack tilts and defended offsets chosen (by search) so the
    # footprint side's mean and median give different p AND r
    ys = {0: [2.1, 1.3, 0.5], 1: [0.9, 0.6, 1.7], 2: [2.6, 2.3, 2.0], 3: [0.3, 1.9, 2.4],
          4: [1.1, 1.5, 1.4], 5: [1.6, 0.5, 1.6], 6: [1.0, 1.7, 0.4]}
    tilts = {0: [8, 3, 19], 1: [23, 18, 10], 2: [11, 15, 5], 3: [1, 21, 12],
             4: [18, 11, 7], 5: [1, 15, 13], 6: [0, 4, 23]}
    spec = [(f"workspace_user_task_{u}_injection_task_{j}", y, float(tilts[u][j]))
            for u, v in ys.items() for j, y in enumerate(v)]
    rows, _, _, raw = rq2.analyse_attack(rq2.load(str(_grouped_dir(tmp_path, spec)), L),
                                         "template_ipi")
    e = rows[0]
    sps = pytest.importorskip("scipy.stats")
    by = {}
    for x in raw["enc"]:
        by.setdefault(x["task_id"].rsplit("_injection_task_", 1)[0], []).append(x)
    fo = [[x["footprint_orth"] for x in v] for v in by.values()]
    assert all(np.ptp(f) > 0.3 for f in fo)                # really varies within a task
    go = np.array([np.mean([x["orth"] for x in v]) for v in by.values()])
    gf_mean = np.array([np.mean(f) for f in fo])
    gf_med = np.array([np.median(f) for f in fo])
    _, p = sps.wilcoxon(go, gf_mean, zero_method="wilcox", alternative="two-sided")
    _, p_med = sps.wilcoxon(go, gf_med, zero_method="wilcox", alternative="two-sided")
    r, r_med = rq2.matched_rank_biserial(go - gf_mean), rq2.matched_rank_biserial(go - gf_med)
    assert not (p == pytest.approx(p_med, rel=1e-3) and r == pytest.approx(r_med, rel=1e-3))
    assert e["p_raw"] == pytest.approx(p) and e["rank_biserial"] == pytest.approx(r)


# --- geometry.py's per-outcome RQ1 split follows the same scoring rule -------

def test_geometry_by_outcome_scoring_error_final_is_unscored():
    import geometry as geo
    v = {L: np.ones(2, dtype=np.float32)}
    paired = {("d", "workspace", "t0"): {
        "clean": {"acts": v, "row": {}},
        "tipi": {"acts": {L: np.full(2, 3.0, dtype=np.float32)},
                 "row": {"asr_success": False, "scoring_error_final": True}}}}
    out = geo.per_condition_geometry_by_outcome(paired)
    assert set(out["tipi"]) == {"unscored"}
