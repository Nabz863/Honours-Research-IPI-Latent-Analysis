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
                                      "--layer", str(L), "--turn", "mean", "--out", str(out)])

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


# ===========================================================================
# RQ2 extensions: behaviour link, probe-based restoration, layer sweep
# ===========================================================================

def test_load_outcomes_scoring_rules(tmp_path):
    d = tmp_path / "o"
    d.mkdir()
    rows = [
        {"sample_id": "a1", "condition": "attacked", "attack_name": "A", "task_id": "t1",
         "defense_name": None, "asr_success": True, "error": None},
        {"sample_id": "a2", "condition": "attacked", "attack_name": "A", "task_id": "t2",
         "defense_name": None, "asr_success": False, "error": None, "scoring_error_final": True},
        {"sample_id": "a3", "condition": "attacked", "attack_name": "A", "task_id": "t3",
         "defense_name": None, "asr_success": True, "error": "boom"},
        {"sample_id": "d1", "condition": "defended_attacked", "attack_name": "A", "task_id": "t1",
         "defense_name": "dm", "asr_success": False, "error": None},
        {"sample_id": "d2", "condition": "defended_attacked", "attack_name": "A", "task_id": "t2",
         "defense_name": "dm", "asr_success": "yes", "error": None},
    ]
    (d / "attack_results.json").write_text(json.dumps(rows))
    o = rq2.load_outcomes(str(d))
    assert o["t1"] == {("attacked", "A"): True, ("def_att", "A", "dm"): False}
    assert o["t2"] == {("attacked", "A"): None, ("def_att", "A", "dm"): None}
    assert "t3" not in o


def _xs(spec):
    """spec: list of (user_task, inj, along_frac, orth, footprint_orth)"""
    return [{"task_id": f"workspace_user_task_{u}_injection_task_{j}",
             "user_task_group": f"workspace_user_task_{u}", "along_frac": af,
             "along_frac_def": af, "attack_disp": 10.0,
             "orth": o, "footprint_orth": fo} for u, j, af, o, fo in spec]


def _probe_states(n_users=12, dim=20, seed=0, foot_along=0.0):
    rng = np.random.default_rng(seed)
    e1, e2 = np.eye(dim)[0], np.eye(dim)[1]
    st = {}
    for u in range(n_users):
        for j in range(2):
            t = f"workspace_user_task_{u}_injection_task_{j}"
            c = rng.normal(size=dim)
            a = c + 4.0 * e1 + 0.3 * rng.normal(size=dim)
            f = c + 0.5 * e2 + foot_along * e1
            st[t] = {"clean": c, ("attacked", "A"): a,
                     ("def_clean", "restore"): f, ("def_att", "A", "restore"): f + 0.05 * rng.normal(size=dim),
                     ("def_clean", "persist"): f, ("def_att", "A", "persist"): a + 0.5 * e2}
    return st


def test_probe_restoration_scores_only_held_out_tasks(monkeypatch):
    pytest.importorskip("sklearn")
    import sklearn.pipeline as skp
    st = _probe_states()
    trained, scored = [], []
    real = skp.make_pipeline
    def spy(*steps, **kw):
        pipe = real(*steps, **kw)
        fit, dec = pipe.fit, pipe.decision_function
        def fit_(X, y):
            trained.append({tuple(np.round(x, 6)) for x in X})
            return fit(X, y)
        def dec_(X):
            scored.append((len(trained) - 1, tuple(np.round(X[0], 6))))
            return dec(X)
        pipe.fit, pipe.decision_function = fit_, dec_
        return pipe
    monkeypatch.setattr(skp, "make_pipeline", spy)
    rq2.probe_restoration(st, "A")
    assert len(trained) >= 4
    owner = {tuple(np.round(v, 6)): t.rsplit("_injection_task_", 1)[0]
             for t, sv in st.items() for v in sv.values()}
    assert len(scored) > 100                           # both probes really were spied on
    for k, v in scored:
        g = owner[v]                                   # the scored vector's user task ...
        assert all(owner[w] != g for w in trained[k])  # ... contributed nothing to training


def test_probe_restoration_too_few_user_tasks():
    r = rq2.probe_restoration(_probe_states(n_users=3), "A")
    assert "note" in r and "per_defense" not in r


def _ep_layers(d, sid, by_layer):
    torch.save([{L: torch.tensor(v, dtype=torch.float32) for L, v in by_layer.items()}],
               d / f"{sid}.pt")


def test_layer_sweep_reads_each_file_once_and_keeps_layers_apart(tmp_path, monkeypatch):
    d = tmp_path / "ls"
    d.mkdir()
    rows = []
    for i in range(8):
        tid = f"workspace_user_task_{i}_injection_task_0"
        # layer 30: defence restores fully (d = c); layer 31: d = a (along_frac 1)
        for sid, cond, atk, dn, v30, v31 in [
                (f"c_{tid}", "clean", "clean", None, [0, 0], [0, 0]),
                (f"a_{tid}", "attacked", "A", None, [10, 0], [10, 0]),
                (f"f_{tid}", "defended_clean", "clean", "dm", [0, 1], [0, 1]),
                (f"d_{tid}", "defended_attacked", "A", "dm", [0, 0.5], [10, 0.5])]:
            _row(rows, sid, cond, atk, tid, dn)
            _ep_layers(d, sid, {30: np.array(v30, float), 31: np.array(v31, float)})
    # P2a.1 dedup: a second injection task's defended-clean rows are pointers
    # to the representative's file -- the cache must not re-read it
    for i in range(8):
        tid2 = f"workspace_user_task_{i}_injection_task_1"
        tid = f"workspace_user_task_{i}_injection_task_0"
        rows.append({"sample_id": f"f_{tid2}", "condition": "defended_clean", "attack_name": "clean",
                     "task_id": tid2, "suite_name": "workspace", "defense_name": "dm",
                     "asr_success": None, "error": None, "deduplicated_to": f"f_{tid}"})
    (d / "attack_results.json").write_text(json.dumps(rows))
    n_loads = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (n_loads.append(1), real(*a, **k))[1])
    by_layer = rq2.load_multi(str(d), [30, 31])
    assert len(n_loads) == 32                          # 32 distinct files, each read once
    assert np.allclose(by_layer[31]["workspace_user_task_3_injection_task_1"][("def_clean", "dm")], [0, 1])
    sw = rq2.layer_sweep(by_layer, "A")["dm"]
    assert [x["layer"] for x in sw] == [30, 31]
    assert sw[0]["along_frac_median"] == pytest.approx(0.0)
    assert sw[1]["along_frac_median"] == pytest.approx(1.0)


def test_parse_layers():
    assert rq2._parse_layers("24-26,31, 24") == [24, 25, 26, 31]


def test_main_writes_rq2_extensions(tmp_path, monkeypatch):
    d = make(tmp_path, n_tasks=8)
    rows = json.loads((d / "attack_results.json").read_text())
    for r in rows:                                     # undefended succeeded; defended prevented
        if r["condition"] == "attacked":
            r["asr_success"] = True
        if r["condition"] == "defended_attacked":
            r["asr_success"] = False
    (d / "attack_results.json").write_text(json.dumps(rows))
    out = tmp_path / "o" / "rq2.json"
    monkeypatch.setattr(sys, "argv", ["rq2_geometry.py", "--results_dir", str(d), "--layer", str(L),
                                      "--sweep_layers", str(L), "--turn", "mean", "--out", str(out)])
    rq2.main()
    res = json.loads(out.read_text())["per_attack"]["template_ipi"]
    assert res["per_defense"][0]["by_outcome"]["prevented"]["n"] == 8
    assert "probe_restoration" in res
    assert res["layer_sweep"]["datamarking"][0]["layer"] == L



def test_probe_restoration_needs_six_user_tasks_for_a_test():
    pytest.importorskip("sklearn")
    r = rq2.probe_restoration(_probe_states(n_users=5), "A")
    assert r["n_groups"] == 5
    assert all("p_raw" not in e for e in r["per_defense"])


# ===========================================================================
# Payload-arrival turn (shared with geometry.py's RQ1 headline)
# ===========================================================================
import gzip, hashlib  # noqa: E402

SYS = {"role": "system", "content": [{"type": "text", "text": "sys"}]}


def _tool(text):
    return {"role": "user", "content": [{"type": "text", "text": "[TOOL RESULT] " + text}]}


def _write_tr(d, sid, msgs, comps):
    steps = []
    for k in range(len(comps)):
        conv = msgs[: k + 1]
        steps.append({"input": list(range(k + 1)),
                      "prompt_sha256": hashlib.sha256(json.dumps(conv).encode()).hexdigest(),
                      "completion": comps[k], "tool_calls": [comps[k]]})
    (d / f"{sid}.transcript.json.gz").write_bytes(
        gzip.compress(json.dumps({"messages": msgs, "steps": steps}).encode()))


def _turns(d, sid, vecs):
    torch.save([{L: torch.tensor(v, dtype=torch.float32)} for v in vecs], d / f"{sid}.pt")


def _arrival_dir(tmp):
    """8 normal tasks (payload arrives at step 1 in both pairs), plus one
    unexposed, one non-arrival divergence, one defended pair arriving later.
    Turn-1 geometry: c=(0,0), a=(10,0), f=(0,1), d=(2,3)  -> along_frac 0.2.
    Turns 0 and 2 carry junk, so a mean over turns gives different numbers."""
    d = tmp / "arr"
    d.mkdir()
    rows = []
    junk = [50.0, -50.0]
    def episode(sid, cond, atk, tid, dn, msgs, comps, t1):
        _row(rows, sid, cond, atk, tid, dn)
        _write_tr(d, sid, msgs, comps)
        _turns(d, sid, [junk, t1, junk])
    for i in range(11):
        tid = f"workspace_user_task_{i}_injection_task_0"
        clean_m = [SYS, _tool("clean"), _tool("next")]
        att_m = [SYS, _tool("clean + PAYLOAD"), _tool("next")]
        comps = ["call_a", "call_b", "done"]
        att_comps = comps
        if i == 8:                         # never exposed: identical prompts
            att_m = clean_m
        if i == 9:                         # diverges, but the model's own output differed first
            att_comps = ["call_X", "call_b", "done"]
        episode(f"c_{tid}", "clean", "clean", tid, None, clean_m, comps, [0.0, 0.0])
        episode(f"a_{tid}", "attacked", "A", tid, None, att_m, att_comps,
                [0.0, 0.0] if i == 8 else [10.0, 0.0])          # unexposed: identical states
        dcm = [SYS, _tool("<<clean>>"), _tool("next")]
        dam = [SYS, _tool("<<clean + PAYLOAD>>"), _tool("next")]
        if i == 10:                        # defended pair: payload arrives at step 2 instead
            dam = [SYS, _tool("<<clean>>"), _tool("next PAYLOAD")]
        episode(f"f_{tid}", "defended_clean", "clean", tid, "dm", dcm, comps, [0.0, 1.0])
        episode(f"d_{tid}", "defended_attacked", "A", tid, "dm", dam, comps, [2.0, 3.0])
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_arrival_mode_measures_the_arrival_step(tmp_path):
    d = _arrival_dir(tmp_path)
    info = {}
    st = rq2.load_multi(str(d), [L], turn="arrival", info=info)[L]
    rows, n_axis, _, raw = rq2.analyse_attack(st, "A")
    assert n_axis == 9                                     # 8 normal + the late-defended task
    e = rows[0]
    assert e["n"] == 8                                     # late-defended task dropped from RQ2
    assert e["along_frac_median"] == pytest.approx(0.2)    # turn-1 geometry, not the turn mean
    assert e["orth_median"] == pytest.approx(3.0)
    assert info["A"]["ok"] == 9 and info["A"]["not_exposed"] == 1
    assert info["A"]["not_payload_arrival"] == 1
    assert info["A"]["defended"]["dm"] == {"ok": 8, "defended_arrival_differs": 1,
                                          "undefended_pair_unusable": 2}


def test_mean_mode_differs_on_the_same_data(tmp_path):
    d = _arrival_dir(tmp_path)
    st = rq2.load_multi(str(d), [L], turn="mean")[L]
    e = rq2.analyse_attack(st, "A")[0][0]
    assert e["along_frac_median"] != pytest.approx(0.2)


def test_arrival_step0_only_with_flag(tmp_path):
    d = tmp_path / "vwa"
    d.mkdir()
    _write_tr(d, "c", [_tool("page")], ["act"])
    _write_tr(d, "a", [_tool("page + PAYLOAD")], ["act"])
    import geometry as geo
    assert geo.arrival_step(str(d), "c", "a") == (None, "not_payload_arrival")
    assert geo.arrival_step(str(d), "c", "a", accept_step0=True) == (0, "ok")
    assert geo.arrival_step(str(d), "c", "missing") == (None, "no_transcript")


def test_load_turns_reads_each_file_once(tmp_path, monkeypatch):
    import geometry as geo
    d = tmp_path / "lt"
    d.mkdir()
    _turns(d, "x", [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    n = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (n.append(1), real(*a, **k))[1])
    v = geo.load_turns(str(d), {"x": {0, 2, 7}, "missing": {0}}, [L])
    assert len(n) == 2                                     # x once (+ the missing attempt)
    assert np.allclose(v[("x", 2)][L], [3.0, 3.0]) and ("x", 7) not in v and ("missing", 0) not in v



# --- behaviour link (RQ2 review round: defended-clean reference, paired primary)

def _oc(per, rule):
    return {x["task_id"]: dict(zip([("attacked", "A"), ("def_att", "A", "dm")], rule(x))) for x in per}


def test_outcome_split_paired_primary_within_user_tasks():
    sps = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    spec = []
    for u in range(9):                     # every user task has both prevented and persisted pairs
        for j in range(4):
            base = 0.2 if j < 2 else 0.7
            spec.append((u, j, base + rng.uniform(-0.15, 0.15) + (0.3 if (u == 3 and j == 0) else 0), 1.0, 1.0))
    per = _xs(spec)
    oc = _oc(per, lambda x: (True, int(x["task_id"][-1]) >= 2))   # j 0,1 prevented; 2,3 persisted
    r = rq2.outcome_split(per, oc, "A", "dm")
    assert r["n_tasks_in_both"] == 9
    gm = lambda js: [np.mean([x["along_frac_def"] for x in per
                              if x["user_task_group"].endswith(f"_{u}") and int(x["task_id"][-1]) in js])
                     for u in range(9)]
    diff = np.array(gm((0, 1))) - np.array(gm((2, 3)))
    assert r["prevented_vs_persisted_p"] == pytest.approx(sps.wilcoxon(diff, zero_method="wilcox")[1])
    assert r["prevented_vs_persisted_r"] == pytest.approx(rq2.matched_rank_biserial(diff))
    assert "mw_p_exploratory" in r and r["mw_auc"] == pytest.approx(1.0)


def test_outcome_split_floors():
    per = _xs([(u, j, 0.1 if j == 0 else 0.9, 1.0, 1.0) for u in range(7) for j in range(2)])
    r = rq2.outcome_split(per, _oc(per, lambda x: (True, x["task_id"].endswith("_1"))), "A", "dm")
    assert r["n_tasks_in_both"] == 7 and "prevented_vs_persisted_p" not in r      # < 8 paired
    per = _xs([(u, 0, 0.1, 1.0, 1.0) for u in range(3)] + [(u, 0, 0.9, 1.0, 1.0) for u in range(3, 6)])
    r = rq2.outcome_split(per, _oc(per, lambda x: (True, int(x["user_task_group"][-1]) >= 3)), "A", "dm")
    assert "mw_p_exploratory" not in r                                            # 3 vs 3
    per = _xs([(u, 0, 0.1, 1.0, 1.0) for u in range(4)] + [(u, 0, 0.9, 1.0, 1.0) for u in range(4, 8)])
    r = rq2.outcome_split(per, _oc(per, lambda x: (True, int(x["user_task_group"][-1]) >= 4)), "A", "dm")
    assert "mw_p_exploratory" in r                                                # 4 vs 4


def test_outcome_split_unscored_either_side():
    per = _xs([(0, 0, 0.1, 1.0, 1.0), (1, 0, 0.1, 1.0, 1.0), (2, 0, 0.1, 1.0, 1.0)])
    oc = {per[0]["task_id"]: {("attacked", "A"): True, ("def_att", "A", "dm"): None},
          per[1]["task_id"]: {("attacked", "A"): None, ("def_att", "A", "dm"): False},
          per[2]["task_id"]: {("attacked", "A"): True, ("def_att", "A", "dm"): False}}
    r = rq2.outcome_split(per, oc, "A", "dm")
    assert r["n_unscored"] == 2 and r["prevented"]["n"] == 1


def test_outcome_split_is_not_confounded_by_attack_strength():
    """Identical restoration (half the attack kept) under a defence whose
    footprint has an on-axis part. Weak attacks prevented, strong ones
    persisted: along_frac (from clean) differs by class purely through
    <f-c,u>/||a-c||; along_frac_def does not."""
    per = []
    for u in range(10):
        for j, (size, cls) in enumerate([(2.0, "p"), (2.0, "p"), (20.0, "s"), (20.0, "s")]):
            c, a = np.zeros(2), np.array([size, 0.0])
            f = np.array([1.0, 1.0])
            m = rq2.decompose(c, a, f + 0.5 * (a - c), f)
            m.update(task_id=f"workspace_user_task_{u}_injection_task_{j}",
                     user_task_group=f"workspace_user_task_{u}")
            per.append(m)
    oc = _oc(per, lambda x: (True, int(x["task_id"][-1]) >= 2))
    r = rq2.outcome_split(per, oc, "A", "dm")
    assert r["prevented"]["along_frac_median"] != pytest.approx(r["persisted"]["along_frac_median"])
    assert r["prevented"]["along_frac_def_median"] == pytest.approx(0.5)
    assert r["persisted"]["along_frac_def_median"] == pytest.approx(0.5)
    assert r["prevented"]["attack_disp_median"] == pytest.approx(2.0)
    assert "prevented_vs_persisted_p" not in r                       # every paired diff is 0


# --- signature removal (probe on logits) -------------------------------------

def test_probe_signature_removed_vs_intact():
    pytest.importorskip("sklearn")
    r = rq2.probe_restoration(_probe_states(), "A")
    assert r["probe_heldout_accuracy"] > 0.9 and r["n_groups"] == 12
    by = {e["defense"]: e for e in r["per_defense"]}
    rs, ps = by["restore"], by["persist"]
    assert rs["signature_removed_fraction_median"] > 0.8 and ps["signature_removed_fraction_median"] < 0.2
    lo, hi = rs["signature_removed_fraction_ci95"]
    assert lo <= rs["signature_removed_fraction_median"] <= hi
    assert ps["p_holm"] < 0.05 and ps["rank_biserial_defatt_vs_defclean"] > 0.9
    assert ps["indist_paired_auc"] > 0.9 and "indist_p" not in ps        # descriptive only
    assert ps["indist_n_groups"] == 12                  # one paired difference per USER task
    assert rs["indist_paired_auc"] < 0.9


def test_probe_reference_is_defended_clean_in_logits():
    pytest.importorskip("sklearn")
    r = rq2.probe_restoration(_probe_states(foot_along=2.0), "A")
    e = {x["defense"]: x for x in r["per_defense"]}["restore"]
    assert e["z_def_clean_median"] > e["z_clean_median"] + 1.0        # the footprint shows
    assert e["signature_removed_fraction_median"] == pytest.approx(1.0, abs=0.1)


def test_probe_counts_tasks_it_cannot_score():
    pytest.importorskip("sklearn")
    # footprint alone makes clean read as attacked: z_att - z_def_clean < 1 -> dropped
    r = rq2.probe_restoration(_probe_states(foot_along=4.0), "A")
    e = {x["defense"]: x for x in r["per_defense"]}["restore"]
    assert e["n_fraction_dropped"] > 0
    assert e["n_fraction_defined"] + e["n_fraction_dropped"] == e["n_groups"]


def test_probe_signature_removal_is_not_restoration():
    """A defence that removes the attack component but jumps somewhere new:
    the probe reports the signature removed; the decomposition reports
    off-axis movement far beyond the footprint (suppression)."""
    pytest.importorskip("sklearn")
    st = _probe_states()
    dim = len(st[next(iter(st))]["clean"])
    e3 = np.eye(dim)[2]
    for t, s in st.items():
        s[("def_clean", "jump")] = s["clean"] + 0.5 * np.eye(dim)[1]
        s[("def_att", "A", "jump")] = s[("def_clean", "jump")] + 8.0 * e3
    pr = {x["defense"]: x for x in rq2.probe_restoration(st, "A")["per_defense"]}["jump"]
    dec = {x["defense"]: x for x in rq2.analyse_attack(st, "A")[0]}["jump"]
    assert pr["signature_removed_fraction_median"] > 0.8
    assert dec["orth_over_footprint_orth_median"] > 5 and dec["p_holm"] < 0.05


def test_probe_floor_and_unreached_tasks():
    pytest.importorskip("sklearn")
    st = _probe_states(n_users=7)
    for t in list(st)[:2]:                                   # make two pairs unreached (a == c)
        st[t][("attacked", "A")] = st[t]["clean"].copy()
    r = rq2.probe_restoration(st, "A")
    assert r["n_tasks"] == 12                                # 14 pairs - 2 unreached
    assert all("p_raw" not in e for e in r["per_defense"])   # 7 user tasks < 8
    r8 = rq2.probe_restoration(_probe_states(n_users=8), "A")
    assert all("p_raw" in e for e in r8["per_defense"])


# --- robustness: 0/1 outcomes, non-finite vectors, missing sweep layers ------

def test_load_outcomes_accepts_0_1(tmp_path):
    d = tmp_path / "o01"
    d.mkdir()
    rows = [{"sample_id": "a", "condition": "attacked", "attack_name": "A", "task_id": "t",
             "defense_name": None, "asr_success": 1, "error": None},
            {"sample_id": "b", "condition": "attacked", "attack_name": "B", "task_id": "t",
             "defense_name": None, "asr_success": 2, "error": None}]
    (d / "attack_results.json").write_text(json.dumps(rows))
    o = rq2.load_outcomes(str(d))["t"]
    assert o[("attacked", "A")] is True and o[("attacked", "B")] is None


def test_nonfinite_vectors_are_dropped_and_counted(tmp_path):
    d = make(tmp_path, n_tasks=6)
    _ep(d, "template_ipi_workspace_user_task_0_injection_task_0", [float("nan"), 0.0])
    info = {}
    st = rq2.load_multi(str(d), [L], info=info)[L]
    assert info["nonfinite"] == 1
    assert ("attacked", "template_ipi") not in st["workspace_user_task_0_injection_task_0"]
    assert rq2.analyse_attack(st, "template_ipi")[1] == 5


def test_holm_ignores_nonfinite():
    es = [{"p": 0.01}, {"p": float("nan")}, {"p": 0.04}]
    rq2.holm(es, "p", "q")
    assert "q" not in es[1] and es[0]["q"] == pytest.approx(0.02) and es[2]["q"] == pytest.approx(0.04)


def test_main_reports_missing_sweep_layers_and_sweeps_other_layers(tmp_path, monkeypatch):
    d = make(tmp_path, n_tasks=8)
    out = tmp_path / "o" / "rq2.json"
    monkeypatch.setattr(sys, "argv", ["rq2_geometry.py", "--results_dir", str(d), "--layer", str(L),
                                      "--sweep_layers", f"{L - 1}-{L}", "--turn", "mean",
                                      "--no_probe", "--out", str(out)])
    rq2.main()
    res = json.loads(out.read_text())
    assert res["layer_sweep_missing"] == [L - 1]
    assert [x["layer"] for x in res["per_attack"]["template_ipi"]["layer_sweep"]["datamarking"]] == [L]


def test_probe_logit_gap_threshold_is_applied():
    pytest.importorskip("sklearn")
    r = rq2.probe_restoration(_probe_states(), "A", min_logit_gap=1e6)
    for e in r["per_defense"]:
        assert e["n_fraction_defined"] == 0 and "signature_removed_fraction_median" not in e



# ===========================================================================
# review round 7
# ===========================================================================
import geometry as geo  # noqa: E402


def test_arrival_or_acts_text_identical_but_activations_differ(tmp_path):
    d = tmp_path / "img"
    d.mkdir()
    _write_tr(d, "c", [_tool("page")], ["act"])
    _write_tr(d, "a", [_tool("page")], ["act"])          # same text: payload is image-only
    _write_tr(d, "u", [_tool("page")], ["act"])
    _turns(d, "c", [[0.0, 0.0]])
    _turns(d, "a", [[3.0, 0.0]])
    _turns(d, "u", [[0.0, 0.0]])
    assert geo.arrival_or_acts(str(d), "c", "a", [L], accept_step0=True) == (0, "ok_acts")
    assert geo.arrival_or_acts(str(d), "c", "a", [L]) == (None, "prompt_same_acts_differ")
    assert geo.arrival_or_acts(str(d), "c", "u", [L], accept_step0=True) == (None, "not_exposed")
    assert geo.arrival_or_acts(str(d), "c", "nope", [L]) == (None, "no_transcript")


def test_arrival_length_differs_status(tmp_path):
    d = tmp_path / "len"
    d.mkdir()
    _write_tr(d, "c", [_tool("x"), _tool("y")], ["a", "b"])
    _write_tr(d, "a", [_tool("x")], ["a"])
    assert geo.arrival_step(str(d), "c", "a") == (None, "length_differs_no_diff")


def _spot_tr(d, sid, key, extra="", payload="", key0="0000000000000000"):
    """spotlighting_delimiting as seen on the cluster (diag_arrival 10-29): the
    delimiter is derived from the tool output and named in the system message,
    so both change when the payload arrives at step 1."""
    sysm = lambda k, x="": {"role": "system", "content": [{"type": "text", "text": f"Data is marked SPOTLIGHT-{k}.{x}"}]}
    msgs = [sysm(key0), {"role": "user", "content": [{"type": "text", "text": "task"}]},
            sysm(key, extra),
            {"role": "user", "content": [{"type": "text", "text": f"[TOOL RESULT] SPOTLIGHT-{key} ok{payload}"}]}]
    steps = [{"input": [0, 1], "prompt_sha256": "s0", "completion": "call f", "tool_calls": [{"function": "f"}]},
             {"input": [2, 1, 3], "prompt_sha256": hashlib.sha256((key + extra + payload).encode()).hexdigest(),
              "completion": "", "tool_calls": []}]
    (d / f"{sid}.transcript.json.gz").write_bytes(gzip.compress(json.dumps({"messages": msgs, "steps": steps}).encode()))


def test_delimiting_arrival_accepted_when_only_the_spotlight_key_differs(tmp_path):
    d = tmp_path / "spot"
    d.mkdir()
    _spot_tr(d, "dc", "bf5fcbac6b6efaf4")
    _spot_tr(d, "da", "8edc7809bad8670b", payload=" INJECT")
    _spot_tr(d, "other", "8edc7809bad8670b", extra=" Also obey the user.", payload=" INJECT")
    assert geo.arrival_step(str(d), "dc", "da") == (1, "ok")
    # any change beyond the key in a non-tool message is still rejected
    assert geo.arrival_step(str(d), "dc", "other") == (None, "not_payload_arrival")
    assert geo.arrival_step(str(d), "dc", "dc") == (None, "not_exposed")
    m = geo._mask_spotlight({"messages": [{"role": "system", "content": "k SPOTLIGHT-8edc7809bad8670b"},
                                          {"role": "user", "content": [{"type": "image"}]}], "steps": []})
    assert m["messages"][0]["content"] == "k SPOTLIGHT-#" and m["messages"][1]["content"] == [{"type": "image"}]


def test_vwa_length_differs_is_scanned_for_an_image_payload(tmp_path):
    """10-29 runs: ~45% of VWA pairs were dropped as length_differs_no_diff --
    the image changed the last shared decision, so the episodes end at different
    steps with identical text on every shared step. VWA scans those turns like
    a text-identical pair; AgentDojo keeps the anomaly status."""
    d = tmp_path / "lenimg"
    d.mkdir()
    _write_tr(d, "c", [_tool("p0"), _tool("p1")], ["a", "b"])
    _write_tr(d, "img", [_tool("p0")], ["a"])               # stops one step earlier
    _write_tr(d, "same", [_tool("p0")], ["a"])
    _turns(d, "c", [[100.0, 0.0], [100.0, 0.0]])
    _turns(d, "img", [[100.0, 30.0]])                        # image moves turn 0
    _turns(d, "same", [[100.0, 0.0]])                        # no difference anywhere shared
    assert geo.arrival_or_acts(str(d), "c", "img", accept_step0=True) == (0, "ok_acts")
    assert geo.arrival_or_acts(str(d), "img", "c", accept_step0=True) == (0, "ok_acts")
    assert geo.arrival_or_acts(str(d), "c", "same", accept_step0=True) == (None, "length_differs_no_diff")
    assert geo.arrival_or_acts(str(d), "c", "img") == (None, "length_differs_no_diff")


def test_normalise_asr():
    assert geo.normalise_asr({"asr_success": True}) is True
    assert geo.normalise_asr({"asr_success": 0}) is False
    assert geo.normalise_asr({"asr_success": 2}) is None
    assert geo.normalise_asr({"asr_success": "yes"}) is None
    assert geo.normalise_asr({"asr_success": True, "scoring_error_final": True}) is None


def _path_dir(tmp, n=9, path_differs=(), missing_pt=(), dedup=False):
    """like _arrival_dir, plus: defended-clean episodes that reach step 1 by a
    different tool call; missing attacked tensors; dedup pointer rows."""
    d = tmp / "pd"
    d.mkdir()
    rows = []
    junk = [50.0, -50.0]
    def ep(sid, cond, atk, tid, dn, msgs, comps, t1, write=True, ptr=None):
        if ptr:
            rows.append({"sample_id": sid, "condition": cond, "attack_name": atk, "task_id": tid,
                         "suite_name": "workspace", "defense_name": dn, "asr_success": None,
                         "error": None, "deduplicated_to": ptr})
            return
        _row(rows, sid, cond, atk, tid, dn)
        _write_tr(d, sid, msgs, comps)
        if write:
            _turns(d, sid, [junk, t1, junk])
    for i in range(n):
        for j in range(2 if dedup else 1):
            tid = f"workspace_user_task_{i}_injection_task_{j}"
            comps = ["call_a", "call_b", "done"]
            ep(f"c_{tid}", "clean", "clean", tid, None, [SYS, _tool("clean"), _tool("n")], comps, [0.0, 0.0])
            ep(f"a_{tid}", "attacked", "A", tid, None, [SYS, _tool("clean P"), _tool("n")], comps,
               [10.0, 0.0], write=(i not in missing_pt))
            dcomps = ["OTHER", "call_b", "done"] if i in path_differs else comps
            rep_tid = f"workspace_user_task_{i}_injection_task_0"
            ep(f"f_{tid}", "defended_clean", "clean", tid, "dm", [SYS, _tool("<c>"), _tool("n")], dcomps,
               [0.0, 1.0], ptr=(f"f_{rep_tid}" if dedup and j == 1 else None))
            ep(f"d_{tid}", "defended_attacked", "A", tid, "dm", [SYS, _tool("<c P>"), _tool("n")], dcomps,
               [2.0, 3.0])
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_defended_pair_must_reach_the_step_by_the_same_path(tmp_path):
    info = {}
    st = rq2.load_multi(str(_path_dir(tmp_path, path_differs=(0, 1))), [L], turn="arrival", info=info)[L]
    assert info["A"]["defended"]["dm"] == {"ok": 7, "defended_path_differs": 2}
    assert rq2.analyse_attack(st, "A")[0][0]["n"] == 7


def test_missing_tensor_is_counted_not_dropped(tmp_path):
    info = {}
    st = rq2.load_multi(str(_path_dir(tmp_path, missing_pt=(3,))), [L], turn="arrival", info=info)[L]
    assert info["A"]["no_tensor"] == 1 and info["A"]["ok"] == 8      # counted once loaded
    assert rq2.analyse_attack(st, "A")[1] == 8


def test_arrival_follows_dedup_pointers(tmp_path):
    info = {}
    st = rq2.load_multi(str(_path_dir(tmp_path, dedup=True)), [L], turn="arrival", info=info)[L]
    assert info["A"]["defended"]["dm"]["ok"] == 18      # pointer rows resolved to the representative
    e = rq2.analyse_attack(st, "A")[0][0]
    assert e["n"] == 18 and e["along_frac_median"] == pytest.approx(0.2)


def test_arrival_nonfinite_dropped(tmp_path):
    d = _arrival_dir(tmp_path)
    _turns(d, "a_workspace_user_task_0_injection_task_0", [[50.0, -50.0], [float("nan"), 0.0], [1.0, 1.0]])
    info = {}
    st = rq2.load_multi(str(d), [L], turn="arrival", info=info)[L]
    assert info["nonfinite"] == 1
    assert rq2.analyse_attack(st, "A")[1] == 8


def test_median_ci95_order_statistics():
    x = np.arange(1.0, 13.0)                          # n=12 -> k=3 -> [x3, x10]
    assert rq2._median_ci95(x) == [3.0, 10.0]
    assert rq2._median_ci95(np.arange(1.0, 9.0)) == [1.0, 8.0]     # n=8 -> k=1
    assert rq2._median_ci95([1.0, 2.0, 3.0]) is None                # too small for 95%


def test_probe_ci_needs_min_test_tasks():
    pytest.importorskip("sklearn")
    r = rq2.probe_restoration(_probe_states(n_users=7), "A")
    assert all("signature_removed_fraction_ci95" not in e for e in r["per_defense"])


def test_probe_wilcoxon_reference_is_defended_clean():
    pytest.importorskip("sklearn")
    # footprint along the probe direction: def_att == def_clean (+noise) is
    # NOT detectable against def_clean, but would be against clean
    r = rq2.probe_restoration(_probe_states(foot_along=2.0), "A")
    e = {x["defense"]: x for x in r["per_defense"]}["restore"]
    assert e["p_raw"] > 0.01


def test_main_writes_json_and_hints_when_vwa_step0_not_accepted(tmp_path, monkeypatch, capsys):
    d = tmp_path / "vwa"
    d.mkdir()
    rows = []
    for i in range(6):
        tid = f"reddit_{i}_wrong_object_cap"
        for sid, cond, atk, dn, txt in [(f"c{i}", "clean", "clean", None, "page"),
                                        (f"a{i}", "attacked", "A", None, "page P"),
                                        (f"f{i}", "defended_clean", "clean", "dm", "<page>"),
                                        (f"d{i}", "defended_attacked", "A", "dm", "<page P>")]:
            _row(rows, sid, cond, atk, tid, dn)
            _write_tr(d, sid, [_tool(txt)], ["act"])
            _turns(d, sid, [[float(i), float("P" in txt)]])
    (d / "attack_results.json").write_text(json.dumps(rows))
    out = tmp_path / "o" / "r.json"
    monkeypatch.setattr(sys, "argv", ["rq2_geometry.py", "--results_dir", str(d), "--layer", str(L),
                                      "--no_probe", "--out", str(out)])
    with pytest.raises(SystemExit) as ex:
        rq2.main()
    assert "--accept_step0" in str(ex.value.code)
    assert json.loads(out.read_text())["arrival"]["A"]["not_payload_arrival"] == 6
    monkeypatch.setattr(sys, "argv", ["rq2_geometry.py", "--results_dir", str(d), "--layer", str(L),
                                      "--no_probe", "--accept_step0", "--out", str(out)])
    rq2.main()
    res = json.loads(out.read_text())
    assert res["arrival"]["A"]["ok"] == 6 and res["per_attack"]["A"]["per_defense"][0]["n"] == 6


def test_rq1_headline_honours_accept_step0(tmp_path):
    d = tmp_path / "h"
    d.mkdir()
    rows = []
    for i in range(3):
        tid = f"reddit_{i}_x"
        for sid, cond, atk, txt in [(f"clean_{tid}", "clean", "clean", "page"), (f"A_{tid}", "attacked", "A", "page P")]:
            rows.append({"sample_id": sid, "condition": cond, "attack_name": atk, "task_id": tid,
                         "defense_name": None, "asr_success": False, "error": None})
            _write_tr(d, sid, [_tool(txt)], ["act"])
            _turns(d, sid, [[1.0, float("P" in txt)]])
    (d / "attack_results.json").write_text(json.dumps(rows))
    _, e0 = geo.displacement_by_outcome_payload(str(d))
    _, e1 = geo.displacement_by_outcome_payload(str(d), accept_step0=True)
    assert e0["A"]["not_payload_arrival"] == 3 and e1["A"]["exposed"] == 3



def test_probe_ci_is_the_order_statistic_interval(monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(rq2, "_median_ci95", lambda x: ["sentinel", len(x)])
    r = rq2.probe_restoration(_probe_states(), "A")
    e = {x["defense"]: x for x in r["per_defense"]}["persist"]
    assert e["signature_removed_fraction_ci95"] == ["sentinel", e["n_fraction_defined"]]


# ===========================================================================
# review round 8
# ===========================================================================

def test_arrival_or_acts_relative_tolerance_and_later_turns(tmp_path):
    d = tmp_path / "r8"
    d.mkdir()
    for sid in ("c", "noise", "late"):
        _write_tr(d, sid, [_tool("p0"), _tool("p1"), _tool("p2")], ["a", "b", "c"])
    _turns(d, "c", [[100.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
    _turns(d, "noise", [[100.001, 0.0], [100.0, 0.001], [100.0, 0.0]])        # GPU-noise level
    _turns(d, "late", [[100.0, 0.0], [100.0, 0.0], [100.0, 40.0]])           # image on a later page
    assert geo.arrival_or_acts(str(d), "c", "noise", [L], accept_step0=True) == (None, "not_exposed")
    assert geo.arrival_or_acts(str(d), "c", "late", [L], accept_step0=True) == (2, "ok_acts")
    assert geo.arrival_or_acts(str(d), "c", "late", [L]) == (None, "prompt_same_acts_differ")


def test_same_path_compares_completions_and_missing_transcripts(tmp_path):
    d = tmp_path / "sp"
    d.mkdir()
    def tr(sid, comps):
        steps = [{"input": [0], "prompt_sha256": "x", "completion": c, "tool_calls": None} for c in comps]
        (d / f"{sid}.transcript.json.gz").write_bytes(gzip.compress(json.dumps({"messages": [], "steps": steps}).encode()))
    tr("a", ["click 3", "type x"]); tr("b", ["click 4", "type x"]); tr("c", ["click 3", "type y"])
    assert geo.same_path(str(d), "a", "b", 1) is False       # VWA: action only in the completion
    assert geo.same_path(str(d), "a", "c", 1) is True
    assert geo.same_path(str(d), "a", "missing", 1) is None


def test_missing_defended_transcript_is_counted(tmp_path):
    d = _path_dir(tmp_path)
    (d / "f_workspace_user_task_0_injection_task_0.transcript.json.gz").unlink()
    info = {}
    rq2.load_multi(str(d), [L], turn="arrival", info=info)
    dm = info["A"]["defended"]["dm"]
    assert dm["ok"] == 8 and sum(v for k, v in dm.items() if k != "ok") == 1


def test_blocked_defence_counts_as_full_restoration(tmp_path):
    """A defence that keeps the payload out entirely (withheld tool result):
    defended-attacked == defended-clean. Kept, at the undefended arrival step,
    with along_frac_def = 0 -- not dropped (which inflated along_frac)."""
    d = tmp_path / "blk"
    d.mkdir()
    rows = []
    for i in range(8):
        tid = f"workspace_user_task_{i}_injection_task_0"
        comps = ["call_a", "call_b", "done"]
        for sid, cond, atk, dn, msgs, t1 in [
                (f"c_{tid}", "clean", "clean", None, [SYS, _tool("x"), _tool("n")], [0.0, 0.0]),
                (f"a_{tid}", "attacked", "A", None, [SYS, _tool("x P"), _tool("n")], [10.0, 0.0]),
                (f"f_{tid}", "defended_clean", "clean", "at", [SYS, _tool("[withheld]"), _tool("n")], [0.0, 1.0]),
                (f"d_{tid}", "defended_attacked", "A", "at", [SYS, _tool("[withheld]"), _tool("n")], [0.0, 1.0])]:
            _row(rows, sid, cond, atk, tid, dn)
            _write_tr(d, sid, msgs, comps)
            _turns(d, sid, [[50.0, -50.0], t1, [50.0, -50.0]])
    (d / "attack_results.json").write_text(json.dumps(rows))
    info = {}
    st = rq2.load_multi(str(d), [L], turn="arrival", info=info)[L]
    assert info["A"]["defended"]["at"] == {"blocked_identical": 8}
    e = rq2.analyse_attack(st, "A")[0][0]
    # its own stratum: not pooled with ordinary pairs, never silently dropped
    assert e["n"] == 0 and e["n_blocked"] == 8 and e["blocked_footprint_nonzero"] == 8
    assert rq2.verdict(e).startswith("BLOCKED")
    rq2.print_attack("A", [e], 8, 8, 0, "x", L)                       # prints without crashing


def _img_only_dir(tmp, prefix="reddit"):
    d = tmp / "imgonly"
    d.mkdir()
    rows = []
    for i in range(8):
        tid = f"{prefix}_{i}_wrong_object_cap"
        for sid, cond, atk, dn, v in [(f"c{i}", "clean", "clean", None, [i + 1.0, 0.0]),
                                      (f"a{i}", "attacked", "A", None, [i + 1.0, 6.0]),
                                      (f"f{i}", "defended_clean", "clean", "dm", [i + 1.0, 1.0]),
                                      (f"d{i}", "defended_attacked", "A", "dm", [i + 3.0, 4.0])]:
            rows.append({"sample_id": sid, "condition": cond, "attack_name": atk, "task_id": tid,
                         "defense_name": dn, "asr_success": True if cond == "attacked" else False,
                         "error": None, **({"scoring_error_final": True} if i == 0 and cond == "attacked" else {})})
            _write_tr(d, sid, [_tool("page")], ["act"])               # identical text everywhere
            _turns(d, sid, [v])
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_rq2_loader_handles_image_only_payloads(tmp_path):
    info = {}
    st = rq2.load_multi(str(_img_only_dir(tmp_path)), [L], turn="arrival", accept_step0=True, info=info)[L]
    assert info["A"]["ok_acts"] == 8 and info["A"]["defended"]["dm"] == {"ok": 8}
    assert rq2.analyse_attack(st, "A")[0][0]["n"] == 8


def test_rq1_headline_image_only_and_unscored(tmp_path):
    d = _img_only_dir(tmp_path)
    summ, exp = geo.displacement_by_outcome_payload(str(d), accept_step0=True)
    assert exp["A"]["exposed"] == 8
    outcomes = summ["A"]
    assert outcomes["unscored"][L]["n"] == 1 and outcomes["success"][L]["n"] == 7



# ===========================================================================
# review round 9
# ===========================================================================

def test_exposure_check_uses_every_layer_not_the_callers(tmp_path):
    d = tmp_path / "lay"
    d.mkdir()
    _write_tr(d, "c", [_tool("p")], ["a"]); _write_tr(d, "a", [_tool("p")], ["a"])
    torch.save([{30: torch.tensor([1.0, 0.0]), 31: torch.tensor([1.0, 0.0])}], d / "c.pt")
    torch.save([{30: torch.tensor([1.0, 9.0]), 31: torch.tensor([1.0, 0.0])}], d / "a.pt")   # only layer 30 differs
    for layers in ([31], [30], None):
        assert geo.arrival_or_acts(str(d), "c", "a", layers, accept_step0=True) == (0, "ok_acts")


def test_agentdojo_text_identical_pairs_must_be_bit_identical(tmp_path):
    d = tmp_path / "bit"
    d.mkdir()
    _write_tr(d, "c", [_tool("p")], ["a"]); _write_tr(d, "a", [_tool("p")], ["a"])
    _turns(d, "c", [[100.0, 0.0]]); _turns(d, "a", [[100.0, 0.001]])     # 1e-5 relative
    assert geo.arrival_or_acts(str(d), "c", "a") == (None, "prompt_same_acts_differ")
    assert geo.arrival_or_acts(str(d), "c", "a", accept_step0=True) == (None, "not_exposed")


def test_same_path_uses_tool_calls_when_present(tmp_path):
    d = tmp_path / "tc"
    d.mkdir()
    def tr(sid, steps):
        (d / f"{sid}.transcript.json.gz").write_bytes(gzip.compress(json.dumps(
            {"messages": [], "steps": [{"input": [0], "prompt_sha256": "x", "completion": c, "tool_calls": tc}
                                       for c, tc in steps]}).encode()))
    tr("u", [("I will search.", ["search(x)"])])
    tr("dfn", [("Sure, searching the marked data.", ["search(x)"])])       # defence reworded the text
    tr("other", [("I will search.", ["read(y)"])])
    assert geo.same_path(str(d), "u", "dfn", 1) is True
    assert geo.same_path(str(d), "u", "other", 1) is False


def test_blocked_pairs_kept_out_of_behaviour_link_and_probe(tmp_path):
    pytest.importorskip("sklearn")
    st = _probe_states()
    for t, s in st.items():                                   # 'restore' blocks the payload everywhere
        s[("def_att", "A", "restore")] = s[("def_clean", "restore")].copy()
        s[("blocked", "A", "restore")] = True
    r = rq2.probe_restoration(st, "A")
    assert "restore" not in {e["defense"] for e in r["per_defense"]}
    rows = rq2.analyse_attack(st, "A")[0]
    e = {x["defense"]: x for x in rows}["restore"]
    assert e["n"] == 0 and e["n_blocked"] == 24


def test_main_behaviour_link_excludes_blocked_pairs(tmp_path, monkeypatch):
    d = tmp_path / "blk2"
    d.mkdir()
    rows = []
    for i in range(8):
        tid = f"workspace_user_task_{i}_injection_task_0"
        comps = ["call_a", "call_b", "done"]
        for sid, cond, atk, dn, msgs, t1, asr in [
                (f"c_{tid}", "clean", "clean", None, [SYS, _tool("x"), _tool("n")], [0.0, 0.0], None),
                (f"a_{tid}", "attacked", "A", None, [SYS, _tool("x P"), _tool("n")], [10.0, 0.0], True),
                (f"f_{tid}", "defended_clean", "clean", "at", [SYS, _tool("[withheld]"), _tool("n")], [0.0, 1.0], None),
                (f"d_{tid}", "defended_attacked", "A", "at", [SYS, _tool("[withheld]"), _tool("n")], [0.0, 1.0], False)]:
            _row(rows, sid, cond, atk, tid, dn)
            rows[-1]["asr_success"] = asr
            _write_tr(d, sid, msgs, comps)
            _turns(d, sid, [[50.0, -50.0], t1, [50.0, -50.0]])
    (d / "attack_results.json").write_text(json.dumps(rows))
    out = tmp_path / "o" / "r.json"
    monkeypatch.setattr(sys, "argv", ["rq2_geometry.py", "--results_dir", str(d), "--layer", str(L),
                                      "--no_probe", "--out", str(out)])
    rq2.main()
    e = json.loads(out.read_text())["per_attack"]["A"]["per_defense"][0]
    assert e["n_blocked"] == 8 and e["by_outcome"]["prevented"]["n"] == 0
