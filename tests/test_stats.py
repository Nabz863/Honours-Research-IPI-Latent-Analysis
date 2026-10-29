"""analysis/stats.py: the arithmetic (Wilson, rank-biserial, reachability,
reach-vs-ASR contingency) on known inputs, and the permutation probe on a
linearly separable fixture (accuracy high, permutation p low) versus pure
noise (accuracy ~chance, p not significant)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import stats as st  # noqa: E402

L = 31


def test_wilson_bounds():
    lo, hi = st.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9) and 0 < hi < 1
    lo, hi = st.wilson(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-9) and 0 < lo < 1
    lo, hi = st.wilson(5, 10)
    assert lo < 0.5 < hi
    assert np.isnan(st.wilson(0, 0)[0])


def test_rank_biserial_sign():
    assert st.rank_biserial([3, 4, 5], [1, 1, 1]) == pytest.approx(1.0)
    assert st.rank_biserial([1, 1, 1], [3, 4, 5]) == pytest.approx(-1.0)
    assert st.rank_biserial([1, 1], [1, 1]) == 0.0      # all ties


def _mk(d: Path, sid, vec):
    torch.save([{L: torch.tensor(vec, dtype=torch.float32)}], d / f"{sid}.pt")


def _pair_dir(tmp, sep, n=12, seed=0):
    """n user tasks, each a clean + attacked ('template_ipi') pair. Attacked
    vector is clean shifted by `sep` on the first axis (plus per-task jitter),
    so a probe can separate them iff sep != 0. asr_success True iff the task
    index is even AND the pair is non-zero, to exercise reach_vs_asr."""
    d = tmp / f"stats_sep{sep}"
    d.mkdir()
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tid = f"workspace_user_task_{i}_injection_task_0"
        base = rng.normal(size=8)
        _mk(d, f"clean_{tid}", base)
        rows.append({"sample_id": f"clean_{tid}", "attack_name": "clean",
                     "task_id": tid, "defense_name": None, "asr_success": None})
        shift = np.zeros(8); shift[0] = sep
        _mk(d, f"template_ipi_{tid}", base + shift)
        rows.append({"sample_id": f"template_ipi_{tid}", "attack_name": "template_ipi",
                     "task_id": tid, "defense_name": "none",
                     "asr_success": bool(i % 2 == 0 and sep != 0)})
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_reachability_counts(tmp_path):
    d = _pair_dir(tmp_path, sep=5.0, n=10)
    pairs = st.load_pairs(str(d), L)
    reach = st.reachability(pairs)["template_ipi"]
    assert reach["n"] == 10 and reach["n_nonzero"] == 10 and reach["rate"] == 1.0
    assert reach["delta_mean_nonzero"] == pytest.approx(5.0, abs=1e-5)


def test_reachability_zero_delta(tmp_path):
    d = _pair_dir(tmp_path, sep=0.0, n=8)
    pairs = st.load_pairs(str(d), L)
    reach = st.reachability(pairs)["template_ipi"]
    assert reach["n_nonzero"] == 0 and reach["rate"] == 0.0


def test_reach_vs_asr_table(tmp_path):
    d = _pair_dir(tmp_path, sep=5.0, n=10)       # all reached; evens are ASR hits
    pairs = st.load_pairs(str(d), L)
    rva = {e["attack"]: e for e in st.reach_vs_asr(pairs)}["template_ipi"]
    (a, b), (c, cc) = rva["table"]
    assert a == 5 and b == 5           # 5 reached+hit, 5 reached+miss
    assert c == 0 and cc == 0          # nothing unreached
    assert rva["asr_given_nonzero"] == pytest.approx(0.5)


def test_probe_separable_vs_noise(tmp_path):
    p1 = st.load_pairs(str(_pair_dir(tmp_path, sep=8.0, n=16, seed=1)), L)
    sep = st.probe(p1, p1, n_perm=100)["template_ipi"]
    assert sep["accuracy"] > 0.9
    assert sep["p_perm"] <= 0.05

    p0 = st.load_pairs(str(_pair_dir(tmp_path, sep=0.0, n=16, seed=2)), L)
    noise = st.probe(p0, p0, n_perm=100)["template_ipi"]
    # sep=0 -> clean and attacked are identical -> every pair excluded
    assert noise.get("accuracy") is None or noise["n"] < 12


# ===========================================================================
# Regression tests B, C, G (2026-10-22). Each fails on the pre-fix code
# (verified by reverting the fix) and passes on the fix.
# ===========================================================================


def _row_g(rows, sid, cond, attack, task, asr=None, error=None):
    rows.append({"sample_id": sid, "condition": cond, "attack_name": attack,
                 "task_id": task, "suite_name": "workspace", "defense_name": None,
                 "asr_success": asr, "error": error})


# --- B (P2a.2): unscored episodes are excluded, not counted as misses -------

def test_B_reach_vs_asr_excludes_unscored():
    hc, ha = np.zeros(3), np.ones(3)                  # reached pair
    items = [("t0", "g0", hc, ha, True), ("t1", "g1", hc, ha, True),
             ("t2", "g2", hc, ha, None), ("t3", "g3", hc, ha, None),
             ("t4", "g4", hc, ha, None),
             ("t5", "g5", hc, hc, False),             # unreached, scored miss
             ("t6", "g6", hc, hc, None), ("t7", "g7", hc, hc, None)]  # unreached, unscored
    e = st.reach_vs_asr({"atk": items})[0]
    assert e["table"] == [[2, 0], [0, 1]]
    assert e["asr_given_nonzero"] == pytest.approx(1.0)


# --- C (audit 2b.1): errored rows never enter the RQ1 stats ------------------

def test_C_load_pairs_skips_errored_rows_with_loadable_tensors(tmp_path):
    d = tmp_path / "final_c"
    d.mkdir()
    rows = []
    for i in range(4):
        tid = f"workspace_user_task_{i}_injection_task_0"
        _row_g(rows, f"clean_{tid}", "clean", "clean", tid)
        _mk(d, f"clean_{tid}", [0.0, 0.0])
        _row_g(rows, f"tipi_{tid}", "attacked", "template_ipi", tid, asr=False)
        _mk(d, f"tipi_{tid}", [1.0, 0.0])
    # task 0: the attacked row is a later errored retry with a loadable tensor
    # that overwrites the good one in by_task if errors are not skipped
    t0 = "workspace_user_task_0_injection_task_0"
    _row_g(rows, f"tipi_retry_{t0}", "attacked", "template_ipi", t0, error="CUDA OOM")
    _mk(d, f"tipi_retry_{t0}", [999.0, 0.0])
    # task 3: the CLEAN row errored -> the whole pair must go
    for r in rows:
        if r["sample_id"] == "clean_workspace_user_task_3_injection_task_0":
            r["error"] = "timeout"
    (d / "attack_results.json").write_text(json.dumps(rows))

    pairs = st.load_pairs(str(d), L)
    items = pairs["template_ipi"]
    assert sorted(t for t, *_ in items) == [
        f"workspace_user_task_{i}_injection_task_0" for i in range(3)]
    for _, _, hc, ha, _ in items:
        assert np.linalg.norm(ha - hc) == pytest.approx(1.0)   # never the 999 retry


# --- G (finding 4): main() survives a zero-reach attack and writes the JSON
#     before printing ---------------------------------------------------------

def _zero_reach_dir(tmp_path):
    """Three attacks over 8 tasks: two never reach (attacked == clean), one
    does. Zero reach makes delta_mean_nonzero/median None, the z1-vs-z2
    comparison has n_pairs=0 (medians None), and their probes are skipped."""
    d = tmp_path / "final_g"
    d.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i in range(8):
        tid = f"workspace_user_task_{i}_injection_task_0"
        cv = rng.normal(size=4)
        _row_g(rows, f"clean_{tid}", "clean", "clean", tid)
        _mk(d, f"clean_{tid}", cv)
        for z in ("z1", "z2"):
            _row_g(rows, f"{z}_{tid}", "attacked", z, tid, asr=False)
            _mk(d, f"{z}_{tid}", cv)
        _row_g(rows, f"r_{tid}", "attacked", "reached", tid, asr=bool(i % 2))
        _mk(d, f"r_{tid}", cv + rng.normal(size=4))
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def test_G_zero_reach_attack_does_not_kill_the_run(tmp_path, monkeypatch, capsys):
    d = _zero_reach_dir(tmp_path)
    out = tmp_path / "out" / "stats.json"
    monkeypatch.setattr(sys, "argv", ["stats.py", "--results_dir", str(d),
                                      "--layer", str(L), "--n_perm", "5", "--turn", "mean",
                                      "--out", str(out)])
    st.main()                                       # must not raise
    res = json.loads(out.read_text())
    assert res["reachability"]["z1"]["n_nonzero"] == 0
    assert res["reachability"]["z1"]["delta_mean_nonzero"] is None
    assert res["reachability"]["reached"]["n_nonzero"] == 8
    printed = capsys.readouterr().out
    assert "z1" in printed and "n/a" in printed


def test_G_json_is_written_before_any_report_printing(tmp_path, monkeypatch):
    """If the report print dies for ANY reason, the output file must already
    exist -- the 5.5 failure mode lost hours of permutation work."""
    d = _zero_reach_dir(tmp_path)
    out = tmp_path / "out" / "stats.json"
    monkeypatch.setattr(sys, "argv", ["stats.py", "--results_dir", str(d),
                                      "--layer", str(L), "--n_perm", "5", "--turn", "mean",
                                      "--out", str(out)])

    def exploding_print(*args, **kw):
        if args and "REACHABILITY" in str(args[0]):
            raise RuntimeError("report printing failed")

    monkeypatch.setattr(st, "print", exploding_print, raising=False)
    with pytest.raises(RuntimeError):
        st.main()
    assert out.exists(), "JSON not on disk when the report print failed"
    assert "reachability" in json.loads(out.read_text())


# --- review finding 6: scoring_error_final rows are unscored, not misses -----

def test_scoring_error_final_is_excluded_from_asr(tmp_path):
    d = tmp_path / "final_sef"
    d.mkdir()
    rows = []
    for i in range(4):
        tid = f"workspace_user_task_{i}_injection_task_0"
        _row_g(rows, f"clean_{tid}", "clean", "clean", tid)
        _mk(d, f"clean_{tid}", [0.0, 0.0])
        _row_g(rows, f"tipi_{tid}", "attacked", "template_ipi", tid, asr=(i == 0))
        _mk(d, f"tipi_{tid}", [1.0, 0.0])
    rows[-1]["scoring_error_final"] = True          # task 3: scorer gave up, asr False
    rows[-1]["asr_success"] = False
    (d / "attack_results.json").write_text(json.dumps(rows))
    pairs = st.load_pairs(str(d), L)
    assert len(pairs["template_ipi"]) == 4           # activations still count (RQ1)
    e = st.reach_vs_asr(pairs)[0]
    assert e["table"] == [[1, 2], [0, 0]]            # 1 hit, 2 misses; task 3 out


# --- review finding 7: an untestable 2x2 gets no p, and no Holm slot ---------

def test_degenerate_fisher_table_has_no_pvalue():
    pytest.importorskip("scipy")
    hc, ha = np.zeros(2), np.ones(2)
    reached_only = [(f"t{i}", f"g{i}", hc, ha, i % 2 == 0) for i in range(8)]   # no unreached row
    # testable: 4 user tasks reached & hit, 4 unreached & missed
    mixed = [(f"workspace_user_task_{u}_injection_task_{k}", f"workspace_user_task_{u}", hc, ha if u < 4 else hc, u < 4)
             for u in range(8) for k in range(3)]
    # an attack with 0% ASR: reach varies but the hit COLUMN is empty
    all_miss = ([(f"m{i}", f"q{i}", hc, ha, False) for i in range(5)] +
                [(f"n{i}", f"w{i}", hc, hc, False) for i in range(5)])
    out = {e["attack"]: e for e in st.reach_vs_asr(
        {"deg": reached_only, "ok": mixed, "zero_asr": all_miss})}
    assert "p_raw" not in out["deg"] and "p_holm" not in out["deg"]
    assert "p_raw" not in out["zero_asr"] and "degenerate" in out["zero_asr"]["note"]
    assert "degenerate" in out["deg"]["note"]
    assert out["ok"]["p_holm"] == pytest.approx(out["ok"]["p_raw"])   # Holm m == 1


# --- review finding 5: stats uses the matched-pairs rank-biserial ------------

def test_stats_rank_biserial_uses_ranks():
    assert st.rank_biserial([0.1] * 6 + [0.0] * 4, [0.0] * 6 + [10.0] * 4) == pytest.approx(-13 / 55)


# --- review finding 3: stats groups VWA variants by base task ----------------

def test_load_pairs_groups_vwa_by_base_task(tmp_path):
    d = tmp_path / "final_vwa"
    d.mkdir()
    rows = []
    for k in (1, 2):
        tid = f"reddit_36_wrong_object_cap_{k}"
        _row_g(rows, f"clean_{tid}", "clean", "clean", tid)
        _mk(d, f"clean_{tid}", [0.0, 0.0])
        _row_g(rows, f"img_{tid}", "attacked", "img", tid)
        _mk(d, f"img_{tid}", [1.0, 0.0])
    (d / "attack_results.json").write_text(json.dumps(rows))
    groups = {g for _, g, *_ in st.load_pairs(str(d), L)["img"]}
    assert groups == {"reddit_36"}


# --- review finding 11: every report field is None-safe ----------------------

def test_print_report_never_formats_none(capsys):
    from types import SimpleNamespace
    a = SimpleNamespace(results_dir="x", layer=31)
    reach = {"z": {"n": 4, "n_nonzero": 0, "rate": 0.0, "rate_ci95": [0.0, 0.5],
                   "delta_mean_nonzero": None, "delta_median_nonzero": None}}
    cmp_ = [{"attack_a": "z", "attack_b": "y", "n_pairs": 0, "median_a": None,
             "median_b": None, "rank_biserial_r": 0.0}]
    prb = {"z": {"n_pairs": 6, "accuracy": None, "null_mean": None, "null_p95": None,
                 "p_perm": None}}
    rva = [{"attack": "z", "table": [[0, 0], [0, 4]], "asr_given_nonzero": None,
            "asr_given_zero": 0.0, "odds_ratio_mh": None, "note": "degenerate"}]
    st.print_report(a, reach, cmp_, prb, rva)
    assert "None" not in capsys.readouterr().out


def test_compare_attacks_reports_the_rank_biserial_in_a_minus_b_direction():
    # per-task ||dh||: A - B = +0.1 on six tasks, -10 on four -> r = -13/55
    hc = np.zeros(2)
    A = [1.1] * 6 + [1.0] * 4
    B = [1.0] * 6 + [11.0] * 4
    pairs = {"A": [(f"t{i}", f"g{i}", hc, np.array([a, 0.0]), None) for i, a in enumerate(A)],
             "B": [(f"t{i}", f"g{i}", hc, np.array([b, 0.0]), None) for i, b in enumerate(B)]}
    e = st.compare_attacks(pairs)[0]
    assert (e["attack_a"], e["attack_b"]) == ("A", "B")
    assert e["rank_biserial_r"] == pytest.approx(-13 / 55)



# --- review finding 2: the user task is the independent unit in stats.py ----

def _rep_pairs(spec_a, spec_b):
    """spec: {user_task: [||dh|| per injection task]} -> pairs for two attacks"""
    hc = np.zeros(2)
    mk = lambda spec: [(f"workspace_user_task_{u}_injection_task_{j}",
                        f"workspace_user_task_{u}", hc, np.array([v, 0.0]), None)
                       for u, vs in spec.items() for j, v in enumerate(vs)]
    return {"A": mk(spec_a), "B": mk(spec_b)}


def test_compare_attacks_one_observation_per_user_task():
    sps = pytest.importorskip("scipy.stats")
    # 6 user tasks x 3 injection tasks; A beats B by 2 on five tasks, loses by
    # 0.5 on one: per user task 5 up / 1 down -> p = 0.0625 (n.s.); counting the
    # 18 repeats as independent gives p << 0.05
    A = {u: [3.0] * 3 if u < 5 else [0.5] * 3 for u in range(6)}
    B = {u: [1.0] * 3 for u in range(6)}
    e = st.compare_attacks(_rep_pairs(A, B))[0]
    _, p_grp = sps.wilcoxon([3.0] * 5 + [0.5], [1.0] * 6, zero_method="wilcox")
    _, p_pair = sps.wilcoxon([3.0] * 15 + [0.5] * 3, [1.0] * 18, zero_method="wilcox")
    assert p_pair < 0.05 < p_grp
    assert e["n_pairs"] == 18 and e["n_groups"] == 6
    assert e["p_raw"] == pytest.approx(p_grp)
    assert e["rank_biserial_r"] == pytest.approx(19 / 21)


def test_reachability_ci_uses_the_clustered_effective_n():
    # 10 user tasks x 6 injection tasks; reach is all-or-nothing per user task
    hc, ha = np.zeros(2), np.ones(2)
    items = [(f"t{u}_{j}", f"g{u}", hc, ha if u < 5 else hc, None)
             for u in range(10) for j in range(6)]
    r = st.reachability({"x": items})["x"]
    assert r["rate"] == pytest.approx(0.5) and r["n_groups"] == 10
    # cluster-robust var = 10/9 * 10*9 / 60^2 -> n_eff = 0.25 / that = 9
    assert r["n_eff"] == pytest.approx(9.0)
    assert r["rate_ci95"] == pytest.approx(list(st.wilson_p(0.5, 9.0)))
    assert r["rate_ci95"][1] - r["rate_ci95"][0] > np.subtract(*st.wilson(30, 60)[::-1])


def test_reach_vs_asr_secondary_is_within_user_tasks():
    hc, ha = np.zeros(2), np.ones(2)
    items = []
    # confounding between user tasks: 6 tasks always reached & always hit,
    # 6 tasks never reached & never hit (non-informative within task) ...
    for u in range(6):
        items += [(f"y{u}_{j}", f"y{u}", hc, ha, True) for j in range(4)]
        items += [(f"z{u}_{j}", f"z{u}", hc, hc, False) for j in range(4)]
    # ... and 6 informative tasks with NO within-task association
    for u in range(6):
        for k, (rch, hit) in enumerate([(1, 1), (1, 0), (0, 1), (0, 0)]):
            items.append((f"x{u}_{k}", f"x{u}", hc, ha if rch else hc, bool(hit)))
    e = st.reach_vs_asr({"atk": items}, n_perm=2000)[0]
    assert e["n_informative_groups"] == 6
    assert e["p_within"] > 0.5                                # within task: none


def test_reach_vs_asr_detects_a_within_task_association():
    hc, ha = np.zeros(2), np.ones(2)
    items = [(f"g{u}_{k}", f"g{u}", hc, ha if k < 2 else hc, k < 2)
             for u in range(8) for k in range(4)]
    e = st.reach_vs_asr({"atk": items}, n_perm=2000)[0]
    assert e["p_within"] < 0.01 and e["n_informative_groups"] == 8


def test_compare_attacks_needs_six_user_tasks_not_six_pairs():
    # 3 user tasks x 4 injection tasks: 12 pairs, 3 independent observations
    A = {u: [3.0] * 4 for u in range(3)}
    B = {u: [1.0] * 4 for u in range(3)}
    e = st.compare_attacks(_rep_pairs(A, B))[0]
    assert e["n_pairs"] == 12 and e["n_groups"] == 3 and "p_raw" not in e


def test_reach_vs_asr_is_two_sided():
    # within every user task the reached episodes MISS and the unreached HIT
    hc, ha = np.zeros(2), np.ones(2)
    items = [(f"g{u}_{k}", f"g{u}", hc, ha if k < 2 else hc, k >= 2)
             for u in range(8) for k in range(4)]
    e = st.reach_vs_asr({"atk": items}, n_perm=2000)[0]
    assert e["p_within"] < 0.01 and e["odds_ratio_mh"] == 0.0



# --- reach vs ASR, primary: between user tasks, one observation per task ----

def test_reach_vs_asr_primary_is_one_observation_per_user_task():
    sps = pytest.importorskip("scipy.stats")
    hc, ha = np.zeros(2), np.ones(2)
    # 2 reached tasks that always hit, 2 unreached that never do, 10 repeats each:
    # pooled Fisher (40 "independent" episodes) is overwhelming; with 4 user
    # tasks the best a permutation can do is 2 / C(4,2) -> p ~ 1/3
    items = [(f"workspace_user_task_{u}_injection_task_{k}", f"workspace_user_task_{u}", hc, ha if u < 2 else hc, u < 2)
             for u in range(4) for k in range(10)]
    e = st.reach_vs_asr({"atk": items}, n_perm=3000)[0]
    (a, b), (c, d) = e["table"]
    assert sps.fisher_exact([[a, b], [c, d]])[1] < 1e-6
    assert e["n_tasks"] == 4 and e["p_raw"] == pytest.approx(1 / 3, abs=0.03)
    assert e["p_within"] is None and e["n_informative_groups"] == 0


def test_reach_vs_asr_primary_detects_task_level_effect_with_fixed_reach():
    # AgentDojo-like: reach fixed per user task; ASR only on reached tasks
    rng = np.random.default_rng(1)
    hc, ha = np.zeros(2), np.ones(2)
    items = []
    for u in range(40):
        reached = u < 20
        for k in range(6):
            items.append((f"workspace_user_task_{u}_injection_task_{k}", f"workspace_user_task_{u}", hc, ha if reached else hc,
                          bool(reached and rng.random() < 0.4)))
    e = st.reach_vs_asr({"atk": items}, n_perm=3000)[0]
    assert e["p_raw"] < 0.01 and e["n_informative_groups"] == 0
    assert e["asr_unreached_tasks"] == 0.0 and e["asr_reached_tasks"] > 0.2


def test_reach_vs_asr_primary_null_holds():
    rng = np.random.default_rng(2)
    hc, ha = np.zeros(2), np.ones(2)
    items = [(f"workspace_user_task_{u}_injection_task_{k}", f"workspace_user_task_{u}", hc, ha if u % 2 else hc, bool(rng.random() < 0.3))
             for u in range(30) for k in range(5)]
    assert st.reach_vs_asr({"atk": items}, n_perm=2000)[0]["p_raw"] > 0.05


# --- review round 4 mutation gaps ------------------------------------------

def test_compare_attacks_collapse_is_the_mean_with_unequal_repeats():
    sps = pytest.importorskip("scipy.stats")
    # unequal repeat counts and varying values: mean != median != sum per task
    # relative to B=1.5 the per-task mean, median and sum disagree in sign
    A = {0: [1.0, 1.0, 7.0], 1: [0.9, 0.9], 2: [4.0], 3: [0.2, 0.2, 0.2, 6.0],
         4: [0.5], 5: [2.0, 2.6], 6: [0.6, 0.6, 0.6], 7: [3.3]}
    B = {u: [1.5] * len(v) for u, v in A.items()}
    e = st.compare_attacks(_rep_pairs(A, B))[0]
    ga = np.array([np.mean(v) for v in A.values()]); gb = np.full(8, 1.5)
    _, p = sps.wilcoxon(ga, gb, zero_method="wilcox")
    assert e["p_raw"] == pytest.approx(p)
    assert e["rank_biserial_r"] == pytest.approx(st.rank_biserial(ga, gb))
    allv = np.concatenate([v for v in A.values()])
    assert e["median_a"] == pytest.approx(np.median(allv))        # descriptive: per pair
    for alt in (np.array([np.median(v) for v in A.values()]), np.array([np.sum(v) for v in A.values()])):
        _, pa = sps.wilcoxon(alt, gb, zero_method="wilcox")
        assert not (pa == pytest.approx(p) and
                    st.rank_biserial(alt, gb) == pytest.approx(e["rank_biserial_r"]))


def test_effective_n_edge_cases():
    # p = 0: no information about clustering -> one count per user task
    assert st.effective_n([0] * 12, [u // 3 for u in range(12)]) == (4.0, 4)
    # a single user task carries one observation's worth
    assert st.effective_n([1, 0, 1, 0], ["g"] * 4) == (1.0, 1)
    # anti-correlated within task (every task exactly half reached): variance
    # ~0 would give n_eff >> N; it is clipped to N
    y = [1, 0] * 10
    n_eff, G = st.effective_n(y, [i // 2 for i in range(20)])
    assert G == 10 and n_eff == pytest.approx(20.0)
    # nearly balanced: the formula exceeds N and must be clipped to N
    y = [1, 0] * 9 + [1, 1]
    n_eff, _ = st.effective_n(y, [i // 2 for i in range(20)])
    assert n_eff == pytest.approx(20.0)


def test_within_task_secondary_handles_between_task_confounding_among_informative():
    # informative tasks differ in BOTH reach rate and hit rate (between-task
    # confounding), with no association inside any task. A shuffle pooled
    # across tasks would report one; the within-task null must not.
    hc, ha = np.zeros(2), np.ones(2)
    items = []
    for u in range(15):      # high reach, high hit; within: hits independent of reach
        for k, (rch, hit) in enumerate([(1, 1)] * 6 + [(1, 0)] * 2 + [(0, 1)] * 3 + [(0, 0)] * 1):
            items.append((f"p{u}_{k}", f"p{u}", hc, ha if rch else hc, bool(hit)))
    for u in range(15):      # low reach, low hit; same within-task independence
        for k, (rch, hit) in enumerate([(1, 1)] * 1 + [(1, 0)] * 3 + [(0, 1)] * 2 + [(0, 0)] * 6):
            items.append((f"q{u}_{k}", f"q{u}", hc, ha if rch else hc, bool(hit)))
    e = st.reach_vs_asr({"atk": items}, n_perm=2000)[0]
    assert e["n_informative_groups"] == 30
    assert e["p_within"] > 0.2


def test_within_task_pvalue_never_zero_and_ties_count():
    hc, ha = np.zeros(2), np.ones(2)
    # 12 tasks, perfect within-task association: no permutation is as extreme
    strong = [(f"g{u}_{k}", f"g{u}", hc, ha if k < 2 else hc, k < 2)
              for u in range(12) for k in range(4)]
    assert st.reach_vs_asr({"a": strong}, n_perm=200)[0]["p_within"] == pytest.approx(1 / 201)
    # one task, reach [T, F], hits [T, F]: every permutation ties |a - E| -> p = 1
    tie = [("t0", "g", hc, ha, True), ("t1", "g", hc, hc, False)]
    assert st.reach_vs_asr({"b": tie}, n_perm=200)[0]["p_within"] == pytest.approx(1.0)


# --- review round 5: the between-task null is stratified by suite and size --

def _task_items(spec, seed=0):
    """spec: list of (suite, n_tasks, reached, asr_prob, n_scored)"""
    rng = np.random.default_rng(seed)
    hc, ha = np.zeros(2), np.ones(2)
    items, nxt = [], {}
    for suite, n_tasks, reached, pr, n_sc in spec:
        for _ in range(n_tasks):
            u = nxt[suite] = nxt.get(suite, -1) + 1        # unique task ids per suite
            g = f"{suite}_user_task_{u}"
            for k in range(n_sc):
                items.append((f"{g}_injection_task_{k}", g, hc, ha if reached else hc,
                              bool(rng.random() < pr)))
    return items


def test_reach_vs_asr_suite_differences_are_not_a_reach_effect():
    # within each suite reach does not predict ASR; banking is high-reach AND
    # high-ASR, workspace low-reach AND low-ASR. An unstratified task-level
    # permutation reads the suite difference as a reach effect.
    spec = [("banking", 10, True, 0.5, 6), ("banking", 10, False, 0.5, 6),
            ("workspace", 4, True, 0.02, 6), ("workspace", 30, False, 0.02, 6)]
    e = st.reach_vs_asr({"atk": _task_items(spec, seed=3)}, n_perm=2000)[0]
    assert e["p_raw"] > 0.05


def test_reach_vs_asr_size_imbalance_is_not_a_reach_effect():
    # reached tasks have 1 scored episode, unreached 9; same ASR everywhere.
    # 0/1 task rates vs smooth ones are not exchangeable: no shared stratum.
    spec = [("workspace", 10, True, 0.05, 1), ("workspace", 87, False, 0.05, 9)]
    e = st.reach_vs_asr({"atk": _task_items(spec, seed=12)}, n_perm=2000)[0]   # unstratified: p=0.008
    assert e.get("p_raw") is None or e["p_raw"] > 0.05
    assert e["mean_scored_episodes_per_task"] == {"reached": 1.0, "unreached": 9.0}


def test_reach_vs_asr_needs_two_tasks_per_reach_class():
    spec = [("workspace", 1, True, 1.0, 6), ("workspace", 8, False, 0.1, 6)]
    e = st.reach_vs_asr({"atk": _task_items(spec, seed=5)}, n_perm=500)[0]
    assert "p_raw" not in e and "p_holm" not in e



# --- payload-arrival turn ----------------------------------------------------
import gzip, hashlib  # noqa: E402


def _tr(d, sid, msgs, comps):
    steps = [{"input": list(range(k + 1)),
              "prompt_sha256": hashlib.sha256(json.dumps(msgs[: k + 1]).encode()).hexdigest(),
              "completion": c, "tool_calls": [c]} for k, c in enumerate(comps)]
    (d / f"{sid}.transcript.json.gz").write_bytes(
        gzip.compress(json.dumps({"messages": msgs, "steps": steps}).encode()))


def test_load_pairs_arrival_mode(tmp_path):
    d = tmp_path / "arr"
    d.mkdir()
    S = {"role": "system", "content": [{"type": "text", "text": "s"}]}
    T = lambda x: {"role": "user", "content": [{"type": "text", "text": "[TOOL RESULT] " + x}]}
    rows = []
    for i in range(4):
        tid = f"workspace_user_task_{i}_injection_task_0"
        cm = [S, T("ok"), T("more")]
        am = [S, T("ok PAYLOAD"), T("more")] if i < 2 else cm      # tasks 2,3 never exposed
        ac = ["x", "y", "z"] if i != 1 else ["DIFF", "y", "z"]     # task 1: not the payload arriving
        _row_g(rows, f"c_{tid}", "clean", "clean", tid)
        _tr(d, f"c_{tid}", cm, ["x", "y", "z"])
        torch.save([{L: torch.tensor(v)} for v in ([9.0, 9.0], [1.0, 0.0], [9.0, 9.0])], d / f"c_{tid}.pt")
        _row_g(rows, f"a_{tid}", "attacked", "A", tid, asr=(i == 0))
        _tr(d, f"a_{tid}", am, ac)
        av = ([9.0, 9.0], [4.0, 0.0], [-9.0, 9.0]) if i < 2 else ([9.0, 9.0], [1.0, 0.0], [9.0, 9.0])
        torch.save([{L: torch.tensor(v)} for v in av], d / f"a_{tid}.pt")      # unexposed: identical
    (d / "attack_results.json").write_text(json.dumps(rows))
    info = {}
    pairs = st.load_pairs(str(d), L, turn="arrival", info=info)["A"]
    assert info["A"] == {"ok": 1, "not_payload_arrival": 1, "not_exposed": 2}
    by = {t: (hc, ha) for t, _, hc, ha, _ in pairs}
    assert len(by) == 3
    hc, ha = by["workspace_user_task_0_injection_task_0"]
    assert np.allclose(hc, [1.0, 0.0]) and np.allclose(ha, [4.0, 0.0])     # turn 1, not the mean
    hc, ha = by["workspace_user_task_2_injection_task_0"]
    assert np.allclose(hc, ha)                                             # unexposed: unreached
    r = st.reachability({"A": pairs})["A"]
    assert (r["n"], r["n_nonzero"]) == (3, 1)



def test_stats_holm_ignores_nonfinite():
    es = [{"p": 0.01}, {"p": float("nan")}, {"p": 0.04}]
    st.holm(es, "p", "q")
    assert "q" not in es[1] and es[0]["q"] == pytest.approx(0.02)


def test_load_pairs_image_only_vwa_payload(tmp_path):
    d = tmp_path / "img"
    d.mkdir()
    T = lambda x: {"role": "user", "content": [{"type": "text", "text": "[TOOL RESULT] " + x}]}
    rows = []
    for i in range(4):
        tid = f"reddit_{i}_wrong_object_cap"
        _row_g(rows, f"c_{tid}", "clean", "clean", tid)
        _tr(d, f"c_{tid}", [T("page")], ["act"])
        torch.save([{L: torch.tensor([0.0, 0.0])}], d / f"c_{tid}.pt")
        _row_g(rows, f"a_{tid}", "attacked", "img", tid, asr=1 if i == 0 else 0)   # 0/1 scores
        _tr(d, f"a_{tid}", [T("page")], ["act"])                                    # text identical
        torch.save([{L: torch.tensor([2.0, 0.0])}], d / f"a_{tid}.pt")
    (d / "attack_results.json").write_text(json.dumps(rows))
    info = {}
    pairs = st.load_pairs(str(d), L, turn="arrival", accept_step0=True, info=info)["img"]
    assert info["img"] == {"ok_acts": 4}
    assert st.reachability({"img": pairs})["img"]["n_nonzero"] == 4                # not "unreached"
    assert all(type(a) is bool for *_, a in pairs)                                  # 0/1 normalised
    assert sorted(a for *_, a in pairs) == [False, False, False, True]
    info = {}
    assert st.load_pairs(str(d), L, turn="arrival", info=info) == {}
    assert info["img"] == {"prompt_same_acts_differ": 4}


def test_stats_empty_writes_json_and_hints(tmp_path, monkeypatch):
    d = tmp_path / "v"
    d.mkdir()
    T = lambda x: {"role": "user", "content": [{"type": "text", "text": "[TOOL RESULT] " + x}]}
    rows = []
    for i in range(3):
        tid = f"reddit_{i}_x"
        for sid, cond, atk, txt in [(f"c{i}", "clean", "clean", "page"), (f"a{i}", "attacked", "A", "page P")]:
            _row_g(rows, sid, cond, atk, tid)
            _tr(d, sid, [T(txt)], ["act"])
            torch.save([{L: torch.tensor([1.0, float("P" in txt)])}], d / f"{sid}.pt")
    (d / "attack_results.json").write_text(json.dumps(rows))
    out = tmp_path / "o" / "s.json"
    monkeypatch.setattr(sys, "argv", ["stats.py", "--results_dir", str(d), "--layer", str(L),
                                      "--n_perm", "5", "--out", str(out)])
    with pytest.raises(SystemExit) as ex:
        st.main()
    assert "--accept_step0" in str(ex.value.code)
    assert json.loads(out.read_text())["arrival"]["A"]["not_payload_arrival"] == 3
