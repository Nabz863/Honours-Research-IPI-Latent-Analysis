"""analysis/directions.py on synthetic states with planted directions: the
leave-one-group-out consistency and its sign-flip null, matched cosines between
attacks with a task-permutation null, the grouped-CV detector (no leakage of
held-out user tasks, cross-attack transfer only when directions are shared,
all-step negatives from held-out tasks only), defence footprint and suppression
cosines (attack direction without the footprint's own task, suppression from the
defended-clean state, blocked pairs excluded), and end-to-end runs whose
topology must reproduce persistent_homology's JSON."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import directions as dr  # noqa: E402

D = 24
W0 = np.zeros(D); W0[0] = 1.0
W1 = np.zeros(D); W1[1] = 1.0
W2 = np.zeros(D); W2[2] = 1.0


def _pairs(n_users, delta_fn, reps=2, seed=0, asr_fn=None, prefix="workspace"):
    rng = np.random.default_rng(seed)
    out = []
    for u in range(n_users):
        base = rng.normal(size=D) * 5
        for j in range(reps):
            tid = f"{prefix}_user_task_{u}_injection_task_{j}"
            dlt = delta_fn(u, j, rng)
            asr = asr_fn(u, j) if asr_fn else None
            out.append((tid, f"{prefix}_user_task_{u}", base, base + dlt, asr))
    return out


def _negs(pairs, k=3, seed=9):
    rng = np.random.default_rng(seed)
    out = {}
    for _, g, c, _, _ in pairs:
        out.setdefault(g, [c + 0.1 * rng.normal(size=D) for _ in range(k)])
    return out


def _folds(pairs, k=5):
    return dr.make_folds({p[1] for p in pairs}, k, np.random.default_rng(0))


# ------------------------------------------------------------ primitives
def test_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    pos = rng.integers(0, 5, 40).astype(float)
    neg = rng.integers(0, 5, 30).astype(float)
    y = np.r_[np.ones(40), np.zeros(30)]
    assert dr.auroc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]))
    sp, dr.sps = dr.sps, None                     # the fallback ranker agrees too
    try:
        assert dr.auroc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]))
    finally:
        dr.sps = sp


def test_logo_from_gram_matches_direct():
    rng = np.random.default_rng(2)
    M = rng.normal(size=(9, D))
    s = rng.choice([-1.0, 1.0], 9)
    direct = [dr.cos(s[g] * M[g], (s[:, None] * M).sum(0) - s[g] * M[g]) for g in range(9)]
    assert dr._logo_cos_from_gram(M @ M.T, s) == pytest.approx(np.mean(direct))
    assert dr._logo_cos_from_gram(M @ M.T, s, np.median) == pytest.approx(np.median(direct))


def test_pairs_of_excludes_unreached_and_reads_outcomes():
    c, a = np.ones(D), np.ones(D) + W0
    st = {"workspace_user_task_0_injection_task_0": {("clean", "A"): c, ("attacked", "A"): a},
          "workspace_user_task_1_injection_task_0": {("clean", "A"): c, ("attacked", "A"): c.copy()},
          "workspace_user_task_2_injection_task_0": {("clean", "A"): c}}
    oc = {"workspace_user_task_0_injection_task_0": {("attacked", "A"): True}}
    ps = dr.pairs_of(st, oc, "A")
    assert [(p[0], p[1], p[4]) for p in ps] == [("workspace_user_task_0_injection_task_0",
                                                "workspace_user_task_0", True)]


# ------------------------------------------------------------ 1. consistency
def test_consistency_planted_shared_direction():
    ps = _pairs(20, lambda u, j, r: 3 * W0 + 0.5 * r.normal(size=D))
    r = dr.consistency(ps, 500, np.random.default_rng(0))
    assert r["logo_cos_mean"] > 0.8 and r["p_signflip"] < 0.01
    assert r["logo_cos_mean"] > r["null_p95"]
    assert r["null_mean"] < 0.5          # the MEAN statistic under the null (the median would sit near 1)


def test_consistency_sign_symmetric_deltas_are_not_consistent():
    """Each user task moves along +-W0 with a random sign: large, aligned with
    one axis, but no shared direction. The null does not reject, although the
    median leave-one-out cosine is large (the majority-sign artefact the mean
    avoids)."""
    sg = np.random.default_rng(21).choice([-1.0, 1.0], 20)
    ps = _pairs(20, lambda u, j, r: sg[u] * 3 * W0 + 0.3 * r.normal(size=D))
    r = dr.consistency(ps, 500, np.random.default_rng(0))
    assert r["p_signflip"] > 0.05
    assert r["logo_cos_median"] > r["logo_cos_mean"] + 0.3


def test_consistency_outcome_split_is_across_tasks():
    """Successes along W0 + the task's own direction, failures along W1 + the
    same own direction. Within a task they share the own component (cos ~0.5);
    across tasks they do not (cos ~0)."""
    rng0 = np.random.default_rng(3)
    own = {u: rng0.normal(size=D) * 3 / np.sqrt(D) * 3 for u in range(16)}
    ps = _pairs(16, lambda u, j, r: (3 * W0 if j == 0 else 3 * W1) + own[u],
                asr_fn=lambda u, j: j == 0)
    r = dr.consistency(ps, 50, np.random.default_rng(0))
    assert r["n_success"] == 16 and r["n_failure"] == 16
    assert abs(r["cos_success_vs_other_tasks_failure_mean"]) < 0.25


def test_consistency_null_uses_the_mean_statistic():
    """The p-value is the mean-statistic sign-flip null, reproduced draw by draw."""
    sg = np.random.default_rng(21).choice([-1.0, 1.0], 20)
    ps = _pairs(20, lambda u, j, r: sg[u] * 3 * W0 + 0.3 * r.normal(size=D))
    r = dr.consistency(ps, 300, np.random.default_rng(7))
    gm = dr.group_means(ps)
    M = np.vstack([gm[g] for g in sorted(gm)])
    rng = np.random.default_rng(7)
    null = [dr._logo_cos_from_gram(M @ M.T, rng.choice([-1.0, 1.0], len(M))) for _ in range(300)]
    assert r["p_signflip"] == pytest.approx(dr.pval(null, r["logo_cos_mean"]))
    assert r["null_p95"] == pytest.approx(np.percentile(null, 95))


def test_consistency_pair_cosines_leave_the_task_out():
    """No shared direction: each pair's cosine with the OTHER tasks' sum is ~0;
    including its own task would bias it upwards."""
    rng0 = np.random.default_rng(4)
    own = {u: rng0.normal(size=D) * 5 for u in range(60)}
    ps = _pairs(60, lambda u, j, r: own[u], reps=1, asr_fn=lambda u, j: u % 2 == 0)
    r = dr.consistency(ps, 50, np.random.default_rng(0))
    assert abs(r["pair_logo_cos_median_success"]) < 0.07
    assert abs(r["pair_logo_cos_median_failure"]) < 0.07


def test_consistency_degenerate_inputs():
    assert "note" in dr.consistency(_pairs(3, lambda u, j, r: W0), 10, np.random.default_rng(0))
    zero = [(t, g, c, c.copy(), None) for t, g, c, _, _ in _pairs(6, lambda u, j, r: W0)]
    assert "note" in dr.consistency(zero, 10, np.random.default_rng(0))


# ------------------------------------------------------------ 2. between attacks
def test_between_matched_vs_permuted():
    rng = np.random.default_rng(3)
    task = {u: rng.normal(size=D) * 3 for u in range(15)}
    pa = _pairs(15, lambda u, j, r: task[u] + 2 * W0, seed=4)
    pb = _pairs(15, lambda u, j, r: task[u] + 2 * W1, seed=4)
    r = dr.between(pa, pb, 500, np.random.default_rng(0))
    assert r["matched_cos_mean"] > r["mismatched_cos_median"] + 0.3
    assert r["ucos_mean_directions"] is None or abs(r["ucos_mean_directions"]) < 0.3   # W0 vs W1
    assert r["p_perm"] < 0.01
    ga, gb = dr.group_means(pa), dr.group_means(pb)
    G = sorted(ga)
    assert r["cos_mean_directions"] == pytest.approx(dr.cos(np.mean([ga[g] for g in G], 0),
                                                            np.mean([gb[g] for g in G], 0)))
    mism = [np.mean([dr.cos(ga[g], gb[h]) for h in G if h != g]) for g in G]
    assert r["mismatched_cos_median"] == pytest.approx(np.median(mism))
    assert r["matched_cos_mean"] == pytest.approx(np.mean([dr.cos(ga[g], gb[g]) for g in G]))


def test_between_drops_zero_group_means():
    pa = _pairs(8, lambda u, j, r: (0 * W0 if u == 0 else 2 * W0 + r.normal(size=D)), reps=1)
    pa[0] = (pa[0][0], pa[0][1], pa[0][2], pa[0][2] + 1e-9 * W0, None)       # reached, zero mean Delta
    pb = _pairs(8, lambda u, j, r: 2 * W1 + r.normal(size=D), reps=1, seed=2)
    assert dr.between(pa, pb, 20, np.random.default_rng(0))["n_groups"] == 7


def test_between_unrelated_attacks_do_not_reject():
    """Calibration: over 100 unrelated pairs of attacks, about 5% reject at 0.05."""
    rng = np.random.default_rng(5)
    ps = [dr.between(_pairs(15, lambda u, j, r: r.normal(size=D) * 3, seed=100 + i),
                     _pairs(15, lambda u, j, r: r.normal(size=D) * 3, seed=300 + i), 200, rng)["p_perm"]
          for i in range(100)]
    assert 0.0 <= np.mean(np.array(ps) < 0.05) <= 0.12 and np.mean(np.array(ps) < 0.5) > 0.35


# ------------------------------------------------------------ 3. detector
def test_detector_shared_direction_and_transfer():
    # 40 user tasks: 8 per fold, so a fold's sign-flip p can reach 1/256 (x5 folds)
    pa = _pairs(40, lambda u, j, r: 4 * W0 + 0.3 * r.normal(size=D))
    pb = _pairs(40, lambda u, j, r: 4 * W0 + 4 * W1 + 0.3 * r.normal(size=D))
    pc = _pairs(40, lambda u, j, r: 4 * W1 + 0.3 * r.normal(size=D))
    by = {"A": pa, "B": pb, "C": pc}
    negs, folds = _negs(pa), _folds(pa)
    rng = np.random.default_rng(0)
    aa = dr.detector(by, "A", "A", negs, folds, 2000, 20, rng)
    assert aa["auroc_vs_clean_same_step"] > 0.7 and aa["p_same_step"] < 0.05
    assert aa["auroc_vs_clean_all_steps"] > aa["random_dir_p95_all_steps"]
    assert aa["n_clean_all_steps"] == 120         # 3 per user task, each task held out once
    assert aa["n_clean_same_step"] == 40          # the two injection tasks share a clean state
    ab = dr.detector(by, "A", "B", negs, folds, 50, 20, rng)
    assert ab["auroc_vs_clean_same_step"] > 0.7                 # shares W0
    ac = dr.detector(by, "A", "C", negs, folds, 50, 20, rng)
    assert abs(ac["auroc_vs_clean_same_step"] - 0.5) < 0.15     # orthogonal: no transfer
    assert ac["auroc_vs_clean_same_step"] < ac["random_dir_p95_same_step"] + 0.05


def test_detector_holds_out_user_tasks():
    """Every user task has its own Delta direction (nothing shared). A direction
    that saw the test task would score it; held-out folds must not."""
    rng0 = np.random.default_rng(5)
    own = {u: rng0.normal(size=D) * 6 for u in range(25)}
    pa = _pairs(25, lambda u, j, r: own[u] + 0.05 * r.normal(size=D), seed=6)
    r = dr.detector({"A": pa}, "A", "A", _negs(pa), _folds(pa), 50, 20, np.random.default_rng(0))
    assert abs(r["auroc_vs_clean_same_step"] - 0.5) < 0.2
    assert r["p_same_step"] > 0.05


def test_detector_all_step_negatives_and_norm_baseline():
    """Clean states at other decisions far along the attack direction make the
    all-step AUROC fall while the same-step AUROC stays high; the ||h|| baseline
    is computed on the same states."""
    pa = _pairs(20, lambda u, j, r: 4 * W0 + 0.2 * r.normal(size=D))
    negs = {p[1]: [p[2] + 20 * W0 for _ in range(3)] for p in pa}
    r = dr.detector({"A": pa}, "A", "A", negs, _folds(pa), 20, 20, np.random.default_rng(0))
    assert r["auroc_vs_clean_same_step"] > 0.7
    assert r["auroc_vs_clean_all_steps"] < 0.3
    nrm = lambda X: np.linalg.norm(X, axis=1)
    exp = []
    for fg in _folds(pa):
        gs = dr._by_group(pa, fg)
        P, N = np.vstack([x for x, _ in gs]), np.vstack([y for _, y in gs])
        exp.append((dr.auroc(nrm(P), nrm(N)), len(P) * len(N)))
    assert r["norm_baseline_same_step"] == pytest.approx(dr._wavg(exp))
    assert r["norm_baseline_same_step"] != pytest.approx(0.5, abs=0.05)


def test_detector_test_is_calibrated():
    """No effect along any direction (attacked = clean + isotropic noise): the
    per-fold sign flip, combined by Bonferroni, rejects at most at the nominal
    rate. (Flipping the held-out d_g across all folds rejected ~12% at 0.05.)"""
    rng = np.random.default_rng(8)
    ps = []
    for i in range(400):
        pa = _pairs(15, lambda u, j, r: r.normal(size=D), seed=500 + i)
        ps.append(dr.detector({"A": pa}, "A", "A", {}, _folds(pa), 200, 0, rng)["p_same_step"])
    assert np.mean(np.array(ps) < 0.05) <= 0.06


def test_detector_fold_statistics():
    """shift_stat = sum over held-out tasks of <mean pair Delta, fold direction>;
    p = Bonferroni over the folds' p-values."""
    pa = _pairs(12, lambda u, j, r: 2 * W0 + r.normal(size=D))
    pb = _pairs(12, lambda u, j, r: 2 * W0 + W1 + r.normal(size=D), seed=3)
    folds = _folds(pa, 4)
    r = dr.detector({"A": pa, "B": pb}, "A", "B", {}, folds, 50, 0, np.random.default_rng(0))
    ma, eb = dr.group_means(pa), dr.group_means(pb)
    t = 0.0
    for fg in folds:
        w = np.mean([ma[h] for h in ma if h not in fg], axis=0)
        t += sum(float(eb[g] @ w) for g in eb if g in fg)
    assert r["shift_stat"] == pytest.approx(t)
    assert r["p_same_step"] is None                       # train != test
    rr = dr.detector({"A": pa}, "A", "A", {}, folds, 50, 0, np.random.default_rng(0))
    assert rr["p_same_step"] == pytest.approx(min(1.0, 4 * min(rr["p_folds"])))


def test_detector_fold_weighting():
    assert dr._wavg([(1.0, 1), (0.5, 3)]) == pytest.approx(0.625)
    assert dr._wavg([(None, 4), (0.5, 0)]) is None


def test_transfer_no_refit_and_null():
    pa = _pairs(12, lambda u, j, r: 20 * W0 + 0.3 * r.normal(size=D))
    qb = _pairs(12, lambda u, j, r: 20 * W0 + 0.3 * r.normal(size=D), seed=11, prefix="reddit")
    qc = _pairs(12, lambda u, j, r: 20 * W1 + 0.3 * r.normal(size=D), seed=12, prefix="reddit")
    r = dr.transfer({"A": pa}, {"B": qb, "C": qc}, _negs(qb), 50, 20, np.random.default_rng(0))
    assert r["A -> B"]["auroc_vs_clean_same_step"] > 0.9 and r["A -> B"]["p_same_step"] is None
    assert r["A -> B"]["ucos_directions"] > 0.9 and abs(r["A -> C"]["ucos_directions"]) < 0.2
    assert r["A -> B"]["n_clean_all_steps"] == 36
    assert abs(r["A -> C"]["auroc_vs_clean_same_step"] - 0.5) < 0.2


def test_transfer_dimension_mismatch_fails():
    pa = _pairs(6, lambda u, j, r: W0)
    qb = [(t, g, c[:10], a[:10], s) for t, g, c, a, s in _pairs(6, lambda u, j, r: W0)]
    with pytest.raises(SystemExit):
        dr.transfer({"A": pa}, {"B": qb}, {}, 5, 5, np.random.default_rng(0))


# ------------------------------------------------------------ 4. defences
def _def_states(n_users=12, blocked_users=(), seed=0, own_scale=0.0):
    """attack Delta = 3 W0 (+ own_scale * the task's own direction); defence 'x'
    footprint -W0 (against the attack), 'y' footprint +W1 (off-axis), 'z'
    footprint = the task's own direction; x and y both suppress along W2,
    measured from their own defended-clean state."""
    rng = np.random.default_rng(seed)
    st, out = {}, {}
    for u in range(n_users):
        c = rng.normal(size=D) * 5
        own = rng.normal(size=D)
        own[:3] = 0
        own /= np.linalg.norm(own)
        for j in range(2):
            t = f"workspace_user_task_{u}_injection_task_{j}"
            a = c + 3 * W0 + own_scale * own + 0.1 * rng.normal(size=D)
            s = {("clean", "A"): c, ("attacked", "A"): a}
            for dn, F in (("x", -W0), ("y", W1), ("z", 2 * own)):
                f = c + F + 0.02 * rng.normal(size=D)
                d = f + 0.5 * (a - c) + 2 * W2 + 0.02 * rng.normal(size=D)
                s[("def_clean", "A", dn)] = f
                s[("def_att", "A", dn)] = f if u in blocked_users else d
                if u in blocked_users:
                    s[("blocked", "A", dn)] = True
            st[t] = s
            out[t] = {("attacked", "A"): False}
    return st, out


def test_defences_footprint_and_suppression():
    st, oc = _def_states()
    ps = dr.pairs_of(st, oc, "A")
    r = dr.defences(st, "A", ps, 50, np.random.default_rng(0))
    assert r["per_defence"]["x"]["footprint_attack_ucos"] < -0.9
    st40, oc40 = _def_states(n_users=40)
    ps40 = dr.pairs_of(st40, oc40, "A")
    r40 = dr.defences(st40, "A", ps40, 50, np.random.default_rng(0))
    assert r40["per_defence"]["x"]["footprint_attack_ucos_ci95"][1] < -0.5
    assert abs(r["per_defence"]["y"]["footprint_attack_ucos"]) < 0.2
    assert r["per_defence"]["x"]["footprint_logo_cos_mean"] > 0.9
    e = r["defence_pairs"]["x | y"]
    assert abs(e["footprint_cos_median"]) < 0.2
    assert e["suppression_cos_median"] > 0.99                 # both 2 W2 from their own f
    assert e["suppression_from_clean_cos_median"] < 0.95     # y's footprint W1 mixes in from c


def test_defences_attack_direction_leaves_the_task_out():
    """No shared attack direction: each Delta is the task's own direction, and
    defence z's footprint follows it. The attack direction estimated WITHOUT the
    task is unrelated to z (cos ~0, not significant); the pair's own Delta is
    aligned with it."""
    st, oc = _def_states(n_users=40, own_scale=30.0)
    for s in st.values():                     # remove the shared 3 W0
        s[("attacked", "A")] = s[("attacked", "A")] - 3 * W0
    ps = dr.pairs_of(st, oc, "A")
    e = dr.defences(st, "A", ps, 50, np.random.default_rng(0))["per_defence"]["z"]
    # no shared attack direction: not interpretable (or, if an interval exists, it covers 0)
    ci = e["footprint_attack_ucos_ci95"]
    assert e["footprint_attack_ucos_interpretable"] is False
    assert ci is None or ci[0] < 0 < ci[1]
    assert e["footprint_cos_own_delta_median"] > 0.9


def test_ucos_paired_matches_brute_force():
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(9, D)) + W0, rng.normal(size=(9, D)) + W0 + W1
    u = lambda A, B: np.mean([A[g] @ B[h] for g in range(9) for h in range(9) if g != h])
    r = dr.ucos_paired(X, Y)
    assert r["ucos"] == pytest.approx(u(X, Y) / np.sqrt(u(X, X) * u(Y, Y)))
    loo = []
    for k in range(9):
        Xk, Yk = np.delete(X, k, 0), np.delete(Y, k, 0)
        uk = lambda A, B: np.mean([A[g] @ B[h] for g in range(8) for h in range(8) if g != h])
        loo.append(uk(Xk, Yk) / np.sqrt(uk(Xk, Xk) * uk(Yk, Yk)))
    se = np.sqrt(8 / 9 * np.sum((np.array(loo) - np.mean(loo)) ** 2))
    assert r["se"] == pytest.approx(se)


def test_ucos_flags_weak_directions_and_undefined_replicates():
    rng = np.random.default_rng(6)
    r = dr.ucos_paired(rng.normal(size=(13, D)) * 3, rng.normal(size=(13, D)) * 3)   # no shared direction
    assert r["interpretable"] is False
    strong = dr.ucos_paired(5 * W0 + rng.normal(size=(30, D)), 5 * W0 + rng.normal(size=(30, D)))
    assert strong["interpretable"] is True and strong["ucos"] > 0.9
    assert dr._jk_ci(0.5, [0.4, None, 0.6, 0.5]) == (None, None)
    r0 = np.random.default_rng(0)                       # an interval exists, but U_xx's does not exclude 0
    weak = dr.ucos_paired(0.3 * W0 + r0.normal(size=(13, D)), 5 * W0 + r0.normal(size=(13, D)))
    assert weak["ci95"] is not None and weak["u_xx_ci95"][0] < 0 and weak["interpretable"] is False


def test_ucos_indep_matches_brute_force():
    rng = np.random.default_rng(4)
    X, Y = rng.normal(size=(8, D)) + 2 * W0, rng.normal(size=(6, D)) + 2 * W0
    u = lambda A: np.mean([A[g] @ A[h] for g in range(len(A)) for h in range(len(A)) if g != h])
    assert dr.ucos_indep(X, Y)["ucos"] == pytest.approx(X.mean(0) @ Y.mean(0) / np.sqrt(u(X) * u(Y)))


def test_footprint_vs_attack_interval_covers_zero():
    """The attack shares W0; every footprint shares phi (orthogonal) and also
    follows its own task's Delta: the population cosine is 0 and the interval
    covers it about 95% of the time (a per-task test would reject often here)."""
    cover = []
    for i in range(200):
        r0 = np.random.default_rng(900 + i)
        M = 3 * W0 + r0.normal(size=(40, D))
        F = 3 * W2 + 0.8 * (M - 3 * W0) + r0.normal(size=(40, D))
        lo, hi = dr.ucos_paired(F, M)["ci95"]
        cover.append(lo < 0 < hi)
    assert np.mean(cover) >= 0.9


def test_detector_cross_attack_has_no_p():
    pa = _pairs(20, lambda u, j, r: 4 * W0 + r.normal(size=D))
    pc = _pairs(20, lambda u, j, r: 4 * W1 + r.normal(size=D))
    r = dr.detector({"A": pa, "C": pc}, "A", "C", {}, _folds(pa), 50, 5, np.random.default_rng(0))
    assert r["p_same_step"] is None and r["auroc_vs_clean_same_step"] is not None


def test_defences_blocked_pairs_have_no_suppression_vector():
    st, oc = _def_states(blocked_users=range(12))
    ps = dr.pairs_of(st, oc, "A")
    r = dr.defences(st, "A", ps, 50, np.random.default_rng(0))
    assert r["defence_pairs"]["x | y"]["n_groups_suppression"] == 0
    assert r["defence_pairs"]["x | y"]["n_groups_footprint"] == 12


# ------------------------------------------------------------ end to end
L = 31
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


def _run_dir(tmp, name="run", n_users=14, attacks=("A", "B"), seed=0):
    d = tmp / name
    d.mkdir()
    rows = []
    rng = np.random.default_rng(seed)
    comps = ["call_a", "call_b", "done"]
    junk = np.full(D, 7.0)

    def ep(sid, cond, atk, tid, dn, msgs, t1, asr=None):
        rows.append({"sample_id": sid, "condition": cond, "attack_name": atk, "task_id": tid,
                     "suite_name": "workspace", "defense_name": dn, "asr_success": asr, "error": None})
        _write_tr(d, sid, msgs, comps)
        torch.save([{L: torch.tensor(v, dtype=torch.float32)} for v in (junk, t1, junk)], d / f"{sid}.pt")

    for u in range(n_users):
        c = rng.normal(size=D) * 5
        for j in range(2):
            tid = f"workspace_user_task_{u}_injection_task_{j}"
            cm = [SYS, _tool(f"clean {u}"), _tool("next")]
            ep(f"c_{tid}", "clean", "clean", tid, None, cm, c)
            fm = [SYS, _tool(f"<<clean {u}>>"), _tool("next")]
            f = c - W0 + 0.05 * rng.normal(size=D)
            ep(f"f_{tid}", "defended_clean", "clean", tid, "dm", fm, f)
            for atk, w in zip(attacks, (W0, W1)):
                a = c + 15 * w + 0.3 * rng.normal(size=D) * (1 + u % 3)
                am = [SYS, _tool(f"clean {u} + PAYLOAD {atk} {j}"), _tool("next")]
                ep(f"{atk}_{tid}", "attacked", atk, tid, None, am, a, asr=bool(j))
                dam = [SYS, _tool(f"<<clean {u} + PAYLOAD {atk} {j}>>"), _tool("next")]
                ep(f"d{atk}_{tid}", "defended_attacked", atk, tid, "dm", dam, c + 7.5 * w + W1)
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def _homology(tmp, d, monkeypatch, name="hom.json", extra=()):
    import persistent_homology as ph
    hom = tmp / name
    monkeypatch.setattr(sys, "argv", ["x", "--results_dir", str(d), "--layer", str(L), "--pca_dim", "10",
                                      "--max_points", "150", "--n_perm", "5", *extra, "--out", str(hom)])
    ph.main()
    return hom


def _directions(d, out, hom, monkeypatch, extra=()):
    monkeypatch.setattr(sys, "argv", ["x", "--results_dir", str(d), "--layer", str(L), "--folds", "5",
                                      "--n_null", "50", "--n_null_detector", "20", "--n_random_dirs", "20", "--topology",
                                      "--n_swap", "5", "--homology_json", str(hom), "--out", str(out),
                                      *extra])
    dr.main()


def test_end_to_end_and_topology_matches_homology(tmp_path, monkeypatch):
    pytest.importorskip("ripser")
    import persistent_homology as ph
    d = _run_dir(tmp_path)
    d2 = _run_dir(tmp_path, name="other", seed=1)
    hom = _homology(tmp_path, d, monkeypatch)
    out = tmp_path / "dir.json"
    _directions(d, out, hom, monkeypatch, ("--transfer_dir", str(d2)))
    r = json.load(open(out))
    assert r["n_pairs"] == {"A": 28, "B": 28}
    assert r["n_clean_all_steps"] == {"groups": 14, "states": 42}
    assert r["consistency"]["A"]["logo_cos_mean"] > 0.5 and "p_holm" in r["consistency"]["A"]
    assert abs(r["between"]["A | B"]["cos_mean_directions"]) < 0.3
    assert r["detector"]["A -> A"]["auroc_vs_clean_same_step"] > 0.8
    assert r["detector"]["pooled -> B"]["auroc_vs_clean_same_step"] > 0.7
    import rq2_geometry as rq2
    det = [dict(v) for v in r["detector"].values()]
    rq2.holm(det, "p_same_step", "x")
    got = [v.get("p_holm_same_step") for v in r["detector"].values()]
    want = [v.get("x") for v in det]
    assert got == want and sum(x is not None for x in got) == 2          # A -> A and B -> B only
    assert r["transfer"]["results"]["A -> A"]["auroc_vs_clean_same_step"] > 0.8
    assert r["transfer"]["results"]["A -> B"]["auroc_vs_clean_same_step"] < 0.7
    assert r["defences"]["A"]["per_defence"]["dm"]["footprint_attack_ucos"] < -0.8
    topo = r["topology"]["A"]
    assert len(topo["pca2"]) == 14 and len(topo["curves"]["0"]["clean"]) == 50
    assert all(p["success_frac"] == 0.5 for p in topo["pca2"])
    h = json.load(open(hom))["per_attack"]
    for lab in ("A", "B"):
        for dim in ("0", "1"):
            assert r["topology"][lab]["curves"][dim]["observed_l2"] == pytest.approx(h[lab]["betti_l2"][dim]["observed"])
    # the band comes from within-task swaps (label A is first: the rng is fresh)
    cbt, _, att = ph._load_undefended(str(d), L, "arrival", False, {})
    C, A, groups = ph.matched_pairs(cbt, att)["A"]
    cz, oz, _ = ph.preprocess(C, A, 10, 0, scale_on="pooled")
    radii = np.array(topo["radius"])
    rng = np.random.default_rng(0)
    band = {0: [], 1: []}
    for _ in range(5):
        sw = rng.random(len(cz)) < 0.5
        for gg in (np.where(sw[:, None], oz, cz), np.where(sw[:, None], cz, oz)):
            dg = ph.diagrams(gg, 1)
            for k in (0, 1):
                band[k].append(ph.betti_curve(dg[k], radii))
    for k in (0, 1):
        assert topo["curves"][str(k)]["swap_hi"] == pytest.approx(np.percentile(band[k], 97.5, axis=0).tolist())
        assert topo["curves"][str(k)]["swap_lo"] == pytest.approx(np.percentile(band[k], 2.5, axis=0).tolist())
    # the map is the same matched clouds, clean first
    c2, o2, _ = ph.preprocess(C, A, 2, 0, scale_on="pooled")
    assert topo["pca2"][0]["clean"] == pytest.approx(c2[0].tolist())
    assert topo["pca2"][0]["attacked"] == pytest.approx(o2[0].tolist())


def _topology_fails(tmp_path, monkeypatch, mutate):
    pytest.importorskip("ripser")
    d = _run_dir(tmp_path)
    hom = _homology(tmp_path, d, monkeypatch)
    j = json.load(open(hom))
    mutate(j)
    hom.write_text(json.dumps(j))
    out = tmp_path / "dir.json"
    with pytest.raises(SystemExit):
        _directions(d, out, hom, monkeypatch)
    assert not out.exists()


def test_topology_l2_mismatch_fails(tmp_path, monkeypatch):
    _topology_fails(tmp_path, monkeypatch,
                    lambda j: j["per_attack"]["A"]["betti_l2"]["0"].update(observed=12345.0))


def test_topology_missing_label_fails(tmp_path, monkeypatch):
    _topology_fails(tmp_path, monkeypatch, lambda j: j["per_attack"].pop("B"))


def test_topology_extra_label_in_homology_fails(tmp_path, monkeypatch):
    _topology_fails(tmp_path, monkeypatch,
                    lambda j: j["per_attack"].__setitem__("Z", j["per_attack"]["A"]))


def test_topology_missing_dim_fails(tmp_path, monkeypatch):
    _topology_fails(tmp_path, monkeypatch, lambda j: j["per_attack"]["A"]["betti_l2"].pop("1"))


def test_topology_swapped_clouds_fail(tmp_path, monkeypatch):
    def swap(j):
        s = j["per_attack"]["A"]["summaries"]
        s["clean"], s["other"] = s["other"], s["clean"]
    _topology_fails(tmp_path, monkeypatch, swap)


def test_topology_other_settings_fail(tmp_path, monkeypatch):
    _topology_fails(tmp_path, monkeypatch, lambda j: j["config"].update(layer=7))


def test_topology_other_run_fails(tmp_path, monkeypatch):
    _topology_fails(tmp_path, monkeypatch, lambda j: j["config"].update(results_dir="final_other"))


def test_topology_without_config_fails(tmp_path, monkeypatch):
    _topology_fails(tmp_path, monkeypatch, lambda j: j.pop("config"))


def test_topology_needs_homology_json(tmp_path, monkeypatch):
    d = _run_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--results_dir", str(d), "--layer", str(L), "--topology",
                                      "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        dr.main()


def test_empty_transfer_fails(tmp_path, monkeypatch):
    """A VWA-style run read without --transfer_accept_step0 has no pairs: fail."""
    d = _run_dir(tmp_path)
    v = tmp_path / "vwa"
    v.mkdir()
    rows = []
    for u in range(6):
        tid = f"reddit_{u}_x_cap_1"
        for sid, cond, atk, txt, vec in ((f"c{u}", "clean", "clean", "page", np.zeros(D)),
                                         (f"a{u}", "attacked", "A", "page PAYLOAD", W0)):
            rows.append({"sample_id": sid, "condition": cond, "attack_name": atk, "task_id": tid,
                         "defense_name": None, "asr_success": False, "error": None})
            _write_tr(v, sid, [_tool(txt)], ["act"])
            torch.save([{L: torch.tensor(vec, dtype=torch.float32)}], v / f"{sid}.pt")
    (v / "attack_results.json").write_text(json.dumps(rows))
    monkeypatch.setattr(sys, "argv", ["x", "--results_dir", str(d), "--layer", str(L), "--n_null", "10",
                                      "--n_null_detector", "5", "--n_random_dirs", "5", "--transfer_dir", str(v),
                                      "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        dr.main()
    assert not (tmp_path / "o.json").exists()
