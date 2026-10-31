#!/usr/bin/env python3
"""analysis/directions.py -- is there a canonical injection direction? CPU only,
read-only on the run directories.

Uses the RQ2 arrival-step states (rq2_geometry.load_multi(turn='arrival')), so
every state is the same decision as the RQ1 and RQ2 tables: clean c and
attacked a at the payload's arrival step; defended-clean f and defended-attacked
d at the same step of the same task. Delta = a - c over reached pairs. Every
group statistic is computed on per-user-task means (injection-task repeats of
one user task are not independent), and every direction used on a user task is
estimated without that user task. p-values are Holm-corrected within each
family (one family per numbered section and run; the detector's same-step and
all-step p-values are separate families).

  1. consistency  -- per attack: cosine of each user task's mean Delta with the
     summed Delta of all OTHER user tasks (leave-one-group-out, LOGO); the mean
     over user tasks is the statistic. Null: random sign per user task (exact
     under "user-task Deltas are independent and sign-symmetric"). The cosine
     between successes and failures is taken across user tasks (a task's
     successes against the other tasks' failures), so a task's own component
     shared by its successes and failures does not inflate it.
  2. between attacks -- per user task reached by both: cosine of the two
     attacks' Deltas (matched); null: the other attack's user tasks permuted
     (exact under exchangeability of user tasks). Also the population cosine of
     the two attacks' mean Deltas (U-statistics, jackknife interval), which a
     shared direction of either attack alone does not bias.
  3. detector -- the cheapest monitor: score(h) = <h, w>, w = mean Delta of the
     training attack (one dot product per decision, no clean reference at test
     time). Grouped K-fold over user tasks, one fold assignment per run. AUROC
     (per fold, weighted by n_pos x n_neg) of attacked arrival states against
     (i) the clean states at the same step, with a test of the attacked-minus-
     clean shift along w (per-fold sign flip, folds combined by Bonferroni;
     see detector()), and (ii) the clean states
     at EVERY decision of the held-out user tasks (what a deployed monitor
     sees; unpaired, so reported against random unit directions, not tested).
     Baseline: the norm ||h||. With
     --transfer_dir every attack's direction (all user tasks) is applied,
     unchanged, to another run of the same model and layer (AgentDojo -> VWA).
  4. defences -- per defence: footprint F = f - c; the population cosine of
     the mean footprint and the attack's mean Delta (U-statistics that leave
     out same-task products; jackknife interval), its cosine with the
     pair's own Delta; whether the footprint has a consistent direction (as 1);
     per pair of defences on the same task, the cosine of their footprints and
     of their off-axis suppression displacements, measured from each defence's
     own defended-clean state, s = (d - f) - <d - f, u> u (u = Delta/||Delta||;
     blocked pairs excluded), and, for comparison, from c.
  5. --topology (requires --homology_json): figure data for RQ1's homology: the
     matched clean/attacked clouds exactly as persistent_homology.main builds
     them (its own settings, read from its JSON), their H0/H1 Betti curves, a
     band from the within-task swap null, and a 2-D PCA map. The labels, the
     homology dimensions, the observed Betti L2 and the per-cloud diagram
     summaries must all reproduce the homology JSON, or nothing is written.

  python3 analysis/directions.py --results_dir final_agentdojo_gemma4 --layer 42 \
      --transfer_dir final_vwa_gemma4 --transfer_accept_step0 --topology \
      --homology_json analysis_out/homology_agentdojo_gemma4.json \
      --out analysis_out/directions_agentdojo_gemma4.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq2_geometry as rq2  # noqa: E402
from geometry import user_task_group, load_turns  # noqa: E402

try:
    from scipy import stats as sps
except ImportError:
    sps = None

TOL = rq2.ZERO_TOL
MIN_GROUPS = 4


def unit(x):
    n = float(np.linalg.norm(x))
    return x / n if n > TOL else None


def cos(x, y):
    if x is None or y is None:
        return None
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    return float(np.dot(x, y) / (nx * ny)) if nx > TOL and ny > TOL else None


def _avg_ranks(v):
    if sps is not None:
        return sps.rankdata(v)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), by average ranks."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    ranks = _avg_ranks(np.concatenate([pos, neg]))
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def pval(null, obs):
    null = np.asarray([v for v in null if v is not None and np.isfinite(v)], float)
    if obs is None or not len(null):
        return None
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def wilcoxon_vs0(x):
    x = np.asarray([v for v in x if v is not None], float)
    if sps is None or len(x) < 6 or not np.any(x):
        return None
    return float(sps.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)


def _gmean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------- data
def attacks_in(states):
    return sorted({k[1] for s in states.values() for k in s
                   if isinstance(k, tuple) and k[0] == "attacked"})


def pairs_of(states, outcomes, attack):
    """[(task_id, group, c, a, asr)] for reached pairs of one attack."""
    out = []
    for t, s in states.items():
        c, a = s.get(("clean", attack)), s.get(("attacked", attack))
        if c is None or a is None or np.linalg.norm(a - c) <= TOL:
            continue
        out.append((t, user_task_group(t), np.asarray(c, np.float64), np.asarray(a, np.float64),
                    outcomes.get(t, {}).get(("attacked", attack))))
    return out


def group_means(pairs):
    """{group: mean Delta}"""
    by = defaultdict(list)
    for _, g, c, a, _ in pairs:
        by[g].append(a - c)
    return {g: np.mean(v, axis=0) for g, v in by.items()}


def logo_directions(gm):
    """{group: sum of the OTHER groups' mean Deltas} (direction only matters)."""
    if not gm:
        return {}
    S = np.sum(list(gm.values()), axis=0)
    return {g: S - v for g, v in gm.items()}


def clean_all_steps(results_dir, rows, layer, groups=None, max_steps=400):
    """{group: [clean state at every decision]} from ONE clean episode per user
    task (repeats of a user task's clean episode are the same input)."""
    pick = {}
    for r in rows:
        if r.get("condition") == "clean" and not r.get("error") and r.get("sample_id"):
            g = user_task_group(r["task_id"])
            if (groups is None or g in groups) and (g not in pick or r["sample_id"] < pick[g]):
                pick[g] = r["sample_id"]
    vec = load_turns(results_dir, {sid: set(range(max_steps)) for sid in pick.values()}, [layer])
    out = defaultdict(list)
    for g, sid in pick.items():
        for t in range(max_steps):
            v = vec.get((sid, t), {}).get(layer)
            if v is not None and np.all(np.isfinite(v)):
                out[g].append(np.asarray(v, np.float64))
    return dict(out)


# ---------------------------------------------------------------- 1. consistency
def _logo_cos_from_gram(C, s, stat=np.mean):
    """stat (mean: the test statistic) over g of cos(s_g D_g, sum_{h!=g} s_h D_h),
    from the Gram matrix C. The mean, not the median, is tested: under random
    signs the median of the leave-one-out cosines is pulled towards +1 by the
    majority sign (every task in the majority agrees with the rest), the mean
    is not."""
    v = C @ s
    q = float(s @ v)
    diag = np.diag(C)
    num = s * (v - s * diag)
    den2 = q - 2 * s * v + diag
    den = np.sqrt(np.clip(diag, 0, None)) * np.sqrt(np.clip(den2, 0, None))
    ok = den > TOL
    return float(stat(num[ok] / den[ok])) if ok.any() else None


def consistency(pairs, n_null, rng):
    gm = group_means(pairs)
    G = sorted(gm)
    if len(G) < MIN_GROUPS:
        return {"n_groups": len(G), "note": "too few user tasks"}
    M = np.vstack([gm[g] for g in G])
    C = M @ M.T
    ones = np.ones(len(G))
    obs = _logo_cos_from_gram(C, ones)
    if obs is None:
        return {"n_groups": len(G), "note": "all user-task Deltas zero"}
    obs_med = _logo_cos_from_gram(C, ones, np.median)
    S = M.sum(0)
    per_group = [x for x in (cos(M[i], S - M[i]) for i in range(len(G))) if x is not None]
    null = [_logo_cos_from_gram(C, rng.choice([-1.0, 1.0], len(G))) for _ in range(n_null)]
    nn = np.array([x for x in null if x is not None], float)
    gi = {g: i for i, g in enumerate(G)}
    # pair level, by outcome, against the leave-one-group-out direction
    by_out = {"success": [], "failure": []}
    for _, g, c, a, asr in pairs:
        if asr is not None:
            by_out["success" if asr else "failure"].append(cos(a - c, S - M[gi[g]]))
    # success vs failure, across user tasks: a task's successes against the
    # OTHER tasks' failures (its own task component would inflate a within-task cosine)
    gs = group_means([p for p in pairs if p[4] is True])
    gf = group_means([p for p in pairs if p[4] is False])
    Sf = np.sum(list(gf.values()), axis=0) if gf else None
    cross = [cos(gs[g], Sf - gf.get(g, 0.0)) for g in gs] if gf else []
    return {"n_groups": len(G), "n_pairs": len(pairs),
            "logo_cos_mean": obs, "logo_cos_median": obs_med,
            "logo_cos_q1": float(np.percentile(per_group, 25)) if per_group else None,
            "logo_cos_q3": float(np.percentile(per_group, 75)) if per_group else None,
            "null_mean": float(np.mean(nn)) if len(nn) else None,
            "null_p95": float(np.percentile(nn, 95)) if len(nn) else None,
            "p_signflip": pval(nn, obs),
            "pair_logo_cos_median_success": med(by_out["success"]),
            "pair_logo_cos_median_failure": med(by_out["failure"]),
            "n_success": len(by_out["success"]), "n_failure": len(by_out["failure"]),
            "n_groups_success": len(gs), "n_groups_failure": len(gf),
            "cos_success_vs_other_tasks_failure_mean": _gmean(cross)}


# ---------------------------------------------------------------- population cosines
def _jk_ci(est, loo):
    """Jackknife SE and Wald 95% interval; none if ANY leave-one-out replicate is
    undefined (dropping them would shrink the SE exactly when it is unstable)."""
    if est is None or len(loo) < 3 or any(x is None or not np.isfinite(x) for x in loo):
        return None, None
    loo = np.array(loo, float)
    n = len(loo)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return se, [est - 1.96 * se, est + 1.96 * se]


def ucos_paired(X, Y):
    """Cosine of the population means of two per-task vectors (rows aligned by
    user task), from U-statistics that leave out same-task products:
        U_xy = mean_{g != h} <x_g, y_h>   (unbiased for <mu_x, mu_y>),
        cos = U_xy / sqrt(U_xx U_yy),
    so a task's own coupling of x and y (a footprint following its own task's
    Delta) does not enter, and shared directions of x or y alone do not bias
    it. Jackknife over user tasks for a 95% interval. None if U_xx or U_yy <= 0
    (no detectable shared direction)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    n = len(X)
    if n < MIN_GROUPS:
        return {"n": n, "ucos": None, "ci95": None}
    G = {k: A @ B.T for k, (A, B) in {"xy": (X, Y), "xx": (X, X), "yy": (Y, Y)}.items()}
    tot = {k: float(g.sum()) for k, g in G.items()}
    tr = {k: float(np.trace(g)) for k, g in G.items()}
    rs = {k: g.sum(1) for k, g in G.items()}
    cs = {k: g.sum(0) for k, g in G.items()}
    dg = {k: np.diag(g) for k, g in G.items()}

    def est(drop=None):
        m = n if drop is None else n - 1
        u = {}
        for k in G:
            t, d = tot[k], tr[k]
            if drop is not None:
                t = t - rs[k][drop] - cs[k][drop] + dg[k][drop]
                d = d - dg[k][drop]
            u[k] = (t - d) / (m * (m - 1))
        if u["xx"] <= 0 or u["yy"] <= 0:
            return None, u
        return float(u["xy"] / np.sqrt(u["xx"] * u["yy"])), u
    c, u = est()
    loo = [est(k) for k in range(n)]
    se, ci = _jk_ci(c, [x[0] for x in loo])
    _, cxx = _jk_ci(u["xx"], [x[1]["xx"] for x in loo])
    _, cyy = _jk_ci(u["yy"], [x[1]["yy"] for x in loo])
    # interpret only when both mean directions are clearly present and the ratio is a cosine
    ok = bool(ci and cxx and cyy and cxx[0] > 0 and cyy[0] > 0 and abs(c) <= 1)
    return {"n": n, "ucos": c, "ci95": ci, "se": se, "u_xy": u["xy"], "u_xx": u["xx"], "u_yy": u["yy"],
            "u_xx_ci95": cxx, "u_yy_ci95": cyy, "interpretable": ok}


def ucos_indep(X, Y):
    """As ucos_paired for two INDEPENDENT samples of tasks (e.g. AgentDojo and
    VWA-Adv): U_xy = <mean x, mean y> is already unbiased; U_xx, U_yy leave out
    same-task products. Jackknife over both samples (variances added)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if len(X) < MIN_GROUPS or len(Y) < MIN_GROUPS:
        return {"ucos": None, "ci95": None}

    def uself(A):
        g = A @ A.T
        return (g.sum() - np.trace(g)) / (len(A) * (len(A) - 1))

    def est(A, B):
        ua, ub = uself(A), uself(B)
        if ua <= 0 or ub <= 0:
            return None
        return float(A.mean(0) @ B.mean(0) / np.sqrt(ua * ub))
    c = est(X, Y)
    _, cxx = _jk_ci(uself(X), [uself(np.delete(X, k, 0)) for k in range(len(X))])
    _, cyy = _jk_ci(uself(Y), [uself(np.delete(Y, k, 0)) for k in range(len(Y))])
    sx, _ = _jk_ci(c, [est(np.delete(X, k, 0), Y) for k in range(len(X))])
    sy, _ = _jk_ci(c, [est(X, np.delete(Y, k, 0)) for k in range(len(Y))])
    if c is None or sx is None or sy is None:
        return {"ucos": c, "ci95": None, "interpretable": False}
    se = float(np.sqrt(sx ** 2 + sy ** 2))
    ok = bool(cxx and cyy and cxx[0] > 0 and cyy[0] > 0 and abs(c) <= 1)
    return {"ucos": c, "ci95": [c - 1.96 * se, c + 1.96 * se], "se": se, "n_x": len(X), "n_y": len(Y),
            "u_xx_ci95": cxx, "u_yy_ci95": cyy, "interpretable": ok}


# ---------------------------------------------------------------- 2. between attacks
def between(pairs_a, pairs_b, n_null, rng):
    ga, gb = group_means(pairs_a), group_means(pairs_b)
    G = sorted(g for g in set(ga) & set(gb) if unit(ga[g]) is not None and unit(gb[g]) is not None)
    if len(G) < MIN_GROUPS:
        return {"n_groups": len(G), "note": "too few shared user tasks"}
    A = np.vstack([unit(ga[g]) for g in G])
    B = np.vstack([unit(gb[g]) for g in G])
    Cab = A @ B.T                                     # cos(ga[g], gb[h])
    obs = float(np.mean(np.diag(Cab)))
    null = [float(np.mean(Cab[np.arange(len(G)), rng.permutation(len(G))])) for _ in range(n_null)]
    mism = [float(np.mean(np.delete(Cab[i], i))) for i in range(len(G))]
    uc = ucos_paired(np.vstack([ga[g] for g in G]), np.vstack([gb[g] for g in G]))
    return {"n_groups": len(G),
            "cos_mean_directions": cos(np.mean([ga[g] for g in G], 0), np.mean([gb[g] for g in G], 0)),
            "ucos_mean_directions": uc["ucos"], "ucos_ci95": uc["ci95"],
            "ucos_interpretable": uc.get("interpretable", False),
            "matched_cos_mean": obs, "matched_cos_median": float(np.median(np.diag(Cab))),
            "mismatched_cos_median": med(mism),
            "null_p95": float(np.percentile(null, 95)) if null else None,
            "p_perm": pval(null, obs)}


# ---------------------------------------------------------------- 3. detector
def make_folds(groups, k, rng):
    G = sorted(groups)
    rng.shuffle(G)
    return [set(G[i::k]) for i in range(k)]


def _dedup(vs):
    seen, out = set(), []
    for v in vs:
        key = np.asarray(v).tobytes()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _wavg(vals):
    """vals: [(auroc, weight)] -> weighted mean (None if empty)."""
    vals = [(a, w) for a, w in vals if a is not None and w > 0]
    if not vals:
        return None
    return float(sum(a * w for a, w in vals) / sum(w for _, w in vals))


def _fold_auroc(items, X_all_s=None):
    """items: [(scores_pos_g, scores_neg_g)] per held-out user task of one fold
    -> (AUROC of the pooled fold, weight n_pos x n_neg)."""
    sp = np.concatenate([p for p, _ in items])
    sn = np.concatenate([n for _, n in items]) if X_all_s is None else X_all_s
    return auroc(sp, sn), len(sp) * len(sn)


def _evaluate(folds_data, t_obs, null, n_rand, rng):
    """folds_data: per fold (w, [(X_pos_g, X_neg_g)], X_all or None).
    The AUROC is the reported effect; t_obs/null the same-step test (see the
    callers). All steps: no pairing, so a reference only: the 95th percentile
    of random unit directions. Baseline: ||h||."""
    proj = []
    for w, groups, X_all in folds_data:
        proj.append(([(Xp @ w, Xn @ w) for Xp, Xn in groups], None if X_all is None else X_all @ w))
    same = _wavg([_fold_auroc(g) for g, _ in proj])
    allst = _wavg([_fold_auroc(g, a) for g, a in proj if a is not None])
    rand_s, rand_a = [], []
    d = folds_data[0][0].shape[0]
    for _ in range(n_rand):
        r = rng.normal(size=d)
        r /= np.linalg.norm(r)
        rand_s.append(_wavg([_fold_auroc([(Xp @ r, Xn @ r) for Xp, Xn in gs]) for _, gs, _ in folds_data]))
        rand_a.append(_wavg([_fold_auroc([(Xp @ r, Xn @ r) for Xp, Xn in gs], Xa @ r)
                             for _, gs, Xa in folds_data if Xa is not None]))
    nrm = lambda X: np.linalg.norm(X, axis=1)
    base_s = _wavg([_fold_auroc([(nrm(Xp), nrm(Xn)) for Xp, Xn in gs]) for _, gs, _ in folds_data])
    base_a = _wavg([_fold_auroc([(nrm(Xp), nrm(Xn)) for Xp, Xn in gs], nrm(Xa))
                    for _, gs, Xa in folds_data if Xa is not None])
    ra = [x for x in rand_a if x is not None]
    return {"auroc_vs_clean_same_step": same, "auroc_vs_clean_all_steps": allst,
            "norm_baseline_same_step": base_s, "norm_baseline_all_steps": base_a,
            "shift_stat": t_obs, "p_same_step": pval(null, t_obs),
            "random_dir_p95_same_step": float(np.percentile(rand_s, 95)) if rand_s else None,
            "random_dir_p95_all_steps": float(np.percentile(ra, 95)) if ra else None}


def _by_group(pairs, groups):
    """[(X_pos_g, X_neg_g)] for the user tasks in `groups` (AUROC only); the
    clean state of a task's injection-task repeats is the same input, so it is
    counted once among the negatives."""
    out = []
    for g in sorted(groups):
        tp = [p for p in pairs if p[1] == g]
        if tp:
            out.append((np.vstack([p[3] for p in tp]), np.vstack(_dedup([p[2] for p in tp]))))
    return out


def detector(pairs_by_attack, train, test, negs_all, folds, n_null, n_rand, rng):
    """Grouped-CV AUROC of <h, w>, w = mean (per user task) Delta of `train`
    ('pooled' = every attack) on the training groups; `folds` fixed per run.

    Same-step test, per fold: given the fold's training data, its held-out
    tasks are independent of w_f, so d_g = <e_g, w_f> (e_g = mean over the
    task's pairs of a - c, which keeps the pairing) can be sign-flipped per
    task: exact under "the test attack's shift has no component along w_f".
    Across folds the d_g are NOT independent (a task's Deltas are in the other
    folds' directions; flipping only d_g over all folds rejected ~12% at 0.05,
    and a joint flip of training and test Deltas tests a different, stronger
    null), so the folds' p-values are combined by Bonferroni: p = min(1, k min p_f)."""
    tr = [p for ps in pairs_by_attack.values() for p in ps] if train == "pooled" else pairs_by_attack[train]
    te = pairs_by_attack[test]
    if len({p[1] for p in te}) < len(folds) or len({p[1] for p in tr}) < MIN_GROUPS:
        return {"note": "too few user tasks"}
    m_tr, e_te = group_means(tr), group_means(te)
    data, p_f, dg_all, t_all = [], [], [], 0.0
    for fg in folds:
        train_g = [h for h in m_tr if h not in fg]
        test_g = sorted(g for g in e_te if g in fg)
        if len(train_g) < MIN_GROUPS or not test_g:
            continue
        w = np.mean([m_tr[h] for h in train_g], axis=0)
        d = np.array([float(e_te[g] @ w) for g in test_g])
        null = rng.choice([-1.0, 1.0], (n_null, len(d))) @ d
        p_f.append(pval(null, float(d.sum())))
        dg_all += list(d)
        t_all += float(d.sum())
        alln = [v for g in sorted(fg) for v in negs_all.get(g, [])]
        data.append((w, _by_group(te, fg), np.vstack(alln) if alln else None))
    if not data:
        return {"note": "no usable fold"}
    r = _evaluate(data, None, [], n_rand, rng)
    ok = [x for x in p_f if x is not None]
    r.update({"shift_stat": t_all, "p_folds": p_f,
              # valid against "the test attack has no mean shift" only when the
              # direction is learnt from the same attack (if mu = 0 the held-out
              # Deltas are symmetric whatever w is); across attacks, w's
              # estimation noise lets a test attack with ANY shared direction
              # shift along w, so no p-value is given (AUROC vs random directions
              # and the population cosine in 2. carry that comparison)
              "p_same_step": (min(1.0, len(data) * min(ok)) if ok else None) if train == test else None,
              "n_folds": len(data), "n_tasks_tested": len(dg_all),
              "frac_tasks_shifted": float(np.mean(np.array(dg_all) > 0)),
              "n_test_pairs": int(sum(len(Xp) for _, gs, _ in data for Xp, _ in gs)),
              "n_clean_same_step": int(sum(len(Xn) for _, gs, _ in data for _, Xn in gs)),
              "n_clean_all_steps": int(sum(len(Xa) for _, _, Xa in data if Xa is not None))})
    return r


def transfer(pairs_by_attack, other_pairs_by_attack, other_negs_all, n_null, n_rand, rng):
    """Direction of each attack here (all user tasks), applied unchanged to
    another run of the same model and layer: AUROC against random directions
    (no p-value: w's estimation noise lets any shared direction of the target
    attack shift along w), and the population cosine of the two attacks' mean
    Deltas (ucos_indep) with a jackknife interval."""
    out = {}
    negs = [v for g in sorted(other_negs_all) for v in other_negs_all[g]]
    X_all = np.vstack(negs) if negs else None
    for src, ps in sorted(pairs_by_attack.items()):
        gm = group_means(ps)
        if len(gm) < MIN_GROUPS:
            continue
        w = np.mean(list(gm.values()), axis=0)
        for dst, qs in sorted(other_pairs_by_attack.items()):
            e = group_means(qs)
            if len(e) < MIN_GROUPS:
                continue
            if next(iter(e.values())).shape[0] != w.shape[0]:
                sys.exit(f"FATAL: transfer dimension {next(iter(e.values())).shape[0]} != {w.shape[0]} "
                         "(different model or layer?)")
            dg = np.array([float(e[g] @ w) for g in sorted(e)])
            groups = _by_group(qs, set(e))
            r = _evaluate([(w, groups, X_all)], float(dg.sum()), [], n_rand, rng)
            uc = ucos_indep(np.vstack(list(gm.values())), np.vstack([e[g] for g in sorted(e)]))
            r.update({"p_same_step": None, "ucos_directions": uc["ucos"], "ucos_ci95": uc["ci95"],
                      "ucos_interpretable": uc.get("interpretable", False)})
            r.update({"n_tasks_tested": len(dg), "frac_tasks_shifted": float(np.mean(dg > 0)),
                      "n_pos": int(sum(len(Xp) for Xp, _ in groups)),
                      "n_clean_same_step": int(sum(len(Xn) for _, Xn in groups)),
                      "n_clean_all_steps": 0 if X_all is None else len(X_all)})
            out[f"{src} -> {dst}"] = r
    return out


# ---------------------------------------------------------------- 4. defences
def _off_axis(x, u):
    return x - np.dot(x, u) * u


def _footprint_vs_attack(Fg, gm):
    """Population cosine of the defence's mean footprint and the attack's mean
    Delta (ucos_paired over the user tasks that have both): footprints that
    follow their own task's Delta, or share a direction of their own, do not
    bias it."""
    G = sorted(g for g in Fg if g in gm)
    if len(G) < MIN_GROUPS:
        return {}
    return ucos_paired(np.vstack([Fg[g] for g in G]), np.vstack([gm[g] for g in G]))


def defences(states, attack, pairs, n_null, rng):
    """Footprint vs attack direction; footprint consistency; defence vs defence."""
    gm = group_means(pairs)
    if len(gm) < MIN_GROUPS:
        return {}
    reached = sorted({p[0] for p in pairs})
    dns = sorted({k[2] for s in states.values() for k in s
                  if isinstance(k, tuple) and k[0] == "def_att" and k[1] == attack})
    per = {}                      # dn -> {task: (group, F, sup_from_f, sup_from_c, Delta)}
    for dn in dns:
        per[dn] = {}
        for t in reached:
            s = states[t]
            c, a = s[("clean", attack)], s[("attacked", attack)]
            f, d = s.get(("def_clean", attack, dn)), s.get(("def_att", attack, dn))
            if f is None or d is None:
                continue
            c, a, f, d = (np.asarray(x, np.float64) for x in (c, a, f, d))
            u = unit(a - c)
            blocked = bool(s.get(("blocked", attack, dn)))
            per[dn][t] = (user_task_group(t), f - c,
                          None if blocked else _off_axis(d - f, u),
                          None if blocked else _off_axis(d - c, u), a - c)
    out = {"per_defence": {}, "defence_pairs": {}}
    for dn, m in per.items():
        byF, byD = defaultdict(list), defaultdict(list)
        for g, F, _, _, D in m.values():
            byF[g].append(F)
            byD[g].append(cos(F, D))
        Fg = {g: np.mean(v, axis=0) for g, v in byF.items() if np.linalg.norm(np.mean(v, axis=0)) > TOL}
        stat = _footprint_vs_attack(Fg, gm)
        fp = [(t, g, np.zeros_like(F), F, None) for t, (g, F, _, _, _) in m.items()
              if np.linalg.norm(F) > TOL]
        fc = consistency(fp, n_null, rng) if len({x[1] for x in fp}) >= MIN_GROUPS else {}
        out["per_defence"][dn] = {
            "n_pairs": len(m), "n_groups": len(byF),
            "footprint_attack_ucos": stat.get("ucos"), "footprint_attack_ucos_ci95": stat.get("ci95"),
            "footprint_attack_ucos_interpretable": stat.get("interpretable", False),
            # descriptive only: cos(F_g, own Delta_g) is invariant to the task's sign
            "footprint_cos_own_delta_median": med([_gmean(v) for v in byD.values()]),
            "footprint_logo_cos_mean": fc.get("logo_cos_mean"),
            "footprint_p_signflip": fc.get("p_signflip")}
    for i, d1 in enumerate(dns):
        for d2 in dns[i + 1:]:
            byf, bys, byc = defaultdict(list), defaultdict(list), defaultdict(list)
            for t in set(per[d1]) & set(per[d2]):
                g, F1, s1, c1, _ = per[d1][t]
                _, F2, s2, c2, _ = per[d2][t]
                byf[g].append(cos(F1, F2))
                bys[g].append(cos(s1, s2))
                byc[g].append(cos(c1, c2))
            agg = lambda by: [x for x in (_gmean(v) for v in by.values()) if x is not None]
            gf, gs, gcl = agg(byf), agg(bys), agg(byc)
            out["defence_pairs"][f"{d1} | {d2}"] = {
                "n_groups_footprint": len(gf), "footprint_cos_median": med(gf),
                "n_groups_suppression": len(gs), "suppression_cos_median": med(gs),
                "suppression_from_clean_cos_median": med(gcl)}
    return out


# ---------------------------------------------------------------- 5. topology figure data
def topology(results_dir, layer, accept_step0, pairs, n_swap, seed, pca_dim, grid, max_points, min_points):
    import persistent_homology as ph
    if not ph.HAVE_RIPSER:
        sys.exit("FATAL: ripser unavailable -- H1 curves cannot be computed")
    info = {}
    cbt, _, att = ph._load_undefended(results_dir, layer, "arrival", accept_step0, info)
    matched = ph.matched_pairs(cbt, att)
    rng = np.random.default_rng(seed)
    out = {}
    for label in sorted(matched):
        C, A, groups = matched[label]
        if len(groups) < min_points:
            continue
        if len(groups) > max_points:
            # persistent_homology.main subsamples with an rng it also uses for its
            # nulls; the cloud cannot be rebuilt identically here
            sys.exit(f"FATAL: {label}: {len(groups)} pairs > max_points {max_points}: the homology "
                     "run subsampled, so its clouds cannot be reproduced")
        cz, oz, _ = ph.preprocess(C, A, pca_dim, seed, scale_on="pooled")
        radii = ph._radius_grid(cz, oz, grid, paired=True)
        dc, do = ph.diagrams(cz, 1), ph.diagrams(oz, 1)
        dims = [d for d in (0, 1) if dc.get(d) is not None and do.get(d) is not None]
        nulls = {d: [] for d in dims}
        for _ in range(n_swap):
            sw = rng.random(len(cz)) < 0.5
            for gg in (np.where(sw[:, None], oz, cz), np.where(sw[:, None], cz, oz)):
                dg = ph.diagrams(gg, 1)
                for d in dims:
                    if dg.get(d) is not None:
                        nulls[d].append(ph.betti_curve(dg[d], radii))
        ent = {"n_pairs": len(groups), "radius": radii.tolist(), "curves": {}}
        for d in dims:
            bc, bo = ph.betti_curve(dc[d], radii), ph.betti_curve(do[d], radii)
            nl = np.array(nulls[d], float)
            ent["curves"][str(d)] = {
                "clean": bc.tolist(), "attacked": bo.tolist(),
                "observed_l2": float(np.sqrt(np.sum((bc - bo) ** 2))),
                "summary_clean": ph.summarise(dc[d]), "summary_attacked": ph.summarise(do[d]),
                "swap_lo": np.percentile(nl, 2.5, axis=0).tolist(),
                "swap_mid": np.percentile(nl, 50, axis=0).tolist(),
                "swap_hi": np.percentile(nl, 97.5, axis=0).tolist()}
        c2, o2, _ = ph.preprocess(C, A, 2, seed, scale_on="pooled")
        succ = defaultdict(list)                       # over the reached pairs only
        for _, g, _, _, asr in pairs.get(label, []):
            if asr is not None:
                succ[g].append(bool(asr))
        ent["pca2"] = [{"group": g, "clean": c2[i].tolist(), "attacked": o2[i].tolist(),
                        "success_frac": (float(np.mean(succ[g])) if succ.get(g) else None)}
                       for i, g in enumerate(groups)]
        out[label] = ent
    return out, info


def check_topology(topo, hom):
    """Every label and homology dimension in both, observed L2 and the clean /
    attacked diagram summaries equal. Returns a list of mismatches."""
    bad = []
    hp = hom.get("per_attack") or {}
    if set(topo) != set(hp):
        bad.append(f"labels {sorted(topo)} vs homology {sorted(hp)}")
    for label in sorted(set(topo) & set(hp)):
        hb = hp[label].get("betti_l2") or {}
        hs = hp[label].get("summaries") or {}
        cv = topo[label]["curves"]
        if set(cv) != set(map(str, hb)):
            bad.append(f"{label}: dims {sorted(cv)} vs homology {sorted(map(str, hb))}")
        for d in sorted(set(cv) & set(map(str, hb))):
            ref = hb[d]["observed"] if d in hb else hb[int(d)]["observed"]
            if not np.isclose(ref, cv[d]["observed_l2"], rtol=1e-6, atol=1e-6):
                bad.append(f"{label} H{d}: L2 {cv[d]['observed_l2']} vs {ref}")
            for mine, theirs in (("summary_clean", "clean"), ("summary_attacked", "other")):
                r = (hs.get(theirs) or {}).get(d)
                m = cv[d][mine]
                if r is None or r["n"] != m["n"] or not np.isclose(r["total_persistence"], m["total_persistence"],
                                                                   rtol=1e-6, atol=1e-6):
                    bad.append(f"{label} H{d} {theirs} summary {m} vs {r}")
    return bad


# ---------------------------------------------------------------- main
def load_run(results_dir, layer, accept_step0):
    info = {}
    states = rq2.load_multi(results_dir, [layer], turn="arrival", accept_step0=accept_step0, info=info)[layer]
    outcomes = rq2.load_outcomes(results_dir)
    pairs = {atk: pairs_of(states, outcomes, atk) for atk in attacks_in(states)}
    pairs = {k: v for k, v in pairs.items() if v}
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    groups = {p[1] for ps in pairs.values() for p in ps}
    negs = clean_all_steps(results_dir, rows, layer, groups)
    return states, pairs, negs, info


def _f(x, n=3):
    return "  n/a" if x is None else f"{x:.{n}f}"


def _print_det(k, r):
    print(f"   {k:46s} same step {_f(r['auroc_vs_clean_same_step'])} (tasks shifted "
          f"{_f(r['frac_tasks_shifted'], 2)}, p_holm {_f(r.get('p_holm_same_step'), 4)}; random dir p95 "
          f"{_f(r['random_dir_p95_same_step'])}; ||h|| {_f(r['norm_baseline_same_step'])})   all steps "
          f"{_f(r['auroc_vs_clean_all_steps'])} (random dir p95 {_f(r['random_dir_p95_all_steps'])}; ||h|| "
          f"{_f(r['norm_baseline_all_steps'])}; {r['n_clean_all_steps']} clean states)"
          + (f"   population cos {_f(r['ucos_directions'])} {_ci(r['ucos_ci95'])}"
             f"{'' if r['ucos_interpretable'] else ' (not interpretable)'}" if "ucos_directions" in r else ""))


def _ci(c):
    return "[  n/a ]" if not c else f"[{c[0]:.3f},{c[1]:.3f}]"


def _holm(d, key_in, key_out):
    rq2.holm(list(d.values()), key_in, key_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--accept_step0", action="store_true")
    ap.add_argument("--transfer_dir")
    ap.add_argument("--transfer_layer", type=int)
    ap.add_argument("--transfer_accept_step0", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_null", type=int, default=2000, help="consistency / between / footprint nulls")
    ap.add_argument("--n_null_detector", type=int, default=20000,
                    help="per-fold sign flips; x folds (Bonferroni) and ~90 VWA tests (Holm) need a fine floor")
    ap.add_argument("--n_random_dirs", type=int, default=1000)
    ap.add_argument("--topology", action="store_true")
    ap.add_argument("--n_swap", type=int, default=200)
    ap.add_argument("--homology_json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.topology and not a.homology_json:
        sys.exit("FATAL: --topology needs --homology_json (the curves are checked against it)")
    if a.topology and a.n_swap < 1:
        sys.exit("FATAL: --n_swap must be >= 1")
    if a.folds < 2:
        sys.exit("FATAL: --folds must be >= 2")
    rng = np.random.default_rng(a.seed)

    states, pairs, negs, info = load_run(a.results_dir, a.layer, a.accept_step0)
    if not pairs:
        sys.exit(f"FATAL: no reached pairs in {a.results_dir} at layer {a.layer}: {json.dumps(info)[:400]}")
    rep = {"config": vars(a), "arrival": info,
           "n_pairs": {k: len(v) for k, v in pairs.items()},
           "n_clean_all_steps": {"groups": len(negs), "states": int(sum(len(v) for v in negs.values()))}}
    print(f"=== directions: {a.results_dir}, layer {a.layer}")
    print(f"reached pairs: {rep['n_pairs']}; clean states at every decision: {rep['n_clean_all_steps']}")

    print("\n1. consistency (leave-one-user-task-out cosine, mean over user tasks; null: sign flip "
          "per user task; Holm over attacks)")
    rep["consistency"] = {atk: consistency(ps, a.n_null, rng) for atk, ps in sorted(pairs.items())}
    _holm(rep["consistency"], "p_signflip", "p_holm")
    for atk, r in rep["consistency"].items():
        if "note" in r:
            print(f"   {atk:28s} {r['note']} ({r['n_groups']})")
            continue
        print(f"   {atk:28s} groups {r['n_groups']:3d}  cos mean {_f(r['logo_cos_mean'])} median "
              f"{_f(r['logo_cos_median'])} [{_f(r['logo_cos_q1'])},{_f(r['logo_cos_q3'])}]  "
              f"null p95 {_f(r['null_p95'])}  p_holm {_f(r.get('p_holm'), 4)}  pair cos succ/fail "
              f"{_f(r['pair_logo_cos_median_success'])}/{_f(r['pair_logo_cos_median_failure'])}  "
              f"succ vs other tasks' fail {_f(r['cos_success_vs_other_tasks_failure_mean'])}")

    print("\n2. between attacks (per user task; null: user tasks permuted; Holm over pairs)")
    names = sorted(pairs)
    rep["between"] = {f"{x} | {y}": between(pairs[x], pairs[y], a.n_null, rng)
                      for i, x in enumerate(names) for y in names[i + 1:]}
    _holm(rep["between"], "p_perm", "p_holm")
    for k, r in rep["between"].items():
        if "note" not in r:
            print(f"   {k}: groups {r['n_groups']}  population cos of mean Deltas {_f(r['ucos_mean_directions'])} "
                  f"{_ci(r['ucos_ci95'])}{'' if r['ucos_interpretable'] else ' (not interpretable)'} "
                  f"(plug-in {_f(r['cos_mean_directions'])})  "
                  f"matched mean {_f(r['matched_cos_mean'])} (null p95 {_f(r['null_p95'])}, p_holm "
                  f"{_f(r.get('p_holm'), 4)})  mismatched median {_f(r['mismatched_cos_median'])}")

    all_groups = {p[1] for ps in pairs.values() for p in ps}
    folds = make_folds(all_groups, a.folds, rng)
    print(f"\n3. detector <h, w> ({a.folds} folds over {len(all_groups)} user tasks, fixed for the run; "
          f"AUROC vs clean same step / all steps, against random directions; ||h|| baseline; "
          f"p (per-fold sign flip, Bonferroni over folds, Holm) only where train == test)")
    rep["detector"] = {}
    for tr in names + (["pooled"] if len(names) > 1 else []):
        for te in names:
            rep["detector"][f"{tr} -> {te}"] = detector(pairs, tr, te, negs, folds, a.n_null_detector, a.n_random_dirs, rng)
    _holm(rep["detector"], "p_same_step", "p_holm_same_step")
    for k, r in rep["detector"].items():
        if "note" in r:
            print(f"   {k}: {r['note']}")
            continue
        _print_det(k, r)

    if a.transfer_dir:
        L2 = a.transfer_layer if a.transfer_layer is not None else a.layer
        _, pairs2, negs2, info2 = load_run(a.transfer_dir, L2, a.transfer_accept_step0)
        res = transfer(pairs, pairs2, negs2, a.n_null_detector, a.n_random_dirs, rng)
        if not res:
            sys.exit(f"FATAL: transfer to {a.transfer_dir} (layer {L2}) produced nothing: "
                     f"{ {k: len(v) for k, v in pairs2.items()} } {json.dumps(info2)[:300]}")
        rep["transfer"] = {"dir": a.transfer_dir, "layer": L2, "arrival": info2, "results": res}
        print(f"\n3b. transfer to {a.transfer_dir} (layer {L2}), no refitting; AUROC against random "
              f"directions; population cosine of the two attacks' mean Deltas (jackknife 95%)")
        for k, r in res.items():
            _print_det(k, r)

    print("\n4. defences (footprint F = f - c; population cosine of mean footprint and mean attack "
          "Delta, same-task products left out, jackknife 95%; footprint consistency Holm over attack x defence)")
    rep["defences"] = {}
    for atk, ps in sorted(pairs.items()):
        r = defences(states, atk, ps, a.n_null, rng)
        if r:
            rep["defences"][atk] = r
    flat = [e for r in rep["defences"].values() for e in r["per_defence"].values()]
    rq2.holm(flat, "footprint_p_signflip", "footprint_p_holm")
    for atk, r in rep["defences"].items():
        for dn, e in r["per_defence"].items():
            print(f"   {atk:24s} {dn:24s} groups {e['n_groups']:3d}  cos(F, attack) "
                  f"{_f(e['footprint_attack_ucos'])} {_ci(e['footprint_attack_ucos_ci95'])}"
                  f"{'' if e['footprint_attack_ucos_interpretable'] else ' (not interpretable)'}  "
                  f"cos(F, own Delta) {_f(e['footprint_cos_own_delta_median'])}  F consistency "
                  f"{_f(e['footprint_logo_cos_mean'])} (p_holm {_f(e.get('footprint_p_holm'), 4)})")
        for k, e in r["defence_pairs"].items():
            print(f"   {atk:24s} {k:44s} cos footprints {_f(e['footprint_cos_median'])} "
                  f"({e['n_groups_footprint']})  cos suppression {_f(e['suppression_cos_median'])} "
                  f"(from c: {_f(e['suppression_from_clean_cos_median'])}; {e['n_groups_suppression']})")

    if a.topology:
        # the clouds must be built exactly as the homology run built them
        hj = json.load(open(a.homology_json))
        hc = hj.get("config")
        if not hc:
            sys.exit(f"FATAL: {a.homology_json} has no config block")
        want = {"layer": a.layer, "turn_strategy": "arrival", "accept_step0": a.accept_step0,
                "results_dir": os.path.basename(os.path.normpath(a.results_dir))}
        got = {"layer": hc.get("layer"), "turn_strategy": hc.get("turn_strategy"),
               "accept_step0": bool(hc.get("accept_step0")),
               "results_dir": os.path.basename(os.path.normpath(str(hc.get("results_dir"))))}
        if want != got:
            sys.exit(f"FATAL: {a.homology_json} was run with {got}, this run is {want}")
        kw = {k: int(hc[k]) for k in ("pca_dim", "grid", "max_points", "min_points", "seed")}
        topo, _ = topology(a.results_dir, a.layer, a.accept_step0, pairs, a.n_swap, kw.pop("seed"), **kw)
        bad = check_topology(topo, hj)
        print("\n5. topology figure data (labels, dims, Betti L2 and diagram summaries vs the homology JSON)")
        for label, e in topo.items():
            for d, cv in e["curves"].items():
                print(f"   {label:24s} H{d}: L2 {cv['observed_l2']:.2f}  ({e['n_pairs']} pairs)")
        if bad:
            sys.exit(f"FATAL: topology does not reproduce {a.homology_json}: {bad}; nothing written")
        print("   reproduces the homology JSON")
        rep["topology"] = topo
        rep["topology_settings"] = {**kw, "seed": int(hc["seed"])}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    os.replace(a.out + ".tmp", a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
