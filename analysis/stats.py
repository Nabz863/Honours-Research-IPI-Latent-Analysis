"""
analysis/stats.py -- hypothesis testing for the IPI geometry results.

Motivation: 83.5% of AgentDojo episodes have an EXACTLY zero activation
delta -- the injection sits in a tool result the agent never retrieved,
so the attacked forward pass is byte-identical to the clean one. Any
probe trained on the full population is trained on ~84% identical pairs
and cannot exceed chance. Conditioning on delta > 0 is not cherry-picking:
it is restricting to the episodes where the injected content actually
entered the model's context, which is the population the hypothesis is
about.

Tests performed
---------------
1. Reachability rate per attack, with Wilson 95% CIs on the clustered n_eff.
2. Paired comparison of ||dh|| between attacks (same tasks, so paired),
   collapsed to one observation per user task.
   Wilcoxon signed-rank -- deltas are heavily right-skewed (mean 34,
   median 3.7), so a t-test's normality assumption is untenable.
   Effect size: matched-pairs rank-biserial correlation.
3. Linear probe (clean vs attacked) on the delta>0 subset, per layer,
   GroupKFold by user task. Significance by PERMUTATION test on shuffled
   labels rather than comparison to a nominal 0.5, because with grouped
   folds and unbalanced classes the true null is not exactly 0.5.
4. Holm-Bonferroni correction within each family: across attack pairs
   (2), across attacks (3, 5). One layer per run.
5. Does a non-zero delta predict behavioural ASR success? Between user tasks
   (one observation per task, permutation); within-task CMH as secondary.
Throughout, user tasks (geometry.user_task_group) are the independent unit:
injection-task repeats of one user task share its prompt.

Usage
-----
    python3 analysis/stats.py --results_dir agentdojo_matrix_v4_qwen3vl \\
        --layer 31 --n_perm 200 --out analysis_out/stats_qwen3vl_v4.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("torch required")

try:
    from scipy import stats as sps
except ImportError:
    sps = None
    print("[warn] scipy missing -- `venv/bin/pip install scipy`; "
          "falling back to bootstrap-only where possible\n")

ZERO_TOL = 1e-6

# One grouping key and one effect size across the analysis scripts
# (geometry.py sits next to this file).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import user_task_group, matched_rank_biserial  # noqa: E402


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_pairs(results_dir, layer):
    """{attack: [(task_id, group, h_clean, h_adv, asr), ...]}"""
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    by_task = defaultdict(dict)
    for r in rows:
        if r.get("error"):          # audit 2b.1: errored rows never enter stats
            continue
        if r.get("defense_name") in (None, "none"):
            by_task[r["task_id"]][r["attack_name"]] = r

    def vec(sample_id):
        p = os.path.join(results_dir, sample_id + ".pt")
        if not os.path.exists(p):
            return None
        try:
            turns = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            return None
        if not isinstance(turns, list) or not turns:
            return None
        vs = [t[layer].float().numpy() for t in turns if isinstance(t, dict) and layer in t]
        return np.mean(vs, axis=0) if vs else None

    out = defaultdict(list)
    for task_id, conds in by_task.items():
        if "clean" not in conds:
            continue
        hc = vec(conds["clean"]["sample_id"])
        if hc is None:
            continue
        group = user_task_group(task_id)       # AgentDojo user task / VWA base task
        for attack, r in conds.items():
            if attack == "clean":
                continue
            ha = vec(r["sample_id"])
            if ha is None:
                continue
            # audit register: scoring_error_final rows are excluded from ASR
            # (unscored, like asr None) -- the activations are still valid.
            asr = None if r.get("scoring_error_final") else r.get("asr_success")
            out[attack].append((task_id, group, hc, ha, asr))
    return out


# --------------------------------------------------------------------------
# 1. reachability
# --------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """Wilson score interval -- correct for proportions near 0 or 1,
    unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wilson_p(p, n, z=1.96):
    """Wilson interval from a proportion and a (possibly fractional) n."""
    if not n:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def effective_n(y, groups):
    """Episodes of one user task are not independent (the injection-task
    repeats share the prompt), so a proportion's interval must use the
    clustered sample size. n_eff = p(1-p) / Var_cluster(p_hat), with the
    cluster-robust (ratio-estimator) variance over user-task groups, clipped
    to [1, N]. When p is 0 or 1 the variance is 0 and carries no information
    about clustering; fall back to the number of groups (each counts once)."""
    y = np.asarray(y, dtype=float)
    N = len(y)
    tot, m = defaultdict(float), defaultdict(int)
    for v, g in zip(y, groups):
        tot[g] += v
        m[g] += 1
    G = len(m)
    p = y.mean()
    if G < 2:
        return 1.0, G
    t = np.array([tot[g] for g in m]); mm = np.array([m[g] for g in m], dtype=float)
    var = G / (G - 1) * np.sum((t - p * mm) ** 2) / N ** 2
    if p in (0.0, 1.0):
        return float(G), G
    if var <= 0:                       # repeats exactly balance within every task
        return float(N), G
    return float(np.clip(p * (1 - p) / var, 1.0, N)), G


def reachability(pairs):
    res = {}
    for attack, items in sorted(pairs.items()):
        d = np.array([np.linalg.norm(ha - hc) for _, _, hc, ha, _ in items])
        nz = int((d > ZERO_TOL).sum())
        n_eff, n_groups = effective_n(d > ZERO_TOL, [g for _, g, *_ in items])
        lo, hi = wilson_p(nz / len(d), n_eff)
        nzv = d[d > ZERO_TOL]
        res[attack] = {
            "n": len(d), "n_nonzero": nz, "rate": nz / len(d),
            # Wilson on the cluster-effective n (user tasks, finding 8)
            "rate_ci95": [lo, hi], "n_groups": n_groups, "n_eff": n_eff,
            "delta_mean_all": float(d.mean()),
            "delta_mean_nonzero": float(nzv.mean()) if nz else None,
            "delta_median_nonzero": float(np.median(nzv)) if nz else None,
            "delta_max": float(d.max()),
        }
    return res


# --------------------------------------------------------------------------
# 2. paired attack comparison
# --------------------------------------------------------------------------

def rank_biserial(x, y):
    """Matched-pairs rank-biserial r = (R+ - R-) / (R+ + R-) on the ranks of
    the non-zero |x - y| (the Wilcoxon effect size). Bounded [-1, 1]; sign
    says which attack displaces more."""
    return matched_rank_biserial(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))


def compare_attacks(pairs, nonzero_only=True):
    """Wilcoxon signed-rank on ||dh||, paired by task, then collapsed to ONE
    observation per user task (the mean over its injection tasks) before the
    test -- the injection-task repeats are not independent (finding 8, as in
    rq2_geometry). Medians stay per pair (descriptive); the test and r use the
    per-user-task means; min 6 user tasks."""
    mags, grp = {}, {}
    for attack, items in pairs.items():
        mags[attack] = {t: float(np.linalg.norm(ha - hc))
                        for t, _, hc, ha, _ in items}
        for t, g, *_ in items:
            grp[t] = g

    names = sorted(mags)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = sorted(set(mags[a]) & set(mags[b]))
            xa = np.array([mags[a][t] for t in shared])
            xb = np.array([mags[b][t] for t in shared])
            gs = np.array([grp[t] for t in shared])
            if nonzero_only:
                keep = (xa > ZERO_TOL) | (xb > ZERO_TOL)
                xa, xb, gs = xa[keep], xb[keep], gs[keep]
            ug = sorted(set(gs.tolist()))
            ga = np.array([xa[gs == g].mean() for g in ug])
            gb = np.array([xb[gs == g].mean() for g in ug])
            entry = {"attack_a": a, "attack_b": b, "n_pairs": int(xa.size),
                     "n_groups": len(ug),
                     "median_a": float(np.median(xa)) if xa.size else None,
                     "median_b": float(np.median(xb)) if xb.size else None,
                     "rank_biserial_r": rank_biserial(ga, gb)}
            if sps is not None and len(ug) >= 6 and np.any(ga - gb):
                W, p = sps.wilcoxon(ga, gb, zero_method="wilcox",
                                    alternative="two-sided")
                entry["wilcoxon_W"] = float(W)
                entry["p_raw"] = float(p)
            out.append(entry)

    if sps is not None and out:
        holm(out, "p_raw", "p_holm")
    return out


def holm(entries, key_in, key_out):
    """Holm-Bonferroni step-down. Controls family-wise error without
    Bonferroni's conservatism."""
    have = [e for e in entries if e.get(key_in) is not None]
    m = len(have)
    for rank, e in enumerate(sorted(have, key=lambda x: x[key_in])):
        e[key_out] = float(min(1.0, e[key_in] * (m - rank)))
    # enforce monotonicity
    prev = 0.0
    for e in sorted(have, key=lambda x: x[key_in]):
        e[key_out] = max(e[key_out], prev)
        prev = e[key_out]


# --------------------------------------------------------------------------
# 3. probe with permutation test
# --------------------------------------------------------------------------

def probe(pairs, layers_items, n_perm=200, seed=0):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        print("[warn] scikit-learn missing -- skipping probe")
        return {}

    def cv_acc(X, y, g, rng_labels=None):
        yy = y if rng_labels is None else rng_labels
        n_groups = len(set(g.tolist()))
        if n_groups < 2 or len(set(yy.tolist())) < 2:
            return None
        gkf = GroupKFold(n_splits=min(3, n_groups))
        accs = []
        for tr, te in gkf.split(X, yy, groups=g):
            if len(set(yy[tr].tolist())) < 2:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear", dual=True,
                                   max_iter=200, random_state=seed))
            clf.fit(X[tr], yy[tr])
            accs.append(clf.score(X[te], yy[te]))
        return float(np.mean(accs)) if accs else None

    rng = np.random.default_rng(seed)
    results = {}
    for attack, items in sorted(layers_items.items()):
        X, y, g = [], [], []
        for _, group, hc, ha, _ in items:
            if np.linalg.norm(ha - hc) <= ZERO_TOL:
                continue                       # identical pair: no signal
            X.append(hc); y.append(0); g.append(group)
            X.append(ha); y.append(1); g.append(group)
        X, y, g = np.array(X), np.array(y), np.array(g)
        if X.shape[0] < 12:
            results[attack] = {"n": int(X.shape[0]), "note": "too few non-zero pairs"}
            continue

        obs = cv_acc(X, y, g)
        if obs is None:
            results[attack] = {"n": int(X.shape[0]), "note": "cv failed"}
            continue

        # Permutation null: shuffle labels WITHIN group so the grouped-CV
        # structure and class balance are preserved under H0.
        null = []
        for _ in range(n_perm):
            yp = y.copy()
            for grp in set(g.tolist()):
                idx = np.where(g == grp)[0]
                yp[idx] = rng.permutation(yp[idx])
            a = cv_acc(X, y, g, rng_labels=yp)
            if a is not None:
                null.append(a)
        null = np.array(null)
        p = float((1 + np.sum(null >= obs)) / (1 + null.size)) if null.size else None

        results[attack] = {
            "n_samples": int(X.shape[0]),
            "n_pairs": int(X.shape[0] // 2),
            "accuracy": obs,
            "null_mean": float(null.mean()) if null.size else None,
            "null_p95": float(np.percentile(null, 95)) if null.size else None,
            "p_perm": p,
            "n_perm": int(null.size),
        }
    if results:
        holm([v for v in results.values() if v.get("p_perm") is not None],
             "p_perm", "p_holm")
    return results


# --------------------------------------------------------------------------
# 5. does reachability predict ASR?
# --------------------------------------------------------------------------

def _within_group_perm(groups, reached, hit, n_perm, rng):
    """Exact-in-the-limit test of reach-vs-ASR association WITHIN user tasks.
    Under H0 a user task's ASR outcomes are exchangeable across its reached and
    unreached episodes, so hit labels are shuffled within each user task
    (reach and per-task margins fixed -- the Cochran-Mantel-Haenszel null, the
    within-group permutation of chapter 3's lemma). Statistic |a - E[a]|,
    a = #reached&hit. Pooled Fisher is invalid here: reach and ASR both depend
    on the user task, so pooling counts that shared dependence as association
    and treats repeats as independent. Only user tasks with both reach classes
    AND both outcomes carry information; with none, there is no test.
    Returns (p or None, n_informative, Mantel-Haenszel OR or None)."""
    groups, reached, hit = np.asarray(groups), np.asarray(reached, bool), np.asarray(hit, bool)
    strata = []
    num = den = 0.0
    for g in sorted(set(groups.tolist())):
        i = groups == g
        r, h = reached[i], hit[i]
        n = i.sum()
        a_ = np.sum(r & h); b_ = np.sum(r & ~h); c_ = np.sum(~r & h); d_ = np.sum(~r & ~h)
        num += a_ * d_ / n
        den += b_ * c_ / n
        if 0 < r.sum() < n and 0 < h.sum() < n:
            strata.append((r, h))
    mh_or = (num / den) if den > 0 else ("inf" if num > 0 else None)
    if not strata:
        return None, 0, mh_or
    obs = sum(np.sum(r & h) for r, h in strata)
    exp = sum(r.sum() * h.sum() / len(r) for r, h in strata)
    dev = abs(obs - exp)
    null = np.zeros(n_perm)
    for r, h in strata:
        H = rng.permuted(np.tile(h, (n_perm, 1)), axis=1)
        null += H[:, r].sum(axis=1)
    p = (1 + np.sum(np.abs(null - exp) >= dev - 1e-12)) / (1 + n_perm)
    return float(p), len(strata), mh_or


def _task_stratum(group, n_scored):
    """Permutation stratum for a user task: its suite / site (the group id
    without its trailing _N: 'workspace_user_task_3' -> 'workspace_user_task',
    'reddit_36' -> 'reddit') and a scored-episode-count band. Suites differ in
    both reach and base ASR, and a task's rate from 1-2 episodes is not
    exchangeable with one from 6-9 (review, round 5)."""
    band = 0 if n_scored <= 2 else (1 if n_scored <= 5 else 2)
    return (re.sub(r"_\d+$", "", str(group)), band)


def _between_task_perm(groups, reached, hit, n_perm, rng):
    """PRIMARY reach-vs-ASR test, one observation per user task. Under greedy
    decoding and fixed placement, reach is (nearly) constant within a user task,
    so reach is the task-level exposure and the between-task contrast is the
    question. Per user task: r_g = fraction of its scored episodes reached,
    h_g = its ASR rate. Statistic T = |sum_s sum_g (r_g - mean_s r)(h_g - mean_s h)|
    with s the task's stratum (suite x scored-count band, _task_stratum); null:
    permute h_g across user tasks WITHIN each stratum (exact under H0: within a
    suite and size band, a task's ASR rate does not depend on its reach).
    Needs >= 2 tasks in each reach class (r_g >= .5 / < .5) or there is no
    test. Returns (p or None, n_tasks, asr_reached_tasks, asr_unreached_tasks,
    scored episodes per task in each class)."""
    groups, reached, hit = np.asarray(groups), np.asarray(reached, bool), np.asarray(hit, bool)
    ug = sorted(set(groups.tolist()))
    n = np.array([np.sum(groups == g) for g in ug])
    r = np.array([reached[groups == g].mean() for g in ug])
    h = np.array([hit[groups == g].mean() for g in ug])
    cls = r >= 0.5
    asr_r = float(h[cls].mean()) if cls.any() else None
    asr_u = float(h[~cls].mean()) if (~cls).any() else None
    sizes = {"reached": float(n[cls].mean()) if cls.any() else None,
             "unreached": float(n[~cls].mean()) if (~cls).any() else None}
    if cls.sum() < 2 or (~cls).sum() < 2:
        return None, len(ug), asr_r, asr_u, sizes
    strata = defaultdict(list)
    for i, g in enumerate(ug):
        strata[_task_stratum(g, n[i])].append(i)
    # only strata where both r and h vary carry information
    S = [np.array(ix) for ix in strata.values()
         if len(ix) > 1 and np.ptp(r[ix]) > 0 and np.ptp(h[ix]) > 0]
    if not S:
        return None, len(ug), asr_r, asr_u, sizes
    T = abs(sum(np.dot(r[ix] - r[ix].mean(), h[ix] - h[ix].mean()) for ix in S))
    null = np.zeros(n_perm)
    for ix in S:
        rc = r[ix] - r[ix].mean()
        H = rng.permuted(np.tile(h[ix], (n_perm, 1)), axis=1)
        null += (H - h[ix].mean()) @ rc
    p = (1 + np.sum(np.abs(null) >= T - 1e-12)) / (1 + n_perm)
    return float(p), len(ug), asr_r, asr_u, sizes


def reach_vs_asr(pairs, n_perm=5000, seed=0):
    """Does reaching the model predict behavioural ASR success?
    PRIMARY (p_raw, Holm): between user tasks, one observation per task
    (_between_task_perm). SECONDARY (p_within): within user tasks, CMH-style
    permutation with Mantel-Haenszel OR -- answers 'among a task's own
    episodes, are the reached ones likelier to succeed?' and is usually
    untestable on AgentDojo, where reach is fixed per user task."""
    rng = np.random.default_rng(seed)
    out = []
    for attack, items in sorted(pairs.items()):
        a = b = c = d = 0          # nonzero/ASR, nonzero/noASR, zero/ASR, zero/noASR
        G, R, Hh = [], [], []
        for _, grp_, hc, ha, asr in items:
            if asr is None:         # P2a.2: unscored -> exclude, not a miss
                continue
            nz = np.linalg.norm(ha - hc) > ZERO_TOL
            hit = asr is True
            G.append(grp_); R.append(bool(nz)); Hh.append(hit)
            if nz and hit: a += 1
            elif nz: b += 1
            elif hit: c += 1
            else: d += 1
        e = {"attack": attack, "table": [[a, b], [c, d]],
             "asr_given_nonzero": a / (a + b) if a + b else None,
             "asr_given_zero": c / (c + d) if c + d else None}
        # A table with an empty row or column (no scored episodes in a reach
        # class, or no hits/misses at all) has no defined test: scipy returns
        # p=1.0, which would be a fabricated result and would also inflate
        # Holm's m for the other attacks. Report it as untestable instead.
        degenerate = min(a + b, c + d, a + c, b + d) == 0
        if degenerate:
            e["note"] = "degenerate table (an empty row or column): no test"
        else:
            p, n_tasks, asr_rt, asr_ut, sizes = _between_task_perm(G, R, Hh, n_perm, rng)
            e.update({"n_tasks": n_tasks, "asr_reached_tasks": asr_rt,
                      "asr_unreached_tasks": asr_ut, "n_perm": n_perm,
                      "mean_scored_episodes_per_task": sizes})
            if p is None:
                e["note"] = ("fewer than 2 user tasks in a reach class, or no stratum "
                             "(suite x size band) where task reach and ASR both vary: no test")
            else:
                e["p_raw"] = p
            pw, n_inf, mh = _within_group_perm(G, R, Hh, n_perm, rng)
            e.update({"p_within": pw, "odds_ratio_mh": mh, "n_informative_groups": n_inf})
        out.append(e)
    holm(out, "p_raw", "p_holm")
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pairs = load_pairs(a.results_dir, a.layer)
    if not pairs:
        sys.exit("no usable pairs -- check that --layer exists in the saved "
                 "tensors for this model. Layer bands differ by architecture: "
                 "18-31 for a 32-layer stack, 24-42 for Gemma's 48-layer one, "
                 "so a layer valid for one model may be absent or mid-stack "
                 "for another.")

    # Since P2a every .pt holds ALL decoder layers, so the deepest captured
    # layer is the final layer, not the analysis layer -- comparing against
    # max(layer) would warn spuriously. The real silent error is a layer that
    # is ABSENT (wrong model's band) or outside the band; a present analysis
    # layer is fine. Report the band either way so the choice is auditable.
    import glob as _glob
    _pt = sorted(_glob.glob(os.path.join(a.results_dir, "clean_*.pt")))
    if _pt:
        try:
            _t = torch.load(_pt[0], map_location="cpu", weights_only=True)
            _avail = sorted(_t[0].keys()) if _t and isinstance(_t[0], dict) else []
            if _avail and a.layer not in _avail:
                print(f"[warn] --layer {a.layer} is NOT in the captured set "
                      f"({min(_avail)}-{max(_avail)}); the number would be "
                      f"measured at the wrong depth or fail to load.\n")
            elif _avail:
                print(f"[info] layer {a.layer} of captured band "
                      f"{min(_avail)}-{max(_avail)}. Cross-model comparisons "
                      f"should match RELATIVE depth (31/32 ~ 42/48).\n")
        except Exception:
            pass

    reach = reachability(pairs)
    cmp_ = compare_attacks(pairs)
    prb = probe(pairs, pairs, n_perm=a.n_perm)
    rva = reach_vs_asr(pairs, n_perm=max(a.n_perm, 2000))

    # Finding 4: the JSON is the deliverable, the console table is a courtesy.
    # Write it FIRST (atomically: tmp + rename, so a node dying mid-write never
    # leaves a truncated file that looks complete), then print.
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"config": vars(a), "reachability": reach,
                       "attack_comparison": cmp_, "probe": prb,
                       "reach_vs_asr": rva}, fh, indent=2, default=float)
        os.replace(tmp, a.out)

    print_report(a, reach, cmp_, prb, rva)
    if a.out:
        print(f"wrote {a.out}")


def _fmt(v, spec, width=0):
    """None-safe format: a zero-reach attack legitimately yields None for its
    nonzero-delta mean/median, an empty comparison yields None medians, and a
    probe with an empty null yields None p. Print 'n/a', never crash."""
    s = "n/a" if v is None else (v if isinstance(v, str) else format(v, spec))
    return s.rjust(width) if width else s


def print_report(a, reach, cmp_, prb, rva):
    print(f"\n=== {a.results_dir}  layer {a.layer} ===\n")

    print("1. REACHABILITY (fraction of episodes with non-zero delta; 95% CI on the\n"
          "   user-task-clustered effective n)")
    for k, v in reach.items():
        lo, hi = v["rate_ci95"]
        print(f"   {k:24s} {v['n_nonzero']:4d}/{v['n']:<4d} = {100*v['rate']:5.1f}% "
              f"[{100*lo:.1f}, {100*hi:.1f}]   ||dh|| nonzero: "
              f"mean {_fmt(v['delta_mean_nonzero'], '.2f')}  "
              f"median {_fmt(v['delta_median_nonzero'], '.2f')}")

    print("\n2. PAIRED ||dh|| COMPARISON (Wilcoxon on per-user-task means, Holm-corrected)")
    for e in cmp_:
        p = e.get("p_holm")
        sig = "" if p is None else ("  *" if p < 0.05 else "   ")
        print(f"   {e['attack_a']:22s} vs {e['attack_b']:22s} n={e['n_pairs']:4d} "
              f"grp={e.get('n_groups', 0):3d} "
              f"med {_fmt(e['median_a'], '.2f', 7)} vs {_fmt(e['median_b'], '.2f', 7)}  "
              f"r={_fmt(e['rank_biserial_r'], '+.3f')}  "
              f"p={_fmt(p, '.3g')}{sig}")

    print("\n3. LINEAR PROBE on non-zero subset (permutation test, Holm-corrected)")
    for k, v in prb.items():
        if "accuracy" not in v:
            print(f"   {k:24s} {v.get('note')}")
            continue
        p = v.get("p_holm", v.get("p_perm"))
        sig = "  *" if (p is not None and p < 0.05) else "   "
        print(f"   {k:24s} n={v['n_pairs']:4d} pairs  acc={_fmt(v['accuracy'], '.3f')}  "
              f"null={_fmt(v['null_mean'], '.3f')} (p95 {_fmt(v['null_p95'], '.3f')})  "
              f"p={_fmt(p, '.3g')}{sig}")

    print("\n4. NON-ZERO DELTA vs ASR SUCCESS (between user tasks, permutation, Holm-corrected;\n"
          "   p_within = within-task CMH, secondary)")
    for e in rva:
        p = e.get("p_holm")
        az, zz = e["asr_given_nonzero"], e["asr_given_zero"]
        f = lambda v: "  n/a" if v is None else f"{100*v:5.1f}%"
        print(f"   {e['attack']:24s} task ASR reached={f(e.get('asr_reached_tasks'))} "
              f"unreached={f(e.get('asr_unreached_tasks'))}  tasks={e.get('n_tasks', 0)}  "
              f"p={_fmt(p, '.3g')}  p_within={_fmt(e.get('p_within'), '.3g')} "
              f"OR_MH={_fmt(e.get('odds_ratio_mh'), '.3g')}"
              f"{'  (' + e['note'] + ')' if e.get('note') else ''}")

    print("\n   * = significant at alpha=0.05 after Holm correction\n")


if __name__ == "__main__":
    main()
