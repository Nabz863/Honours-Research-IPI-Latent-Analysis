"""
analysis/persistent_homology.py
-------------------------------
RQ (latent topology, after Fay et al.): does an attack change the SHAPE of
the representation cloud, not just its mean position? Displacement (geometry.py)
measures where the centroid moved; persistent homology asks whether the
connectivity and loop structure of the episode cloud differ between the clean
and attacked conditions.

METHOD
    Per (results_dir, layer):
      * matched point clouds per attack -- one clean and one attacked point per
        REACHED user task (attacked = mean over its reached injection tasks,
        clean = mean of their clean partners), at the analysis layer,
        mean-over-turns. Zero-delta episodes are excluded (finding 7).
      * standardise: z-score each feature on the POOLED pair statistics (label
        invariant), then PCA to a bounded dimension (default 50) for tractable
        Vietoris-Rips. Both steps are reported.
      * persistence diagrams H0 (connected components) and H1 (loops) via
        Vietoris-Rips. ripser if installed; otherwise a self-contained
        union-find computes H0 exactly and H1 is reported as unavailable (never
        guessed).
      * summaries per condition/dim: feature count, total persistence
        (sum of death-birth over finite features), max persistence.
      * clean vs each attack: the Betti curve (feature count as a function of
        radius) is compared by L2 distance, per homology dimension, on a shared
        radius grid at quantiles of the pooled between-unit distances. Significance is a
        WITHIN-TASK SWAP null: each user task's clean and attacked points swap
        labels with probability 1/2; p = (1 + #{null >= observed}) / (1 +
        n_perm). The test is sensitive to changes in the spread / scale of the
        within-cloud distances and blind to a uniform shift; a rejection is a
        difference in distance structure, not by itself evidence of new loops
        or components.
      * if persim is installed, the bottleneck and (1-)Wasserstein distances
        between the clean and attacked diagrams are also reported, with the
        same swap null on the bottleneck distance.

    Bounded by design: at most --max_points episodes per cloud (subsampled with
    a fixed seed), --pca_dim features, --grid radii, --n_perm permutations.

Usage
    python3 analysis/persistent_homology.py \\
        --results_dir final_agentdojo_qwen3vl --layer 31 \\
        --out analysis_out/homology_qwen3vl.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("torch required")

# geometry.py sits next to this file; running `python3 analysis/<this>.py`
# puts analysis/ on sys.path[0], so a bare import resolves. Belt and braces
# for `python -m` and tests:
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (load_rows, tensor_path, load_episode, CLEAN_NAME,  # noqa: E402
                      user_task_group, arrival_or_acts, load_turns)

try:
    from ripser import ripser as _ripser
    HAVE_RIPSER = True
except Exception:
    HAVE_RIPSER = False

try:
    from persim import bottleneck as _bottleneck, wasserstein as _wasserstein
    HAVE_PERSIM = True
except Exception:
    HAVE_PERSIM = False


# ---------------------------------------------------------------------------
# point clouds
# ---------------------------------------------------------------------------

ZERO_TOL = 1e-6     # the zero-delta threshold stats.py and rq2_geometry.py use


def _dedup(x, tol=ZERO_TOL):
    """Collapse points within `tol` (L2) of an earlier kept point. Identical
    points add only zero-length H0 features and weight the permutation null
    toward them; one copy carries all the topology. Exact duplicates go first
    (np.unique), then a greedy pass catches float-noise near-duplicates."""
    x = np.unique(x, axis=0)
    keep = []
    for i in range(len(x)):
        if keep and np.linalg.norm(x[keep] - x[i], axis=1).min() <= tol:
            continue
        keep.append(i)
    return x[keep]


def clouds(results_dir, layer, turn_strategy="mean", return_info=False):
    """{condition_label: np.float64[n_points, d]} with 'clean' plus one entry
    per attack_name. Undefended rows only; errored rows skipped.

    Finding 7: an attacked episode enters its cloud only if the injection
    actually entered the representation -- ||a - c|| > ZERO_TOL against the
    clean episode of the SAME task_id (the stats.py / rq2_geometry.py
    reachability gate). An attacked episode with no usable clean partner
    cannot be shown to be reached and is excluded too. Both clouds are then
    deduplicated: AgentDojo's clean episode depends only on the user task and
    recurs once per injection task. Without this the attacked cloud is mostly
    copies of clean points (~83% zero-delta on AgentDojo).

    return_info=True also returns per-label counts of what was excluded.

    These clouds are descriptive (counts, summaries). The TEST uses
    matched_pairs(): see there."""
    clean_by_task, n_clean_loaded, attacked = _load_undefended(results_dir, layer, turn_strategy)
    return _clouds_from(clean_by_task, n_clean_loaded, attacked, return_info)


def _load_undefended(results_dir, layer, turn_strategy, accept_step0=False, info=None):
    """(clean_by_task {task_id: vec}, n_clean_loaded, attacked {label: [(task_id, vec[, clean_vec])]})
    for undefended, non-errored rows at `layer`.
    turn_strategy 'arrival': each attacked item carries its own clean vector,
    both at the payload's arrival step (geometry.arrival_step); an unexposed
    pair carries attacked = clean (zero-delta, excluded downstream and
    counted); a non-arrival or transcript-less pair is dropped and counted in
    info[label][status]."""
    if turn_strategy == "arrival":
        return _load_arrival(results_dir, layer, accept_step0, info if info is not None else {})
    rows = load_rows(results_dir)
    clean_by_task = {}
    n_clean_loaded = 0
    attacked = defaultdict(list)            # label -> [(task_id, vec)]
    for r in rows:
        if r.get("error"):
            continue
        if r.get("defense_name") not in (None, "none"):
            continue
        cond = r.get("condition")
        if cond not in ("clean", "attacked"):
            continue
        acts = load_episode(tensor_path(r), turn_strategy)
        if acts is None or layer not in acts:
            continue
        v = np.asarray(acts[layer], dtype=np.float64)
        if cond == "clean":
            clean_by_task[r.get("task_id")] = v
            n_clean_loaded += 1
        else:
            attacked[r.get("attack_name")].append((r.get("task_id"), v))
    return clean_by_task, n_clean_loaded, attacked


def _load_arrival(results_dir, layer, accept_step0, info):
    rows = load_rows(results_dir)
    clean, att = {}, defaultdict(list)
    for r in rows:
        if r.get("error") or r.get("defense_name") not in (None, "none"):
            continue
        if r.get("condition") == "clean":
            clean[r.get("task_id")] = r["sample_id"]
        elif r.get("condition") == "attacked":
            att[r.get("attack_name")].append((r.get("task_id"), r["sample_id"]))
    plan, needs = [], defaultdict(set)
    for label, items in att.items():
        il = info.setdefault(label, {})
        for tid, a_sid in items:
            c_sid = clean.get(tid)
            if c_sid is None:
                il["no_clean"] = il.get("no_clean", 0) + 1
                continue
            t, st = arrival_or_acts(results_dir, c_sid, a_sid, accept_step0=accept_step0)
            il[st] = il.get(st, 0) + 1
            if st == "not_exposed":
                plan.append((label, tid, c_sid, 0, c_sid, 0))
                needs[c_sid].add(0)
            elif t is not None:
                plan.append((label, tid, c_sid, t, a_sid, t))
                needs[c_sid].add(t)
                needs[a_sid].add(t)
    vec = load_turns(results_dir, needs, [layer])
    clean_by_task, attacked = {}, defaultdict(list)
    for label, tid, c_sid, tc, a_sid, ta in plan:
        c = vec.get((c_sid, tc), {}).get(layer)
        v = vec.get((a_sid, ta), {}).get(layer)
        if c is None or v is None:                      # never silently dropped
            info[label]["no_tensor"] = info[label].get("no_tensor", 0) + 1
            continue
        c, v = np.asarray(c, np.float64), np.asarray(v, np.float64)
        clean_by_task.setdefault(tid, c)
        attacked[label].append((tid, v, c))
    return clean_by_task, len(clean_by_task), attacked


def _partner(clean_by_task, item):
    """the attacked item's clean partner: its own (arrival mode) or by task_id."""
    return item[2] if len(item) > 2 else clean_by_task.get(item[0])


def _clouds_from(clean_by_task, n_clean_loaded, attacked, return_info=False):
    out, info = {}, {}
    if clean_by_task:
        out[CLEAN_NAME] = _dedup(np.vstack(list(clean_by_task.values())))
        info[CLEAN_NAME] = {"n_loaded": n_clean_loaded,
                            "n_unique": int(len(out[CLEAN_NAME]))}
    for label, items in attacked.items():
        reached, n_no_clean, n_zero = [], 0, 0
        for item in items:
            v, c = item[1], _partner(clean_by_task, item)
            if c is None or c.shape != v.shape:
                n_no_clean += 1
            elif np.linalg.norm(v - c) <= ZERO_TOL:
                n_zero += 1
            else:
                reached.append(v)
        info[label] = {"n_loaded": len(items), "n_no_clean": n_no_clean,
                       "n_zero_delta": n_zero, "n_reached": len(reached),
                       "n_unique": 0}
        if reached:
            out[label] = _dedup(np.vstack(reached))
            info[label]["n_unique"] = int(len(out[label]))
    return (out, info) if return_info else out


def matched_pairs(clean_by_task, attacked):
    """The RQ3 test units. {label: (C, A, groups)} with C[i], A[i] the clean and
    attacked point of reached user task groups[i].

    Why matched (review, RQ3 null): the attacked cloud exists only for the user
    tasks the injection reached, K points per task (one per injection task),
    each = that task's clean point + a change; the full clean cloud spans every
    task once. A free label split treats all points as exchangeable, so it
    "detects" which tasks were reached and how many repeats they had, not a
    change of shape (simulated: p=0.01 with no effect). Here each reached user
    task (geometry.user_task_group) contributes exactly one clean point and one
    attacked point (the mean over its reached injection tasks, against the mean
    of their clean partners), so the two clouds cover the same tasks and the
    within-task swap null in compare(paired=True) is valid."""
    out = {}
    for label, items in attacked.items():
        byg = defaultdict(lambda: ([], []))
        for item in items:
            tid, v, c = item[0], item[1], _partner(clean_by_task, item)
            if c is None or c.shape != v.shape or np.linalg.norm(v - c) <= ZERO_TOL:
                continue
            g = user_task_group(tid)
            byg[g][0].append(c)
            byg[g][1].append(v)
        groups = sorted(byg)
        if groups:
            C = np.vstack([np.mean(byg[g][0], axis=0) for g in groups])
            A = np.vstack([np.mean(byg[g][1], axis=0) for g in groups])
            out[label] = (C, A, groups)
    return out


def preprocess(clean, other, pca_dim, seed, scale_on="clean"):
    """z-score, then PCA (fit on the pooled set) to pca_dim.
    scale_on='clean' z-scores on the clean cloud's statistics; 'pooled' on the
    pooled set, which is invariant to swapping clean/attacked labels and so
    keeps the paired permutation null exact.
    Returns (clean_z_pca, other_z_pca, info)."""
    ref = np.vstack([clean, other]) if scale_on == "pooled" else clean
    mu = ref.mean(0)
    sd = ref.std(0)
    sd[sd < 1e-8] = 1.0
    cz = (clean - mu) / sd
    oz = (other - mu) / sd
    pooled = np.vstack([cz, oz])
    k = int(min(pca_dim, pooled.shape[0] - 1, pooled.shape[1]))
    info = {"pca_dim": k, "n_clean": len(clean), "n_other": len(other),
            "scale_on": scale_on}
    if k < 1:
        return cz, oz, info
    # PCA via SVD on the mean-centred pooled matrix (no sklearn dependency)
    pc = pooled - pooled.mean(0)
    _, _, vt = np.linalg.svd(pc, full_matrices=False)
    comp = vt[:k]
    return (cz - pooled.mean(0)) @ comp.T, (oz - pooled.mean(0)) @ comp.T, info


def subsample(x, n, rng):
    if len(x) <= n:
        return x
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


# ---------------------------------------------------------------------------
# persistence diagrams
# ---------------------------------------------------------------------------

def _h0_union_find(points):
    """Exact H0 persistence for a Vietoris-Rips filtration: single-linkage
    merge distances. Births are all 0; deaths are the n-1 merge radii; one
    component survives to infinity. Returns np.array[[0, death], ...] with a
    trailing [0, inf]."""
    n = len(points)
    if n <= 1:
        return np.array([[0.0, np.inf]])
    # condensed pairwise distances, edges sorted ascending
    d = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    iu = np.triu_indices(n, k=1)
    order = np.argsort(dist[iu])
    ei, ej = iu[0][order], iu[1][order]
    ew = dist[iu][order]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    deaths = []
    for a, b, w in zip(ei, ej, ew):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
            deaths.append(float(w))        # a component dies when it merges
            if len(deaths) == n - 1:
                break
    return np.array([[0.0, x] for x in deaths] + [[0.0, np.inf]])


def diagrams(points, maxdim=1):
    """{0: array[[b,d],...], 1: array or None}. ripser when available (H0+H1),
    else union-find H0 and H1 unavailable."""
    if HAVE_RIPSER and len(points) >= 2:
        import warnings
        with warnings.catch_warnings():
            # fewer points than PCA dims is expected here (one point per user
            # task); ripser's "more columns than rows" hint is noise, not an error
            warnings.filterwarnings("ignore", message=".*more columns than rows.*")
            dg = _ripser(points, maxdim=maxdim)["dgms"]
        return {d: dg[d] for d in range(len(dg))}
    return {0: _h0_union_find(points), 1: None}


def _finite(dgm):
    """Drop features with a non-finite death (the infinite H0 component).
    persim's bottleneck/Wasserstein ignore them anyway, with a warning."""
    if dgm is None or len(dgm) == 0:
        return np.empty((0, 2))
    return dgm[np.isfinite(dgm[:, 1])]


def summarise(dgm):
    """feature count / total persistence / max persistence over FINITE
    features (the infinite H0 component is dropped)."""
    if dgm is None or len(dgm) == 0:
        return {"n": 0, "total_persistence": 0.0, "max_persistence": 0.0}
    life = dgm[:, 1] - dgm[:, 0]
    finite = life[np.isfinite(life)]
    if finite.size == 0:
        return {"n": 0, "total_persistence": 0.0, "max_persistence": 0.0}
    return {"n": int(finite.size),
            "total_persistence": float(finite.sum()),
            "max_persistence": float(finite.max())}


def betti_curve(dgm, radii):
    """#features alive at each radius: birth <= r < death (infinite deaths
    count as alive everywhere)."""
    if dgm is None or len(dgm) == 0:
        return np.zeros(len(radii))
    b, dth = dgm[:, 0], dgm[:, 1]
    out = np.empty(len(radii))
    for i, r in enumerate(radii):
        out[i] = int(np.sum((b <= r) & (r < dth)))
    return out


# ---------------------------------------------------------------------------
# clean-vs-attack comparison: paired swap null (main) / label split
# ---------------------------------------------------------------------------

def _radius_grid(a, b, grid, paired=False):
    """Radii at evenly spaced QUANTILES of the pooled pairwise distances that a
    sub-cloud can contain. With paired=True the within-pair distances
    |a[i]-b[i]| are excluded: no swapped sub-cloud ever holds both members of a
    pair, so those (usually the smallest) distances only waste grid points.
    Every H0/H1 birth and finite death of either cloud is an edge length inside
    that cloud, hence one of these distances, so the grid spans them all; and
    the set depends only on the pairs, so it is unchanged by the swap null."""
    pooled = np.vstack([a, b])
    d = pooled[:, None, :] - pooled[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    keep = np.triu(np.ones(dist.shape, bool), k=1)
    if paired:
        n = len(a)
        keep[np.arange(n), np.arange(n) + n] = False
    vals = dist[keep]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, grid)
    return np.quantile(vals, np.linspace(0.0, 1.0, grid))


def compare(clean, other, radii, n_perm, maxdim, rng, paired=False):
    """paired=False: label-permutation null (pool, split at random).
    paired=True: clean[i] and other[i] are the two conditions of the same unit;
    the null swaps each pair's labels with probability 1/2 (within-unit
    permutation), which is exact under H0: the two conditions of a unit are
    exchangeable."""
    dg_c, dg_o = diagrams(clean, maxdim), diagrams(other, maxdim)
    dims = sorted(d for d in (set(dg_c) | set(dg_o))
                  if dg_c.get(d) is not None or dg_o.get(d) is not None)
    obs = {}
    for dim in dims:
        cc = betti_curve(dg_c.get(dim), radii)
        co = betti_curve(dg_o.get(dim), radii)
        obs[dim] = float(np.sqrt(np.sum((cc - co) ** 2)))

    bott = {}
    if HAVE_PERSIM:
        for dim in dims:
            if dg_c.get(dim) is not None and dg_o.get(dim) is not None:
                try:
                    fc, fo = _finite(dg_c[dim]), _finite(dg_o[dim])
                    bott[dim] = {
                        "bottleneck": float(_bottleneck(fc, fo)),
                        "wasserstein": float(_wasserstein(fc, fo)),
                    }
                except Exception:
                    pass

    # permutation null (paired swap or label split) on the Betti-curve distance (and bottleneck)
    pooled = np.vstack([clean, other])
    nA = len(clean)
    null_betti = {dim: [] for dim in dims}
    null_bott = {dim: [] for dim in bott}
    for _ in range(n_perm):
        if paired:
            sw = rng.random(nA) < 0.5
            g1 = np.where(sw[:, None], other, clean)
            g2 = np.where(sw[:, None], clean, other)
        else:
            perm = rng.permutation(len(pooled))
            g1, g2 = pooled[perm[:nA]], pooled[perm[nA:]]
        d1, d2 = diagrams(g1, maxdim), diagrams(g2, maxdim)
        for dim in dims:
            b1 = betti_curve(d1.get(dim), radii)
            b2 = betti_curve(d2.get(dim), radii)
            null_betti[dim].append(float(np.sqrt(np.sum((b1 - b2) ** 2))))
        for dim in bott:
            if d1.get(dim) is not None and d2.get(dim) is not None:
                try:
                    null_bott[dim].append(float(_bottleneck(_finite(d1[dim]), _finite(d2[dim]))))
                except Exception:
                    pass

    res = {"betti_l2": {}, "summaries": {
        "clean": {d: summarise(dg_c.get(d)) for d in dims},
        "other": {d: summarise(dg_o.get(d)) for d in dims}}}
    for dim in dims:
        nn = np.array(null_betti[dim], dtype=float)
        p = float((1 + np.sum(nn >= obs[dim])) / (1 + nn.size)) if nn.size else None
        if nn.size and obs[dim] == 0 and not np.any(nn):
            p = None          # observed and every null curve identical: no information
        res["betti_l2"][dim] = {"observed": obs[dim],
                                "null_mean": float(nn.mean()) if nn.size else None,
                                "null_p95": float(np.percentile(nn, 95)) if nn.size else None,
                                "p_perm": p}
    if bott:
        res["diagram_distance"] = {}
        for dim in bott:
            nn = np.array(null_bott.get(dim, []), dtype=float)
            p = float((1 + np.sum(nn >= bott[dim]["bottleneck"])) / (1 + nn.size)) if nn.size else None
            res["diagram_distance"][dim] = {**bott[dim], "bottleneck_p_perm": p}
    return res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--turn_strategy", default="arrival", choices=["arrival", "first", "mean", "last"],
                    help="arrival (default): clean and attacked at the payload's arrival step, "
                         "the RQ1 headline's decision")
    ap.add_argument("--accept_step0", action="store_true",
                    help="payload is in the first observation (VWA): accept arrival at step 0")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--max_points", type=int, default=150,
                    help="cap per cloud; Vietoris-Rips is superlinear")
    ap.add_argument("--grid", type=int, default=50, help="radii in the Betti curve")
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--maxdim", type=int, default=1, help="1 = H0 and H1")
    ap.add_argument("--min_points", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not HAVE_RIPSER:
        print("[warn] ripser not installed -- H0 via union-find only, H1 "
              "unavailable. Install with "
              "`venv/bin/pip install --break-system-packages ripser persim`.\n")
    if not HAVE_PERSIM:
        print("[warn] persim not installed -- bottleneck/Wasserstein diagram "
              "distances skipped (Betti-curve test still runs).\n")

    arrival_info = {}
    cbt, n_cl, att = _load_undefended(a.results_dir, a.layer, a.turn_strategy,
                                      a.accept_step0, arrival_info)
    cl, cinfo = _clouds_from(cbt, n_cl, att, return_info=True)
    matched = matched_pairs(cbt, att)
    if CLEAN_NAME not in cl:
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out + ".tmp", "w") as fh:
                json.dump({"config": vars(a), "arrival": arrival_info, "per_attack": {}}, fh, indent=2)
            os.replace(a.out + ".tmp", a.out)
        hint = ""
        if a.turn_strategy == "arrival" and not a.accept_step0 and any(
                v.get("not_payload_arrival") for v in arrival_info.values()):
            hint = " -- payload in the first observation (VWA)? pass --accept_step0"
        sys.exit(f"no clean cloud at layer {a.layer} in {a.results_dir}: "
                 f"arrival pairing {json.dumps(arrival_info)[:300]}{hint}")
    clean = cl[CLEAN_NAME]
    print(f"\n=== persistent homology  ({a.results_dir}, layer {a.layer}, "
          f"turns={a.turn_strategy}) ===")
    if arrival_info:
        print("payload-arrival pairing: " + "; ".join(
            f"{k}: " + ", ".join(f"{s_}={n}" for s_, n in sorted(v.items()))
            for k, v in sorted(arrival_info.items())))
    print(f"backend: {'ripser (H0+H1)' if HAVE_RIPSER else 'union-find (H0 only)'}"
          f"{' + persim' if HAVE_PERSIM else ''}")
    ci = cinfo[CLEAN_NAME]
    print(f"clean cloud: {ci['n_unique']} distinct points from {ci['n_loaded']} episodes")
    for label, v in sorted(cinfo.items()):
        if label != CLEAN_NAME:
            print(f"  {label:26s} {v['n_loaded']} episodes: {v['n_zero_delta']} zero-delta, "
                  f"{v['n_no_clean']} without a clean partner -> {v['n_reached']} reached, "
                  f"{v['n_unique']} distinct points")

    rng = np.random.default_rng(a.seed)
    out = {"config": vars(a),
           "backend": {"ripser": HAVE_RIPSER, "persim": HAVE_PERSIM},
           "clouds": cinfo,     # finding 7: what each cloud excluded
           "arrival": arrival_info,
           "per_attack": {}}
    for label in sorted(l for l in cinfo if l != CLEAN_NAME):
        C, A, groups = matched.get(label, (np.empty((0, 0)), np.empty((0, 0)), []))
        if len(groups) < a.min_points:
            print(f"  {label:26s} SKIP ({len(groups)} reached user tasks "
                  f"< min_points {a.min_points})")
            continue
        cz, oz, info = preprocess(C, A, a.pca_dim, a.seed, scale_on="pooled")
        if len(cz) > a.max_points:                     # keep pairs together
            idx = rng.choice(len(cz), size=a.max_points, replace=False)
            cz, oz = cz[idx], oz[idx]
        info["n_pairs"] = len(groups)
        info["n_pairs_used"] = len(cz)
        radii = _radius_grid(cz, oz, a.grid, paired=True)
        res = compare(cz, oz, radii, a.n_perm, a.maxdim, rng, paired=True)
        res["preprocess"] = info
        out["per_attack"][label] = res
        line = []
        for dim, v in sorted(res["betti_l2"].items()):
            p = v["p_perm"]
            star = " *" if (p is not None and p < 0.05) else ""
            line.append(f"H{dim} L2={v['observed']:.2f} "
                        f"p={'n/a' if p is None else f'{p:.3g}'}{star}")
        print(f"  {label:26s} " + "   ".join(line))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        os.replace(tmp, a.out)
        print(f"\nwrote {a.out}")
    print("\nnote: one clean and one attacked point per reached user task; the "
          "null swaps each task's two points. The test responds to changes in "
          "the spread/scale of distances and is blind to a uniform shift; read "
          "alongside geometry.py's displacement, not instead of it.")
    if not out["per_attack"]:
        # The JSON (with the cloud counts saying why) is on disk; but no RQ3
        # comparison was made for this directory, so the job must not PASS.
        sys.exit(f"no attack had >= {a.min_points} reached user tasks in "
                 f"{a.results_dir} -- no topology comparison was made (see "
                 f"'clouds' in the JSON)")


if __name__ == "__main__":
    main()
