"""
analysis/rq2_masking.py
-----------------------
Does a defence actually MASK the injection, or does it merely displace
every representation a long way while leaving the injection intact
inside that displacement?

WHY THIS IS NEEDED
    rq2_geometry.py reports along_frac -- where the defended state sits
    on the line from clean to attacked. Values of 0.42-0.81 read as
    "partway back toward clean". That reading does not survive the
    absolute distances:

        attack displacement      ||a - c||    ~3.7
        defended displacement    ||d - c||    64 - 198
        defence footprint        ||f - c||    50 - 198

    The defence moves the representation 17-56x further than the attack
    does, and it moves a CLEAN input essentially as far (footprint ~=
    defended displacement). So the large displacement is the prompt
    transformation's own signature, present with or without an attack --
    not something the attack causes and not evidence of restoration.
    along_frac is a projection onto a 3.7-unit axis; a 60-unit vector in
    an unrelated direction still projects onto it.

    That leaves the question RQ2 actually asks unanswered. "Restore
    clean-condition representations or produce suppression states"
    presupposes we can tell whether the injection is still represented
    after the defence is applied. The right test is not a distance at
    all -- it is discriminability:

        Can a probe separate defended-clean from defended-attacked?

    High accuracy  -> the injection is STILL fully represented under the
                      defence. Behavioural suppression is happening
                      downstream of the representation, and the defence
                      should be fragile to any attack variant reaching
                      the same region by another route.
    Chance         -> the defence genuinely masks the injection in the
                      representation. That is a mechanistically
                      different and much stronger claim.

    The same probe is run on the undefended pair (clean vs attacked) as
    the reference. If the undefended pair is itself near chance, nothing
    can be concluded about the defended pair -- the measurement has no
    resolution on this data, and the script says so rather than
    reporting a number.

METHOD
    Mirrors analysis/stats.py: GroupKFold by user task so a task's two
    states never straddle a fold, a within-group permutation null rather
    than a nominal 0.5 baseline, and Holm correction across defences.
    Zero-delta episodes are excluded for the same reason as elsewhere --
    where the injection never entered context there is nothing to detect.

Usage
    python3 analysis/rq2_masking.py --results_dir agentdojo_matrix_v4_qwen3vl \\
        --layer 31 --attack template_ipi --n_perm 500 \\
        --out analysis_out/rq2_masking_qwen3vl.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit("torch required")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy import stats as sps
except ImportError:
    sps = None

ZERO_TOL = 1e-6


def user_task_group(task_id: str) -> str:
    """Group by USER task so a task's clean and attacked variants never
    straddle a fold -- same rule as analysis/stats.py."""
    m = re.search(r"(.*user_task_\d+)", task_id or "")
    return m.group(1) if m else (task_id or "")


def holm(entries, key_in, key_out):
    have = [e for e in entries if e.get(key_in) is not None]
    m = len(have)
    prev = 0.0
    for rank, e in enumerate(sorted(have, key=lambda x: x[key_in])):
        v = min(1.0, e[key_in] * (m - rank))
        e[key_out] = max(v, prev)
        prev = e[key_out]


def load_states(results_dir, layer, attack):
    """{task_id: {state: vector}} for clean, attacked, and each
    defence's defended_clean / defended_attacked."""
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    cache = {}

    def vec(sid):
        if sid in cache:
            return cache[sid]
        p = os.path.join(results_dir, sid + ".pt")
        v = None
        if os.path.exists(p):
            try:
                turns = torch.load(p, map_location="cpu", weights_only=True)
                vs = [t[layer].float().numpy() for t in turns
                      if isinstance(t, dict) and layer in t]
                v = np.mean(vs, axis=0) if vs else None
            except Exception:
                v = None
        cache[sid] = v
        return v

    states = defaultdict(dict)
    for r in rows:
        if r.get("error"):
            continue
        cond, dn, t = r.get("condition"), r.get("defense_name"), r["task_id"]
        v = vec(r["sample_id"])
        if v is None:
            continue
        if cond == "clean":
            states[t]["clean"] = v
        elif cond == "attacked" and r["attack_name"] == attack:
            states[t]["attacked"] = v
        elif cond == "defended_clean":
            states[t][("dc", dn)] = v
        elif cond == "defended_attacked" and r["attack_name"] == attack:
            states[t][("da", dn)] = v
    return states


def probe(X, y, groups, n_perm, seed=0):
    """Accuracy plus a within-group permutation null. Returns
    (acc, std, p, null_p95). p is (1 + #{null >= observed}) / (1 + n)."""
    n_splits = min(5, len(set(groups)))
    if n_splits < 2 or len(set(y)) < 2:
        return None, None, None, None
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="liblinear", C=1.0))
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="accuracy")
    acc, std = float(scores.mean()), float(scores.std())

    if n_perm <= 0:
        return acc, std, None, None
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    null = []
    for _ in range(n_perm):
        yp = np.array(y, dtype=int).copy()
        for grp in np.unique(g):
            idx = np.where(g == grp)[0]
            yp[idx] = rng.permutation(yp[idx])
        if len(set(yp)) < 2:
            continue
        s = cross_val_score(clf, X, yp, groups=groups, cv=cv, scoring="accuracy")
        null.append(float(s.mean()))
    if not null:
        return acc, std, None, None
    null = np.array(null)
    p = float((1 + np.sum(null >= acc)) / (1 + len(null)))
    return acc, std, p, float(np.percentile(null, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--attack", default="template_ipi")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    states = load_states(a.results_dir, a.layer, a.attack)
    defences = sorted({k[1] for s in states.values() for k in s
                       if isinstance(k, tuple) and k[0] == "da"})

    # Same reachability gate as everywhere else: where the injection
    # never entered context, clean and attacked are byte-identical and
    # there is nothing for any probe to separate.
    reached = [t for t, s in states.items()
               if "clean" in s and "attacked" in s
               and np.linalg.norm(s["attacked"] - s["clean"]) > ZERO_TOL]
    print(f"\n=== RQ2 masking check ({a.results_dir}, attack={a.attack}, "
          f"layer {a.layer}) ===")
    print(f"tasks with a non-zero attack delta: {len(reached)} of {len(states)}")

    rows = []

    # Reference: undefended clean vs attacked. If this is near chance the
    # defended comparison has no resolution and nothing can be concluded.
    X = np.array([v for t in reached for v in (states[t]["clean"], states[t]["attacked"])])
    y = [0, 1] * len(reached)
    g = [user_task_group(t) for t in reached for _ in range(2)]
    acc, std, p, p95 = probe(X, y, g, a.n_perm)
    rows.append({"pair": "UNDEFENDED  clean vs attacked", "defense": None,
                 "n_pairs": len(reached), "acc": acc, "std": std,
                 "p_raw": p, "null_p95": p95})

    for d in defences:
        ts = [t for t in reached
              if ("dc", d) in states[t] and ("da", d) in states[t]]
        if len(ts) < 10:
            continue
        X = np.array([v for t in ts
                      for v in (states[t][("dc", d)], states[t][("da", d)])])
        y = [0, 1] * len(ts)
        g = [user_task_group(t) for t in ts for _ in range(2)]
        acc, std, p, p95 = probe(X, y, g, a.n_perm)
        rows.append({"pair": f"{d}  defended-clean vs defended-attacked",
                     "defense": d, "n_pairs": len(ts), "acc": acc,
                     "std": std, "p_raw": p, "null_p95": p95})

    holm([r for r in rows if r["defense"]], "p_raw", "p_holm")

    print(f"\n{'comparison':52s} {'n':>5s} {'acc':>7s} {'null p95':>9s} {'p':>9s}")
    print("-" * 88)
    for r in rows:
        if r["acc"] is None:
            print(f"{r['pair']:52s} {r['n_pairs']:5d}   insufficient data")
            continue
        p = r.get("p_holm", r.get("p_raw"))
        star = " *" if (p is not None and p < 0.05) else "  "
        print(f"{r['pair']:52s} {r['n_pairs']:5d} {r['acc']:7.3f} "
              f"{(r['null_p95'] or 0):9.3f} "
              f"{'n/a' if p is None else f'{p:.3g}':>9s}{star}")

    ref = rows[0]
    print("\nreading this:")
    if ref["acc"] is not None and (ref.get("p_raw") or 1) >= 0.05:
        print("  The UNDEFENDED pair is not separable above its own null, so")
        print("  this data has no resolution for the defended comparison.")
        print("  Do not interpret the defended rows.")
    else:
        print("  high accuracy -> the injection is STILL represented under the")
        print("     defence; behavioural suppression happens downstream, and the")
        print("     defence should be fragile to variants reaching the same")
        print("     latent region by another surface form.")
        print("  at chance    -> the defence genuinely masks the injection in")
        print("     the representation.")
        print("  Compare each defended row against the undefended reference,")
        print("  not against 0.5 -- grouped folds do not place chance at 0.5.")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"config": vars(a), "n_reached": len(reached), "rows": rows},
                  open(a.out, "w"), indent=2, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
