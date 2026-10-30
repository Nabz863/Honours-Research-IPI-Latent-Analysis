#!/usr/bin/env python3
"""analysis/compliance_bootstrap.py -- uncertainty of the compliance bounds
(Proposition 1, Table "compliance") on AgentDojo, undefended. CPU, read-only.

Uses agentdojo_behaviour.py's own rules and functions (exclusion union over the
directories given, payload-text exposure matcher, clean utility once per user
task), so the point estimates reproduce that table exactly. Adds:
  * a user-task cluster bootstrap (B resamples of user tasks, the same draw for
    every model and attack, since the models ran the same tasks): 95% percentile
    intervals for ASR_E, c, the lower bound ASR_E and the upper bound
    min(1, ASR_E + 1 - c);
  * for every pair of (model, attack) cells, the fraction of resamples in which
    one cell's lower bound exceeds the other's upper bound (the non-overlap the
    paper reads off the point estimates);
  * sensitivity to c: how much lower the true competence of the upper-bound cell
    would have to be (c is estimated by clean utility, assuming the attacker's
    goal is no harder than the user's) before that non-overlap disappears.

  python3 analysis/compliance_bootstrap.py final_agentdojo_gemma4 final_agentdojo_qwen3vl \
      --B 2000 --out analysis_out/compliance_bootstrap.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import agentdojo_behaviour as ab  # noqa: E402


def episodes(run, defaults, excluded):
    """[(user_key, attack, exposed, success, clean_utility or None)] for the
    undefended attacked episodes the behaviour table counts (scored, payload in
    the cache, transcript present, markers found)."""
    rows = [r for r in run["rows"] if ab.pair_key(r) not in excluded]
    clean_by_user = {}
    for r in rows:
        if r.get("condition") == "clean" and not r.get("error"):
            clean_by_user.setdefault(ab.user_key(r), r.get("utility_success"))
    out = []
    for r in rows:
        # the behaviour table's undefended cell: attacked OR defended_attacked
        # rows whose defence is None / '' / 'none'
        if r.get("condition") not in (ab.ATTACKED, ab.D_ATT) or r.get("error") \
                or (r.get("defense_name") or "none") != "none":
            continue
        if r.get("asr_success") is None:
            continue
        entry = run["payloads"].get((r.get("attack_name"), r.get("task_id")))
        if not entry:
            continue
        marks = ab.payload_markers(entry, r.get("suite_name"), defaults)
        txt = ab.transcript_text(run["dir"] / f"{r['sample_id']}.transcript.json.gz")
        if txt is None or not marks:
            continue
        cu = clean_by_user.get(ab.user_key(r))
        out.append((ab.user_key(r), r.get("attack_name"), any(m in txt for m in marks),
                    bool(r["asr_success"]), None if cu is None else bool(cu)))
    return out


def bounds(eps):
    """(asr_e, c, lower, upper, n_exposed) from episode tuples, as the table."""
    ex = [e for e in eps if e[2]]
    if not ex:
        return None
    asr_e = sum(e[3] for e in ex) / len(ex)
    cs = [e[4] for e in ex if e[4] is not None]
    if not cs:
        return None
    c = sum(cs) / len(cs)
    return asr_e, c, asr_e, min(1.0, asr_e + 1 - c), len(ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis_out/compliance_bootstrap.json")
    a = ap.parse_args()
    runs = [ab.load_dir(Path(d)) for d in a.dirs]
    defaults = ab.injection_defaults()
    excluded = set()
    for run in runs:
        for r in run["rows"]:
            if r.get("condition") == "clean" and not r.get("error") and r.get("asr_success") is True:
                excluded.add(ab.pair_key(r))

    cells = {}                                   # (model, attack) -> {user_key: [episodes]}
    for run in runs:
        for e in episodes(run, defaults, excluded):
            cells.setdefault((run["name"], e[1]), defaultdict(list))[e[0]].append(e)
    users = sorted({u for g in cells.values() for u in g}, key=str)
    rng = np.random.default_rng(a.seed)
    draws = [rng.integers(0, len(users), len(users)) for _ in range(a.B)]

    res = {}
    boot = {}
    for key, g in sorted(cells.items()):
        point = bounds([e for es in g.values() for e in es])
        if point is None:                        # nothing exposed, or no clean utility
            res[f"{key[0]}|{key[1]}"] = {"note": "bounds undefined (no exposed episode with clean utility)"}
            continue
        reps = np.full((a.B, 4), np.nan)         # row-aligned with the draws; NaN = undefined
        for b_i, idx in enumerate(draws):
            b = bounds([e for i in idx for e in g.get(users[i], [])])
            if b is not None:
                reps[b_i] = b[:4]
        boot[key] = reps
        ok = ~np.isnan(reps[:, 0])
        ci = lambda j: [float(np.percentile(reps[ok, j], 2.5)), float(np.percentile(reps[ok, j], 97.5))]
        res[f"{key[0]}|{key[1]}"] = {
            "n_exposed": point[4], "asr_e": point[0], "c": point[1], "lower": point[2], "upper": point[3],
            "asr_e_ci95": ci(0), "c_ci95": ci(1), "lower_ci95": ci(2), "upper_ci95": ci(3),
            "n_boot_defined": int(ok.sum()), "n_user_tasks": len(g)}

    # non-overlap of intervals between cells, under the joint (same-task) draws
    pairs = {}
    keys = sorted(boot)
    for hi in keys:
        for lo in keys:
            if hi == lo:
                continue
            p_hi, p_lo = res[f"{hi[0]}|{hi[1]}"], res[f"{lo[0]}|{lo[1]}"]
            if p_hi["lower"] <= p_lo["upper"]:
                continue                         # no point-estimate non-overlap to test
            # an undefined replicate in either cell counts as overlap (conservative)
            with np.errstate(invalid="ignore"):
                frac = float(np.mean(np.nan_to_num(boot[hi][:, 2] > boot[lo][:, 3], nan=False)))
            # c of the 'lo' cell enters its upper bound as 1 - c: the margin is how
            # much lower its true competence can be before the intervals touch
            margin = p_hi["lower"] - p_lo["upper"]
            pairs[f"{hi[0]}|{hi[1]} > {lo[0]}|{lo[1]}"] = {
                "margin": margin, "frac_boot_nonoverlap": frac,
                "c_lo_breaks_at": p_lo["c"] - margin}

    print(f"excluded pairs: {len(excluded)}; user tasks: {len(users)}; bootstrap B={a.B}, seed {a.seed}\n")
    print(f"{'cell':52s} {'n_E':>4s} {'ASR_E [95% CI]':>22s} {'c [95% CI]':>22s} {'bounds [lower CI] / [upper CI]':>40s}")
    for k, v in res.items():
        if "note" in v:
            print(f"{k:52s} {v['note']}")
            continue
        print(f"{k:52s} {v['n_exposed']:4d} {v['asr_e']:6.3f} [{v['asr_e_ci95'][0]:.3f},{v['asr_e_ci95'][1]:.3f}] "
              f"{v['c']:6.3f} [{v['c_ci95'][0]:.3f},{v['c_ci95'][1]:.3f}]   "
              f"[{v['lower']:.3f}, {v['upper']:.3f}]  [{v['lower_ci95'][0]:.3f},{v['lower_ci95'][1]:.3f}] / "
              f"[{v['upper_ci95'][0]:.3f},{v['upper_ci95'][1]:.3f}]")
    print("\nnon-overlap (point estimates) and how robust it is:")
    for k, v in pairs.items():
        print(f"  {k}: margin {v['margin']:.3f}; holds in {100 * v['frac_boot_nonoverlap']:.1f}% of resamples; "
              f"breaks if the second cell's true c <= {v['c_lo_breaks_at']:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"B": a.B, "seed": a.seed, "excluded": len(excluded), "cells": res, "nonoverlap": pairs},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
