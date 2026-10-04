#!/usr/bin/env python3
"""
analysis/encoding_confound_test.py
----------------------------------
Is the RQ2 VWA-Adv encoding result a representational finding, or an artefact
of episodes that never took a step?

THE RESULT UNDER TEST
    rq2_masking.py on vwa_adv_qwen3vl_defended, attack=comet: probe accuracy
    separating defended-clean from defended-attacked rises from 0.846
    (undefended) to 0.961 under spotlighting_encoding -- reported as "defences
    make the injection MORE linearly detectable".

THE CONFOUND (audit 7.3)
    Under encoding most episodes never take a step, and the two classes stall
    at different rates: defended-clean 20/40 zero-step, defended-attacked
    31/40. States are mean-over-turns, so a zero-step episode is a different
    object from a multi-step one. A probe can separate "stalled" from "not
    stalled" -- a behavioural outcome -- and that would read as separating
    "injected" from "clean".

DESIGN
    Reuses rq2_masking's own probe(), holm(), user_task_group() and ZERO_TOL
    by import, so nothing about the statistics is reimplemented. The only
    difference is one filter in load_states: rows with num_steps < min_steps
    are dropped. The comparison is run twice, from the same code path:

      min_steps = 0   MUST reproduce rq2_masking.py's reported numbers. If it
                      does not, this script is wrong and its second run means
                      nothing. The script prints the reproduction check.
      min_steps = 1   the same comparison with zero-step episodes removed.

    Both runs are written to one JSON so they can never be quoted separately.

A THIRD RUN -- STALL-MATCHED
    Excluding zero-step episodes is likely to leave too few pairs: with the
    attacked side stalling in 31 of 40, at most 9 pairs can have both sides
    active, under rq2_masking's own floor of 10. So a third run MATCHES on
    stalling instead: it keeps only pairs whose two episodes are in the same
    stall class (both zero-step, or both >=1 step). That removes the confound
    while keeping most of the data, and is the column to read first.

READING THE RESULT
    - encoding accuracy stays high with zero-step removed -> the finding
      survives its most obvious alternative and is STRONGER for it;
    - it falls toward the undefended reference -> the rise was produced by
      encoding's utility collapse, not by representation, and must not be
      reported as a representational result;
    - too few pairs survive the filter (< 10, rq2_masking's own floor) -> the
      data cannot distinguish the two readings; say so rather than choose.

USAGE
    python3 analysis/encoding_confound_test.py \\
        --results_dir vwa_adv_qwen3vl_defended --attack comet \\
        --out analysis_out/encoding_confound_comet.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_masking import probe, holm, user_task_group, ZERO_TOL   # noqa: E402


def load_states(results_dir, layer, attack, min_steps):
    """rq2_masking.load_states, plus a num_steps filter. Everything else identical."""
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    cache = {}
    dropped = defaultdict(int)

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
    steps = defaultdict(dict)                 # parallel to states: num_steps per state
    for r in rows:
        if r.get("error"):
            continue
        cond, dn, t = r.get("condition"), r.get("defense_name"), r["task_id"]
        if min_steps:
            ns = r.get("num_steps")
            if ns is None or ns < min_steps:
                # count only rows this comparison would otherwise have used
                if cond in ("clean", "defended_clean") or r.get("attack_name") == attack:
                    dropped[(cond, dn)] += 1
                continue
        v = vec(r["sample_id"])
        if v is None:
            continue
        key = None
        if cond == "clean":
            key = "clean"
        elif cond == "attacked" and r["attack_name"] == attack:
            key = "attacked"
        elif cond == "defended_clean":
            key = ("dc", dn)
        elif cond == "defended_attacked" and r["attack_name"] == attack:
            key = ("da", dn)
        if key is not None:
            states[t][key] = v
            steps[t][key] = r.get("num_steps")
    return states, steps, dict(dropped)


def compare(results_dir, layer, attack, n_perm, min_steps, stall_matched=False):
    """rq2_masking.main's comparison logic, verbatim apart from printing.

    stall_matched: keep only defended pairs whose two episodes fall in the SAME
    stall class -- both zero-step or both >=1 step. Matching removes the
    confound without discarding the stalled episodes, which excluding them
    does: with the attacked side stalling in 31 of 40, at most 9 pairs can
    have both sides active, under rq2_masking's own floor of 10.
    """
    states, steps, dropped = load_states(results_dir, layer, attack, min_steps)
    defences = sorted({k[1] for s in states.values() for k in s
                       if isinstance(k, tuple) and k[0] == "da"})
    reached = [t for t, s in states.items()
               if "clean" in s and "attacked" in s
               and np.linalg.norm(s["attacked"] - s["clean"]) > ZERO_TOL]

    rows = []
    X = np.array([v for t in reached for v in (states[t]["clean"], states[t]["attacked"])])
    y = [0, 1] * len(reached)
    g = [user_task_group(t) for t in reached for _ in range(2)]
    acc, std, p, p95 = probe(X, y, g, n_perm)
    rows.append({"pair": "UNDEFENDED clean vs attacked", "defense": None,
                 "n_pairs": len(reached), "acc": acc, "std": std,
                 "p_raw": p, "null_p95": p95})

    for d in defences:
        ts = [t for t in reached
              if ("dc", d) in states[t] and ("da", d) in states[t]]
        if stall_matched:
            ts = [t for t in ts
                  if ((steps[t].get(("dc", d)) or 0) == 0)
                  == ((steps[t].get(("da", d)) or 0) == 0)]
        if len(ts) < 10:
            rows.append({"pair": f"{d} defended-clean vs defended-attacked",
                         "defense": d, "n_pairs": len(ts), "acc": None,
                         "std": None, "p_raw": None, "null_p95": None,
                         "note": "fewer than 10 pairs -- rq2_masking's own floor"})
            continue
        X = np.array([v for t in ts for v in (states[t][("dc", d)], states[t][("da", d)])])
        y = [0, 1] * len(ts)
        g = [user_task_group(t) for t in ts for _ in range(2)]
        acc, std, p, p95 = probe(X, y, g, n_perm)
        rows.append({"pair": f"{d} defended-clean vs defended-attacked",
                     "defense": d, "n_pairs": len(ts), "acc": acc, "std": std,
                     "p_raw": p, "null_p95": p95})

    holm([r for r in rows if r["defense"] and r["p_raw"] is not None], "p_raw", "p_holm")
    return {"min_steps": min_steps, "stall_matched": stall_matched,
            "n_tasks_reached": len(reached),
            "dropped_rows": {f"{c}|{d}": n for (c, d), n in dropped.items()},
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--attack", default="comet")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--min_steps", type=int, default=1)
    ap.add_argument("--reference", default=None,
                    help="rq2_masking JSON to check the min_steps=0 run against")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = compare(a.results_dir, a.layer, a.attack, a.n_perm, 0)
    filt = compare(a.results_dir, a.layer, a.attack, a.n_perm, a.min_steps)
    match = compare(a.results_dir, a.layer, a.attack, a.n_perm, 0, stall_matched=True)

    by = lambda res: {r["pair"]: r for r in res["rows"]}
    b, f, m = by(base), by(filt), by(match)

    print(f"\n=== encoding confound test: {a.results_dir}, attack={a.attack}, "
          f"layer {a.layer}, n_perm {a.n_perm} ===\n")
    print(f"{'comparison':40s} {'all eps':>17s}   {'>=' + str(a.min_steps) + ' step':>17s}   {'stall-matched':>17s}")
    print(f"{'':40s} {'n':>5s} {'acc':>6s} {'p':>5s}   {'n':>5s} {'acc':>6s} {'p':>5s}   {'n':>5s} {'acc':>6s} {'p':>5s}")
    print("-" * 102)

    def cell(r):
        if r is None:
            return f"{'-':>5s} {'-':>6s} {'-':>5s}"
        if r["acc"] is None:
            return f"{r['n_pairs']:5d} {'<10':>6s} {'-':>5s}"
        p = r.get("p_holm", r.get("p_raw"))
        return f"{r['n_pairs']:5d} {r['acc']:6.3f} {('n/a' if p is None else f'{p:.2g}'):>5s}"

    for pair in b:
        print(f"{pair[:40]:40s} {cell(b[pair])}   {cell(f.get(pair))}   {cell(m.get(pair))}")

    print(f"\nrows removed by the step filter: {filt['dropped_rows'] or 'none'}")

    check = None
    if a.reference and os.path.exists(a.reference):
        try:
            ref = json.load(open(a.reference))
            ref_rows = ref if isinstance(ref, list) else ref.get("rows", [])
            ref_by = {}
            for r in ref_rows:
                key = (r.get("defense"), r.get("n_pairs"))
                ref_by[key] = r.get("acc")
            mism, matched = [], 0
            for r in base["rows"]:
                ra = ref_by.get((r["defense"], r["n_pairs"]))
                if ra is None or r["acc"] is None:
                    continue
                matched += 1
                if abs(ra - r["acc"]) > 1e-6:
                    mism.append((r["pair"], ra, r["acc"]))
            check = {"reference": a.reference, "rows_compared": matched,
                     "mismatches": mism}
            print("\nREPRODUCTION CHECK against", a.reference)
            if matched == 0:
                # Zero comparisons is NOT a pass. An earlier version reported
                # "passed" here, which a reference in an unexpected format
                # would have triggered silently.
                print("  INCONCLUSIVE -- no row of the reference could be matched to")
                print("  this run (keyed on defence + n_pairs). Compare the 'all eps'")
                print("  column against the reported numbers by eye.")
            elif mism:
                print("  FAILED -- the unfiltered run does not reproduce the reported numbers.")
                print("  Do not interpret the filtered column until this is explained:")
                for pair, ra, ba in mism:
                    print(f"    {pair}: reported {ra:.3f}, this run {ba:.3f}")
            else:
                print(f"  passed -- {matched} comparison(s) reproduce the reported accuracies.")
        except Exception as e:                                   # noqa: BLE001
            print(f"\n(could not read reference {a.reference}: {e})")
    else:
        print("\nNo --reference given: compare the 'all eps' column against the "
              "reported 0.846 / 0.961 by eye before reading the filtered column.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"config": vars(a), "unfiltered": base, "filtered": filt,
               "stall_matched": match,
               "reproduction_check": check}, open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
