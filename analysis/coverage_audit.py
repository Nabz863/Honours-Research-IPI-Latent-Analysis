"""
analysis/coverage_audit.py
--------------------------
Ground truth on what has actually been collected.

Reads every result directory on disk and reports, per
(environment, model, attack, defence, condition):
  - row count against the expected task count
  - error rows, grouped by error class
  - whether activation tensors exist and are non-empty
  - provenance: earliest and latest row timestamp per cell, so a cell
    written by two different runs is visible rather than assumed clean

WHY THIS EXISTS RATHER THAN A SPREADSHEET
    Coverage has been tracked conversationally across weeks, and at
    least three claims made from memory turned out wrong: a matrix
    reported complete that was missing an entire attack condition, a
    defence run whose rows had no activations, and a defence cell
    written by two jobs under different configurations with no per-row
    flag to separate them. Anything reported to a supervisor or written
    into a results chapter should come from this, not from recall.

NOT A SUBSTITUTE FOR THE CORRECTNESS AUDIT
    This answers "what exists". It does not answer "is it right".
    A cell can be complete, non-empty and wrong -- that is exactly what
    happened with turn-0 activations, with the published ASR evaluator,
    and with the pre-fix Gemma rows. Use this to scope the correctness
    audit, not to replace it.

Usage
    python3 analysis/coverage_audit.py
    python3 analysis/coverage_audit.py --check_tensors   # slower, reads .pt
    python3 analysis/coverage_audit.py --out analysis_out/coverage.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

EXPECTED = {"agentdojo": 629, "vwa": 40}

# Environment and model are encoded in directory names rather than in the
# rows, so they are parsed here. Anything unrecognised is still reported,
# flagged as unknown, rather than silently dropped.
MODEL_KEYS = ["qwen3vl", "internvl", "llava", "gemma4"]


def classify(dirname: str):
    env = "agentdojo" if dirname.startswith("agentdojo") else (
        "vwa" if dirname.startswith("vwa") else "unknown")
    model = next((m for m in MODEL_KEYS if m in dirname), None)
    if model is None and env == "vwa":
        # early VWA runs predate the model suffix convention
        model = "qwen3vl?"
    return env, model or "unknown"


def audit_dir(d: str, check_tensors: bool):
    p = os.path.join(d, "attack_results.json")
    if not os.path.exists(p):
        return None
    try:
        rows = json.load(open(p))
    except Exception as e:
        return {"dir": d, "error": f"unreadable: {type(e).__name__}"}

    env, model = classify(os.path.basename(d))
    cells = defaultdict(lambda: {"n": 0, "errors": Counter(), "asr": 0,
                                 "ua": 0, "ua_scored": 0})

    for r in rows:
        key = (r.get("attack_name"), r.get("defense_name") or "none",
               r.get("condition"))
        c = cells[key]
        c["n"] += 1
        if r.get("error"):
            c["errors"][str(r["error"])[:40]] += 1
        if r.get("asr_success") is True:
            c["asr"] += 1
        if r.get("utility_success") is not None:
            c["ua_scored"] += 1
            if r["utility_success"] is True:
                c["ua"] += 1

    tensors = None
    if check_tensors:
        import torch
        pts = glob.glob(os.path.join(d, "*.pt"))
        pts = [x for x in pts if "activations_cache" not in x]
        empty = 0
        for x in pts[:200]:                      # sample, not exhaustive
            try:
                t = torch.load(x, map_location="cpu", weights_only=True)
                if not t:
                    empty += 1
            except Exception:
                empty += 1
        tensors = {"files": len(pts), "sampled": min(len(pts), 200),
                   "empty_or_unreadable": empty}

    return {"dir": os.path.basename(d), "env": env, "model": model,
            "rows": len(rows), "cells": cells, "tensors": tensors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check_tensors", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(a.root, "*"))
                  if os.path.isdir(d)
                  and os.path.exists(os.path.join(d, "attack_results.json")))

    report = []
    for d in dirs:
        r = audit_dir(d, a.check_tensors)
        if r:
            report.append(r)

    print(f"{'directory':36s} {'env':10s} {'model':10s} {'rows':>6s}")
    print("-" * 68)
    for r in report:
        if "error" in r:
            print(f"{r['dir']:36s} {r['error']}")
            continue
        print(f"{r['dir']:36s} {r['env']:10s} {r['model']:10s} {r['rows']:6d}")
        if r["tensors"]:
            t = r["tensors"]
            flag = "  <-- EMPTY TENSORS" if t["empty_or_unreadable"] else ""
            print(f"{'':36s} tensors: {t['files']} files, "
                  f"{t['empty_or_unreadable']}/{t['sampled']} empty{flag}")

    print("\n\n=== cells ===")
    print(f"{'directory':30s} {'attack':22s} {'defence':26s} {'cond':18s} "
          f"{'n':>5s} {'ASR':>5s} {'UA':>9s} {'err':>5s}")
    print("-" * 126)
    for r in report:
        if "error" in r:
            continue
        exp = EXPECTED.get(r["env"])
        for (atk, dfn, cond), c in sorted(r["cells"].items(), key=lambda x: str(x[0])):
            gap = ""
            if exp and c["n"] < exp:
                gap = f"  <-- {exp - c['n']} short of {exp}"
            ua = f"{c['ua']}/{c['ua_scored']}" if c["ua_scored"] else "n/a"
            print(f"{r['dir'][:30]:30s} {str(atk)[:22]:22s} {str(dfn)[:26]:26s} "
                  f"{str(cond)[:18]:18s} {c['n']:5d} {c['asr']:5d} {ua:>9s} "
                  f"{sum(c['errors'].values()):5d}{gap}")

    print("\n=== error classes ===")
    allerr = Counter()
    for r in report:
        if "error" in r:
            continue
        for c in r["cells"].values():
            allerr.update(c["errors"])
    for e, n in allerr.most_common(12):
        print(f"  {n:5d}  {e}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        ser = []
        for r in report:
            if "error" in r:
                ser.append(r); continue
            ser.append({**r, "cells": {
                "|".join(str(x) for x in k): {**v, "errors": dict(v["errors"])}
                for k, v in r["cells"].items()}})
        json.dump(ser, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
