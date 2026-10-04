#!/usr/bin/env python3
"""
analysis/cell_report.py
-----------------------
Per-cell integrity report for a matrix output dir.

WHY THIS EXISTS
    Job 61500 exited 0:0, SLURM recorded COMPLETED, and it wrote 600 rows to
    attack_results.json. Every episode in the log tail read

        ASR=None, UA=None, ep=0.6s

    A 0.6-second episode did no forward pass. So the run produced 600 rows of
    nothing and reported success. Row COUNT is not evidence a cell ran; this
    script asks the questions that are:

        - how many rows carry a real ASR / UA verdict, not None
        - how many carry non-empty activations (the thing RQ1/RQ2 need)
        - how many carry an error, and which errors
        - what the episode wall time looks like per cell

    A cell with rows but no activations is worse than a missing cell, because
    it looks done to any resume logic that only checks for presence.

USAGE
    python3 analysis/cell_report.py vwa_adv_qwen3vl_def3
    python3 analysis/cell_report.py vwa_adv_qwen3vl_def3 --min_seconds 2.0
    python3 analysis/cell_report.py <dir> --show_keys      # if fields differ
    python3 analysis/cell_report.py <dir> --json           # machine-readable

Exit status is 1 if any cell is hollow, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

# Field-name candidates, in preference order. The runner's exact names may
# differ; --show_keys prints what is actually there.
F_ATTACK = ("attack_name", "attack", "attack_id")
F_DEFENSE = ("defense_name", "defence_name", "defense", "defence")
F_TASK = ("task_id", "task", "task_name", "config_file")
F_ASR = ("asr_success", "asr", "attack_success")
F_STRICT = ("strict_asr",)
F_UA = ("utility_success", "ua_success", "utility", "ua")
F_ERR = ("error", "err", "exception")
F_SECS = ("episode_time_s", "episode_seconds", "elapsed", "ep_seconds",
          "duration", "seconds")
F_STEPS = ("num_steps", "n_steps", "steps")
F_MEM = ("peak_cuda_memory_gb", "peak_cuda_gb", "peak_memory_gb")
# Activations are NOT inline in the row. They are written as one tensor file
# per episode alongside the results JSON -- confirmed on
# vwa_adv_qwen3vl_def2: 800 .pt files for 800 rows. An earlier version of
# this script looked for a `turn_activations` list in the row, found nothing,
# and reported acts=0 for every cell including healthy ones.
F_ACTS = ("turn_activations", "activations", "step_activations")
MIN_TENSOR_BYTES = 1024          # below this, the file exists but holds nothing


def pick(row: dict, names: tuple[str, ...], default=None):
    for n in names:
        if n in row:
            return row[n]
    return default


def load_rows(d: Path) -> tuple[list[dict], list[str]]:
    """Read every plausible result file under d. Handles JSON and JSONL."""
    rows, seen = [], []
    cands = sorted(
        list(d.glob("attack_results.json"))
        + list(d.glob("*.json"))
        + list(d.glob("*.jsonl"))
        + list(d.glob("**/attack_results.json"))
        + list(d.glob("**/*.jsonl"))
    )
    for p in dict.fromkeys(cands):            # dedupe, keep order
        try:
            raw = p.read_text()
        except Exception:                                     # noqa: BLE001
            continue
        got: list[dict] = []
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                got = [r for r in obj if isinstance(r, dict)]
            elif isinstance(obj, dict):
                # a wrapper: find the first list-of-dicts value
                for k in ("rows", "results", "episodes", "data"):
                    v = obj.get(k)
                    if isinstance(v, list):
                        got = [r for r in v if isinstance(r, dict)]
                        break
                else:
                    # or a mapping of id -> row
                    vals = [v for v in obj.values() if isinstance(v, dict)]
                    if vals and any(pick(v, F_ASR, "?") != "?" for v in vals):
                        got = vals
        except json.JSONDecodeError:
            for ln in raw.splitlines():                       # JSONL fallback
                ln = ln.strip()
                if not ln.startswith("{"):
                    continue
                try:
                    got.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        if got:
            rows.extend(got)
            seen.append(f"{p.name} ({len(got)} rows)")
    return rows, seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--min_seconds", type=float, default=2.0,
                    help="episodes faster than this did no real work "
                         "(default 2.0; a 15-step VLM episode takes tens of "
                         "seconds)")
    ap.add_argument("--show_keys", action="store_true",
                    help="print the first row's field names and exit")
    ap.add_argument("--json", action="store_true", dest="as_json")
    a = ap.parse_args()

    d = Path(a.out_dir)
    if not d.exists():
        print(f"no such dir: {d}", file=sys.stderr)
        return 2

    rows, files = load_rows(d)
    if not rows:
        print(f"NO ROWS PARSED from {d}", file=sys.stderr)
        print("Files present:", file=sys.stderr)
        for p in sorted(d.rglob("*"))[:40]:
            if p.is_file():
                print(f"  {p} ({p.stat().st_size} B)", file=sys.stderr)
        return 2

    if a.show_keys:
        print("files:", ", ".join(files))
        print(f"\n{len(rows)} rows. First row's fields:\n")
        for k, v in rows[0].items():
            t = type(v).__name__
            n = f" len={len(v)}" if isinstance(v, (list, dict, str)) else ""
            print(f"  {k:<28} {t}{n}")
        return 0

    # Tensor files, indexed by stem, so a row can be matched to its capture.
    tensors: dict[str, int] = {}
    for p in d.rglob("*.pt"):
        try:
            tensors[p.stem] = p.stat().st_size
        except OSError:
            tensors[p.stem] = 0

    def tensor_bytes(sid: str) -> Optional[int]:
        """Size of this episode's activation file, or None if there isn't one."""
        if sid in tensors:
            return tensors[sid]
        for stem, size in tensors.items():   # naming may carry a suffix
            if stem.startswith(sid) or sid in stem:
                return size
        return None

    cells: dict[tuple, dict] = collections.defaultdict(
        lambda: {"n": 0, "asr": 0, "strict": 0, "ua": 0, "acts": 0,
                 "empty_acts": 0, "no_acts": 0, "err": 0, "fast": 0,
                 "zero_steps": 0, "secs": [], "mem": []})
    for r in rows:
        k = (str(pick(r, F_ATTACK, "?")), str(pick(r, F_DEFENSE, "none")))
        c = cells[k]
        c["n"] += 1
        if pick(r, F_ASR) is not None:
            c["asr"] += 1
        if pick(r, F_STRICT) is not None:
            c["strict"] += 1
        if pick(r, F_UA) is not None:
            c["ua"] += 1
        if pick(r, F_ERR):
            c["err"] += 1

        # Activations: inline list if this runner uses one, else the tensor file.
        inline = pick(r, F_ACTS)
        if inline:
            c["acts"] += 1
        else:
            sid = str(pick(r, ("sample_id",), ""))
            nb = tensor_bytes(sid) if sid else None
            if nb is None:
                c["no_acts"] += 1
            elif nb < MIN_TENSOR_BYTES:
                c["empty_acts"] += 1      # file exists, holds nothing
            else:
                c["acts"] += 1

        s = pick(r, F_SECS)
        if isinstance(s, (int, float)):
            c["secs"].append(float(s))
            if s < a.min_seconds:
                c["fast"] += 1
        st = pick(r, F_STEPS)
        if isinstance(st, int) and st == 0:
            c["zero_steps"] += 1          # the agent never acted
        m = pick(r, F_MEM)
        if isinstance(m, (int, float)):
            c["mem"].append(float(m))

    hollow = []
    unscored = []
    partial = []
    report = []
    for k in sorted(cells):
        c = cells[k]
        med = (sorted(c["secs"])[len(c["secs"]) // 2] if c["secs"] else None)
        medmem = (sorted(c["mem"])[len(c["mem"]) // 2] if c["mem"] else None)
        # HOLLOW: rows exist but nothing usable came out of them at all.
        is_hollow = c["n"] > 0 and c["acts"] == 0 and c["asr"] == 0
        # UNSCORED: the episodes RAN -- real wall time, real CUDA, activations
        # on disk -- but not one produced a behavioural verdict. This is the
        # `no cached payload` signature: the agent ran against the CLEAN image
        # because the adversarial payload was missing, so the runner refused to
        # score it. More dangerous than hollow, because the activations exist
        # and look legitimate. Any RQ1/RQ2 selection by attack name, or by
        # presence of a tensor file, silently reads these as attacked episodes
        # when no attack was delivered. The only sound filter is `error`.
        is_unscored = (not is_hollow) and c["n"] > 0 and c["asr"] == 0 \
            and c["acts"] > 0
        # PARTIAL: some episodes produced a verdict and some did not. Easy to
        # miss, because the cell has data and looks fine in aggregate.
        is_partial = (not is_hollow) and (not is_unscored) \
            and c["n"] > 0 and 0 < c["asr"] < c["n"]
        if is_hollow:
            hollow.append(k)
        elif is_unscored:
            unscored.append((k, c["acts"], c["n"]))
        elif is_partial:
            partial.append((k, c["asr"], c["n"]))
        report.append({
            "attack": k[0], "defense": k[1], "n": c["n"],
            "asr_scored": c["asr"], "strict_asr_scored": c["strict"],
            "ua_scored": c["ua"],
            "with_activations": c["acts"],
            "empty_activation_files": c["empty_acts"],
            "missing_activation_files": c["no_acts"],
            "errors": c["err"], "faster_than_min": c["fast"],
            "zero_step_episodes": c["zero_steps"],
            "median_seconds": med, "median_peak_cuda_gb": medmem,
            "hollow": is_hollow, "unscored": is_unscored,
            "partial": is_partial,
        })

    errs = collections.Counter(
        str(pick(r, F_ERR))[:90] for r in rows if pick(r, F_ERR))

    if a.as_json:
        print(json.dumps({"dir": str(d), "files": files, "rows": len(rows),
                          "cells": report, "hollow_cells": len(hollow),
                          "errors": errs.most_common(20)}, indent=2))
        return 1 if hollow else 0

    print(f"{d}  --  {len(rows)} rows from: {', '.join(files)}")
    print(f"{len(tensors)} activation file(s) found under this directory\n")
    print(f"{'attack':<24}{'defense':<24}{'n':>4}{'asr':>4}{'ua':>4}"
          f"{'acts':>5}{'err':>5}{'0st':>4}{'med_s':>7}{'GB':>6}")
    print("-" * 95)
    for r in report:
        flag = ("  HOLLOW" if r["hollow"]
                else "  UNSCORED (ran, no verdict)" if r["unscored"]
                else f"  PARTIAL {r['asr_scored']}/{r['n']}" if r["partial"]
                else "")
        med = f"{r['median_seconds']:.1f}" if r["median_seconds"] is not None else "-"
        gb = f"{r['median_peak_cuda_gb']:.1f}" if r["median_peak_cuda_gb"] is not None else "-"
        print(f"{r['attack'][:23]:<24}{r['defense'][:23]:<24}"
              f"{r['n']:>4}{r['asr_scored']:>4}{r['ua_scored']:>4}"
              f"{r['with_activations']:>5}{r['errors']:>5}"
              f"{r['zero_step_episodes']:>4}{med:>7}{gb:>6}{flag}")

    tot = {k: sum(r[k] for r in report) for k in
           ("n", "asr_scored", "ua_scored", "with_activations", "errors",
            "zero_step_episodes", "empty_activation_files",
            "missing_activation_files")}
    print("-" * 95)
    print(f"{'TOTAL':<48}{tot['n']:>4}{tot['asr_scored']:>4}"
          f"{tot['ua_scored']:>4}{tot['with_activations']:>5}"
          f"{tot['errors']:>5}{tot['zero_step_episodes']:>4}")
    if tot["empty_activation_files"] or tot["missing_activation_files"]:
        print(f"\nactivation files: {tot['missing_activation_files']} missing, "
              f"{tot['empty_activation_files']} present but under "
              f"{MIN_TENSOR_BYTES} bytes (i.e. empty)")

    if errs:
        print("\nerrors:")
        for e, n in errs.most_common(15):
            print(f"  {n:5d}  {e}")
    else:
        print("\nerrors: none recorded "
              "-- note that a silent no-op leaves no error either")

    if unscored:
        print(f"\n{len(unscored)} UNSCORED CELL(S): the episodes RAN -- real "
              f"wall time, real CUDA, activations on disk -- but not one "
              f"produced a behavioural verdict.")
        for k, acts, n in unscored:
            print(f"  {k[0]}  +  {k[1]}   {acts}/{n} have activations, "
                  f"0 scored")
        print("\n  THIS IS A CONTAMINATION RISK, not a gap. The usual cause is "
              "a missing\n  adversarial payload: the agent ran against the "
              "CLEAN image, so the\n  activations are real but the attack was "
              "never delivered. Selecting\n  attacked episodes by attack name, "
              "or by presence of a tensor file,\n  reads these as attacked. "
              "Filter on `error` instead.")

    if partial:
        print(f"\n{len(partial)} PARTIAL CELL(S): some episodes produced a "
              f"verdict and some did not. Easy to miss -- the cell has data "
              f"and looks fine in aggregate, but any rate computed over it "
              f"has a different denominator than the cells beside it.")
        for k, got, n in partial:
            print(f"  {k[0]}  +  {k[1]}   {got}/{n} scored")

    if hollow:
        print(f"\n{len(hollow)} HOLLOW CELL(S): rows written, but no "
              f"activations and no ASR verdict.")
        for k in hollow:
            print(f"  {k[0]}  +  {k[1]}")
        print("\nWhether these are retried automatically depends on the "
              "runner's resume check. run_vwa_adv_curated_matrix uses\n"
              "    def done(sid): return sid in index and "
              "results[index[sid]].get('error') is None\n"
              "so a row CARRYING AN ERROR is retried and the directory must "
              "NOT be deleted. A hollow row with error=None is a different "
              "case -- that one would be skipped forever.")
        n_err = sum(r["errors"] for r in report if r["hollow"])
        n_row = sum(r["n"] for r in report if r["hollow"])
        if n_err == n_row:
            print(f"  All {n_row} hollow rows carry an error -> resubmit "
                  f"retries them. Do not delete.")
        else:
            print(f"  WARNING: {n_row - n_err} of {n_row} hollow rows have "
                  f"error=None -> those would be SKIPPED on resubmit and "
                  f"must be removed from the results file.")
        return 1

    if not (partial or unscored):
        print("\nno hollow, unscored or partial cells")
    return 1 if (partial or unscored) else 0


if __name__ == "__main__":
    raise SystemExit(main())
