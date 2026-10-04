#!/usr/bin/env python3
"""
analysis/at_threshold_band.py
-----------------------------
Attention Tracker at a second threshold without re-running every episode
(P2a.1, decision 4 Oct).

Attention Tracker withholds a tool result whose focus score is <= the
threshold. Under greedy decoding an episode therefore plays out identically
at two thresholds lo < hi unless one of its tool results scores in (lo, hi]:
below lo a result is withheld at both, above hi it is shown at both, and
nothing differs before the first result in the band. So the hi run's rows
are also the lo run's rows, except for episodes with a score in the band;
only those are re-run at lo.

    select   writes the sample ids to re-run (one per line) for
             run_attack_matrix.py --only_samples
    merge    writes the lo condition's full row set: the re-run rows plus the
             hi rows outside the band, relabelled, each carrying
             `carried_from` so the provenance is visible

Usage (login node is fine: JSON only):
    python3 analysis/at_threshold_band.py select <out_dir> --lo 0.30 --hi 0.50 > at_t30_samples.txt
    python3 analysis/at_threshold_band.py merge  <out_dir> --lo 0.30 --hi 0.50 > analysis_out/at_t30_rows.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HI_NAME = "attention_tracker"


def lo_name(lo: float) -> str:
    return f"attention_tracker_t{int(round(lo * 100)):02d}"


def to_lo(sid: str, lo: float) -> str:
    return sid.replace(f"_{HI_NAME}_", f"_{lo_name(lo)}_", 1)


EPS = 1e-6     # scores are stored rounded to 6 decimals; widen the band, never narrow it


def in_band(row, lo, hi) -> bool:
    return any(lo - EPS < float(s) <= hi + EPS for s in (row.get("defense_scores") or []))


def resolve(row, rows_by_sid):
    """A deduplicated defended-clean row stands for its representative."""
    rep = row.get("deduplicated_to")
    return rows_by_sid.get(rep, {"error": "missing representative"}) if rep else row


def expected_hi_sids(rows):
    """Every attention_tracker sample id the 0.5 pass should have produced:
    defended-clean per pair, and defended-attacked per (attack, pair) for each
    attack that has attacked rows."""
    tasks = sorted({r["task_id"] for r in rows if r.get("condition") == "clean"})
    attacks = sorted({r["attack_name"] for r in rows if r.get("condition") == "attacked"})
    out = {f"defended_{HI_NAME}_{t}" for t in tasks}
    out |= {f"{a}_defended_{HI_NAME}_{t}" for a in attacks for t in tasks}
    return out


def needs_rerun(row, rows_by_sid, lo, hi) -> bool:
    if str(row.get("error") or "").startswith("skipped:"):
        return False                               # a skip at 0.5 is a skip at 0.3 (no payload etc.)
    if row.get("error"):
        return True                                # not usable for either threshold
    if row.get("deduplicated_to"):                 # defended-clean pointer: follows its episode
        return needs_rerun(resolve(row, rows_by_sid), rows_by_sid, lo, hi)
    return in_band(row, lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "merge"])
    ap.add_argument("out_dir")
    ap.add_argument("--lo", type=float, default=0.30)
    ap.add_argument("--hi", type=float, default=0.50)
    a = ap.parse_args()
    rows = json.loads((Path(a.out_dir) / "attack_results.json").read_text())
    by_sid = {r["sample_id"]: r for r in rows}
    hi_rows = [r for r in rows if r.get("defense_name") == HI_NAME]
    if not hi_rows:
        sys.exit(f"no {HI_NAME} rows in {a.out_dir}")
    missing_hi = sorted(expected_hi_sids(rows) - {r["sample_id"] for r in hi_rows})
    if missing_hi:
        # the 0.5 pass is incomplete: a band computed now would silently cover
        # fewer episodes (review P2a.1, #5)
        sys.exit(f"FATAL: the {HI_NAME} pass is incomplete -- {len(missing_hi)} expected row(s) "
                 f"missing, e.g. {missing_hi[:3]}. Finish the main job (resubmit it) first.")

    if a.mode == "select":
        sel = sorted(to_lo(r["sample_id"], a.lo) for r in hi_rows if needs_rerun(r, by_sid, a.lo, a.hi))
        # a deduplicated pair needs its representative run too
        for r in hi_rows:
            rep = r.get("deduplicated_to")
            if rep and to_lo(r["sample_id"], a.lo) in sel:
                sel.append(to_lo(rep, a.lo))
        sel = sorted(set(sel))
        if sel:
            print("\n".join(sel))
        print(f"[{len(sel)} of {len(hi_rows)} {HI_NAME} rows need re-running at {a.lo}]", file=sys.stderr)
        return 0

    out, carried, rerun, missing = [], 0, 0, []
    for r in hi_rows:
        lo_sid = to_lo(r["sample_id"], a.lo)
        if needs_rerun(r, by_sid, a.lo, a.hi):
            lr = by_sid.get(lo_sid)
            if lr is None or lr.get("error"):
                missing.append(lo_sid)
                continue
            out.append(lr); rerun += 1
        elif str(r.get("error") or "").startswith("skipped:"):
            c = dict(r); c.update(sample_id=lo_sid, defense_name=lo_name(a.lo), carried_from=r["sample_id"])
            out.append(c); carried += 1
        else:
            c = dict(r)
            scores = resolve(r, by_sid).get("defense_scores") or []
            c.update(sample_id=lo_sid, defense_name=lo_name(a.lo), carried_from=r["sample_id"],
                     defense_detections=sum(1 for s in scores if float(s) <= a.lo + EPS),
                     defense_scores=scores,
                     # the 0.5 stamp is kept under its own name: the 0.3 condition did not run here
                     defense_sha256=None, carried_defense_sha256=r.get("defense_sha256"))
            if r.get("deduplicated_to"):
                c["deduplicated_to"] = to_lo(r["deduplicated_to"], a.lo)
            out.append(c); carried += 1
    print(json.dumps(out, indent=1))
    print(f"[{lo_name(a.lo)}: {len(out)} rows = {carried} carried + {rerun} re-run; "
          f"{len(missing)} still to run: {missing[:5]}]", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
