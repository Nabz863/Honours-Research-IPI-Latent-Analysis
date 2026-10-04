#!/usr/bin/env python3
"""
analysis/invalidate_rows.py
---------------------------
Mark rows affected by implementation fault A8 so they are retried and
excluded -- without deleting them. Dry run by default.

WHY NOT DELETE
    Deleting a row destroys the record that it existed, what it measured and
    why it was discarded. The audit needs all three. Instead each affected row
    gets:

        error              = "invalidated: A8 -- <reason>"
        invalidated_from   = the row's original asr/ua/num_steps/time/error

    which has two effects, both already implemented elsewhere:
      - the runner's done() requires `error is None`, so a resubmit RETRIES the
        row and record() overwrites it in place with a fresh episode;
      - geometry.py (after the integrity patch) and rq2_geometry.py skip rows
        carrying an error, so the stale activations are never analysed.

    The row's .pt file is renamed to <sid>.pt.invalid rather than deleted, so a
    resubmit writes a fresh one and the old capture stays inspectable.

WHICH ROWS (fault A8)
    The pixel-space attacks call model.train() and gradient_checkpointing_
    enable() inside generate() and never restore either. So:

    The model state is REPLAYED row by row, because it flips both ways:

      - a pixel-space attack's generate() switches training mode ON
        (adversarial_patch, pixel_perturbation, cross_modal_injection);
      - gcg's and obfuscated_activation's generate() call self.model.eval(),
        switching it OFF again -- and gradient checkpointing only acts in
        training mode, so eval() alone undoes the effect;
      - defended passes reuse cached payloads and call no generate(), so they
        inherit whatever state the undefended loop left behind.

    A row is affected if its episode ran while the replayed state was ON.
    A row "generated" if it carries a non-null attack_generation_time_s.

    First version of this script used "everything after the first pixel
    attack" and would have invalidated 10 valid obfuscated_activation rows in
    vwa_adv_qwen3vl_ua10, whose 50 s median confirms they ran in eval mode.

    Row order is execution order for rows written once. Rows retried in place
    keep their original position, so on a resumed directory the replay is
    approximate -- the report prints median episode time for affected and
    unaffected rows so it can be checked.

USAGE
    python3 analysis/invalidate_rows.py vwa_adv_qwen3vl_ua10_def          # report
    python3 analysis/invalidate_rows.py vwa_adv_qwen3vl_ua10_def --apply  # do it
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

PIXEL_ATTACKS = {"adversarial_patch", "pixel_perturbation", "cross_modal_injection"}
# generate() calls self.model.eval(): gcg.py:261, obfuscated_activation.py:174
EVAL_RESETTERS = {"gcg", "obfuscated_activation"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default is a dry run)")
    ap.add_argument("--pixel_rows_only", action="store_true",
                    help="invalidate only the pixel-attack rows, not everything "
                         "after the first one (use when later rows are known to "
                         "have run in a fresh process)")
    a = ap.parse_args()

    d = Path(a.out_dir)
    rp = d / "attack_results.json"
    if not rp.exists():
        print(f"no {rp}", file=sys.stderr)
        return 2
    rows = json.loads(rp.read_text())

    affected, first = [], None
    training = False                  # replayed model state
    for i, r in enumerate(rows):
        atk = r.get("attack_name")
        generated = r.get("attack_generation_time_s") is not None
        if generated and atk in PIXEL_ATTACKS:
            training = True
            if first is None:
                first = i
        elif generated and atk in EVAL_RESETTERS:
            training = False
        if str(r.get("error") or "").startswith("invalidated:"):
            continue                                  # already done
        ran = r.get("num_steps") is not None
        if training and ran and not (a.pixel_rows_only and atk not in PIXEL_ATTACKS):
            why = ("pixel-attack episode (own generate() set train mode)"
                   if atk in PIXEL_ATTACKS and generated
                   else "ran while a prior pixel attack had left train mode on")
            affected.append((i, why))

    def med(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return f"{statistics.median(xs):.1f}s" if xs else "-"

    aff = {i for i, _ in affected}
    before_t = [r.get("episode_time_s") for i, r in enumerate(rows)
                if i not in aff and r.get("num_steps") is not None]
    after_t = [rows[i].get("episode_time_s") for i, _ in affected]

    print(f"{d}: {len(rows)} rows")
    if first is None:
        print("  no successful pixel-space generation found -- nothing affected by A8")
        return 0
    print(f"  first pixel-attack generation: row {first} ({rows[first].get('sample_id')})")
    print(f"  median episode time, unaffected rows: {med(before_t)}")
    print(f"  median episode time of affected rows: {med(after_t)}")
    print(f"  rows to invalidate: {len(affected)}")
    by = {}
    for i, why in affected:
        k = (rows[i].get("attack_name"), rows[i].get("defense_name") or "none")
        by[k] = by.get(k, 0) + 1
    for (atk, dfn), n in sorted(by.items()):
        print(f"    {atk:<24} {dfn:<26} {n}")

    if not a.apply:
        print("\n  DRY RUN -- nothing written. Re-run with --apply.")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bak = d / f"attack_results.json.bak-{stamp}"
    shutil.copy2(rp, bak)
    renamed = 0
    for i, why in affected:
        r = rows[i]
        r["invalidated_from"] = {k: r.get(k) for k in
                                 ("asr_success", "utility_success", "num_steps",
                                  "episode_time_s", "peak_cuda_memory_gb", "error")}
        r["error"] = f"invalidated: A8 -- {why}"
        pt = d / f"{r.get('sample_id')}.pt"
        if pt.exists():
            pt.rename(pt.with_suffix(".pt.invalid"))
            renamed += 1
    rp.write_text(json.dumps(rows, indent=2))
    print(f"\n  APPLIED. backup: {bak.name}")
    print(f"  {len(affected)} rows marked; {renamed} .pt files renamed to .pt.invalid")
    print("  A resubmit retries every marked row (done() requires error is None).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
