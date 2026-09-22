"""
experiments/invalidate_pre_fix_rows.py
------------------------------------------
One-time repair for agentdojo_matrix_full_v2/ (or any other
run_attack_matrix.py output directory straddling the hooks.py
capture-timing fix): removes rows whose .pt activation file predates
the fix from attack_results.json, so _is_complete() correctly treats
them as not-done and the next run genuinely regenerates them.

CRITICAL, DO NOT SKIP: run this ONLY while the target job is NOT
running -- _persist() overwrites the ENTIRE attack_results.json from
in-memory state after every episode, so editing the file live gets
clobbered within seconds. Cancel the job first.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--fix-timestamp", type=str, required=True)
    parser.add_argument("--delete-stale-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fix_epoch = datetime.fromisoformat(args.fix_timestamp).timestamp()
    results_path = output_dir / "attack_results.json"
    cache_path = output_dir / "clean_activations_cache.pt"

    if not results_path.exists():
        print(f"ERROR: {results_path} not found.", file=sys.stderr)
        sys.exit(1)

    pre_fix_sample_ids = set()
    pt_files = sorted(output_dir.glob("*.pt"))
    for pt_file in pt_files:
        if pt_file.name == "clean_activations_cache.pt":
            continue
        if pt_file.stat().st_mtime < fix_epoch:
            pre_fix_sample_ids.add(pt_file.stem)

    print(f"Found {len(pt_files)} total .pt files, {len(pre_fix_sample_ids)} pre-fix.")

    with open(results_path) as f:
        results_log = json.load(f)
    kept = [r for r in results_log if r["sample_id"] not in pre_fix_sample_ids]
    removed = [r for r in results_log if r["sample_id"] in pre_fix_sample_ids]
    print(f"attack_results.json: {len(results_log)} rows -> {len(kept)} rows ({len(removed)} removed).")

    pre_fix_clean_task_ids = {sid[len("clean_"):] for sid in pre_fix_sample_ids if sid.startswith("clean_")}
    cache_removed = []
    clean_cache = None
    if cache_path.exists() and pre_fix_clean_task_ids:
        clean_cache = torch.load(cache_path, weights_only=False)
        for task_id in pre_fix_clean_task_ids:
            if task_id in clean_cache:
                cache_removed.append(task_id)
                if not args.dry_run:
                    del clean_cache[task_id]
        print(f"clean_activations_cache.pt: purging {len(cache_removed)} contaminated task_id entries.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    with open(results_path, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"\nWrote {results_path} ({len(kept)} rows).")

    if cache_removed and clean_cache is not None:
        torch.save(clean_cache, cache_path)
        print(f"Wrote {cache_path} ({len(clean_cache)} task_ids remaining).")

    if args.delete_stale_files:
        for sid in pre_fix_sample_ids:
            pt_path = output_dir / f"{sid}.pt"
            if pt_path.exists():
                pt_path.unlink()
        print(f"Deleted {len(pre_fix_sample_ids)} stale .pt files.")

    print("\nadv_payloads_cache.json NOT touched -- independent of the capture-timing bug.")


if __name__ == "__main__":
    main()
