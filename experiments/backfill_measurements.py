"""
experiments/backfill_measurements.py
---------------------------------------
Computes TaskTracker and CircuitBreaker measurements against an
ALREADY-COMPLETED run's saved .pt activation files -- no model,
no GPU inference, no re-run needed. Both detectors operate purely on
saved activation tensors (see defenses/task_tracker.py and
defenses/circuit_breaker.py), and every episode this project has ever
run already saves its turn_activations to {output_dir}/{sample_id}.pt
regardless of whether the live run had --enable_task_tracker /
--enable_circuit_breaker_measurement set. This exists specifically so a
run that finished WITHOUT those flags (the completed LLaVA-NeXT full
matrix) doesn't need a second 24-30 hour biggpu re-run just to get this
measurement data -- it can be computed after the fact, cheaply, on CPU.

Does NOT touch attack_results.json -- writes a new file
(attack_results_with_measurements.json in the same directory) so the
original is never at risk.

Usage:
    python3 experiments/backfill_measurements.py agentdojo_matrix_full
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from defenses.task_tracker import TaskTrackerDetector
from defenses.circuit_breaker import CircuitBreakerDirection


def _load_pt(output_dir: Path, sample_id: str) -> Optional[List[Dict[int, torch.Tensor]]]:
    path = output_dir / f"{sample_id}.pt"
    if not path.exists():
        return None
    try:
        turns = torch.load(path, map_location="cpu")
    except Exception:
        return None
    return turns if turns else None


def _pick_layer(output_dir: Path, results: List[dict]) -> int:
    """Finds the first available layer key from any non-error row's saved
    .pt file -- doesn't assume a specific --target_layers value was used,
    since that's a run-time CLI choice this script has no other way to
    know."""
    for r in results:
        if r.get("error") is not None:
            continue
        turns = _load_pt(output_dir, r["sample_id"])
        if turns:
            return sorted(turns[0].keys())[0]
    raise RuntimeError("Could not find any non-empty .pt file to infer a layer from.")


def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "agentdojo_matrix_full")
    results_path = output_dir / "attack_results.json"
    results = json.loads(results_path.read_text())

    layer = _pick_layer(output_dir, results)
    print(f"Using layer {layer} (first layer found in a real saved .pt file).")

    clean_rows = [r for r in results if r["condition"] == "clean" and r.get("error") is None]
    attacked_rows = [r for r in results if r["condition"] in ("attacked", "defended_attacked") and r.get("error") is None]

    # ---- TaskTracker calibration: (turn0, last_turn) pairs from clean
    # episodes with >=2 turns -- exactly what a live run's
    # enable_task_tracker path would have collected, just gathered from
    # disk instead of in-process. ----
    tt_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for r in clean_rows:
        if r["num_turns"] < 2:
            continue
        turns = _load_pt(output_dir, r["sample_id"])
        if turns and layer in turns[0] and layer in turns[-1]:
            tt_pairs.append((turns[0][layer], turns[-1][layer]))

    task_tracker: Optional[TaskTrackerDetector] = None
    if len(tt_pairs) >= 2:
        task_tracker = TaskTrackerDetector(k=3.0)
        task_tracker.calibrate(tt_pairs)
        print(f"TaskTracker calibrated from {len(tt_pairs)} clean pairs (threshold={task_tracker.threshold:.4f}).")
    else:
        print(f"Only {len(tt_pairs)} clean 2+-turn pairs found -- skipping TaskTracker.")

    # ---- CircuitBreaker: (attacked last-turn, clean last-turn) pairs,
    # matched by task_id -- same pairing logic the live run's
    # cb_first_attack_name bootstrap uses, just reading from disk. ----
    clean_last_by_task: Dict[str, torch.Tensor] = {}
    for r in clean_rows:
        turns = _load_pt(output_dir, r["sample_id"])
        if turns and layer in turns[-1]:
            clean_last_by_task[r["task_id"]] = turns[-1][layer]

    cb_attacked: List[torch.Tensor] = []
    cb_clean: List[torch.Tensor] = []
    for r in attacked_rows:
        if r["task_id"] not in clean_last_by_task:
            continue
        turns = _load_pt(output_dir, r["sample_id"])
        if turns and layer in turns[-1]:
            cb_attacked.append(turns[-1][layer])
            cb_clean.append(clean_last_by_task[r["task_id"]])

    circuit_breaker: Optional[CircuitBreakerDirection] = None
    if len(cb_attacked) >= 2 and len(cb_clean) >= 2:
        circuit_breaker = CircuitBreakerDirection(k_sigma=2.0)
        circuit_breaker.fit_direction(cb_attacked, cb_clean)
        circuit_breaker.calibrate(list(clean_last_by_task.values()))
        print(f"CircuitBreaker fit from {len(cb_attacked)} attacked/clean pairs (threshold={circuit_breaker.threshold:.4f}).")
    else:
        print(f"Only {len(cb_attacked)} attacked/clean pairs found -- skipping CircuitBreaker.")

    # ---- Apply both to every non-error row, append fields, write out a
    # NEW file (never touches the original). ----
    augmented = []
    tt_applied = cb_applied = 0
    for r in results:
        row = dict(r)
        if r.get("error") is None:
            turns = _load_pt(output_dir, r["sample_id"])
            if turns:
                if task_tracker is not None and len(turns) >= 2 and layer in turns[0] and layer in turns[-1]:
                    res = task_tracker.detect(turns, layer=layer)
                    row["task_tracker_drift_detected"] = res.is_drifted
                    row["task_tracker_delta"] = res.delta
                    tt_applied += 1
                if circuit_breaker is not None and layer in turns[-1]:
                    res = circuit_breaker.maybe_reroute(turns[-1][layer])
                    row["circuit_breaker_would_reroute"] = res.was_rerouted
                    row["circuit_breaker_projection"] = res.projection
                    cb_applied += 1
        augmented.append(row)

    out_path = output_dir / "attack_results_with_measurements.json"
    out_path.write_text(json.dumps(augmented, indent=2))
    print(f"\nApplied TaskTracker to {tt_applied} rows, CircuitBreaker to {cb_applied} rows.")
    print(f"Wrote {out_path} -- original attack_results.json is untouched.")


if __name__ == "__main__":
    main()
