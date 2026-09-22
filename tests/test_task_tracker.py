"""
tests/test_task_tracker.py
-----------------------------
Tests for TaskTrackerDetector: calibration math against a known
synthetic distribution, correct flagging of a large synthetic delta as
drift, correct non-flagging of a near-zero delta, both distance metrics,
and the error-handling paths (uncalibrated detector, too few turns,
missing layer, too few calibration pairs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from defenses.task_tracker import TaskTrackerDetector


def test_invalid_metric_raises():
    raised = False
    try:
        TaskTrackerDetector(metric="not_a_real_metric")
    except ValueError:
        raised = True
    assert raised


def test_detect_before_calibrate_raises():
    detector = TaskTrackerDetector()
    turn_activations = [{0: torch.randn(8)}, {0: torch.randn(8)}]
    raised = False
    try:
        detector.detect(turn_activations, layer=0)
    except RuntimeError:
        raised = True
    assert raised


def test_calibrate_needs_at_least_two_pairs():
    detector = TaskTrackerDetector()
    raised = False
    try:
        detector.calibrate([(torch.randn(8), torch.randn(8))])
    except ValueError:
        raised = True
    assert raised


def test_detect_needs_at_least_two_turns():
    detector = TaskTrackerDetector()
    detector.calibrate([(torch.randn(8), torch.randn(8)) for _ in range(5)])
    raised = False
    try:
        detector.detect([{0: torch.randn(8)}], layer=0)
    except ValueError:
        raised = True
    assert raised


def test_detect_missing_layer_raises():
    detector = TaskTrackerDetector()
    detector.calibrate([(torch.randn(8), torch.randn(8)) for _ in range(5)])
    turn_activations = [{0: torch.randn(8)}, {0: torch.randn(8)}]
    raised = False
    try:
        detector.detect(turn_activations, layer=5)  # layer 5 was never captured
    except KeyError:
        raised = True
    assert raised


def test_calibration_threshold_matches_known_distribution():
    """Verifies the actual mean + k*std math against a hand-computed
    distribution, not just that SOME threshold gets set."""
    torch.manual_seed(0)
    detector = TaskTrackerDetector(metric="l2", k=2.0)

    # Construct pairs with KNOWN, controlled L2 distances: identical
    # base vector, offset by a fixed scalar times a unit direction, so
    # the resulting delta is exactly that scalar.
    base = torch.zeros(4)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    known_deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
    pairs = [(base, base + d * direction) for d in known_deltas]

    detector.calibrate(pairs)

    mean = sum(known_deltas) / len(known_deltas)
    var = sum((d - mean) ** 2 for d in known_deltas) / len(known_deltas)
    std = var ** 0.5
    expected_threshold = mean + 2.0 * std

    assert abs(detector.threshold - expected_threshold) < 1e-4, (
        f"threshold={detector.threshold}, expected={expected_threshold}"
    )


def test_large_delta_is_flagged_as_drift():
    torch.manual_seed(1)
    detector = TaskTrackerDetector(metric="cosine_distance", k=2.0)
    # Calibrate on pairs that are nearly IDENTICAL (small natural drift).
    clean_pairs = [(v := torch.randn(16), v + 0.01 * torch.randn(16)) for _ in range(20)]
    detector.calibrate(clean_pairs)

    # A "drifted" episode: turn 0 and the last turn point in NEARLY
    # OPPOSITE directions -- cosine_distance close to 2.0, far beyond
    # anything in the near-identical calibration set.
    turn0 = torch.randn(16)
    turn_last = -turn0 + 0.01 * torch.randn(16)
    turn_activations = [{3: turn0}, {3: torch.randn(16)}, {3: turn_last}]

    result = detector.detect(turn_activations, layer=3, reference_turn=0, check_turn=-1)
    assert result.is_drifted, f"Expected drift flagged, got delta={result.delta}, threshold={result.threshold}"
    assert result.check_turn == 2  # -1 resolved against a 3-turn episode


def test_near_identical_activation_is_not_flagged():
    torch.manual_seed(2)
    detector = TaskTrackerDetector(metric="cosine_distance", k=2.0)
    clean_pairs = [(v := torch.randn(16), v + 0.01 * torch.randn(16)) for _ in range(20)]
    detector.calibrate(clean_pairs)

    # A genuinely clean episode NOT in the calibration set, but drawn
    # from the same "small drift" distribution -- should NOT be flagged.
    turn0 = torch.randn(16)
    turn_last = turn0 + 0.01 * torch.randn(16)
    turn_activations = [{3: turn0}, {3: turn_last}]

    result = detector.detect(turn_activations, layer=3)
    assert not result.is_drifted, f"Expected no drift, got delta={result.delta}, threshold={result.threshold}"


def test_l2_metric_also_works():
    torch.manual_seed(3)
    detector = TaskTrackerDetector(metric="l2", k=2.0)
    clean_pairs = [(v := torch.randn(16), v + 0.01 * torch.randn(16)) for _ in range(20)]
    detector.calibrate(clean_pairs)

    turn0 = torch.randn(16)
    turn_last = turn0 + 50.0 * torch.randn(16)  # huge L2 shift
    turn_activations = [{0: turn0}, {0: turn_last}]

    result = detector.detect(turn_activations, layer=0)
    assert result.is_drifted
    assert result.metric == "l2"


if __name__ == "__main__":
    tests = [
        test_invalid_metric_raises,
        test_detect_before_calibrate_raises,
        test_calibrate_needs_at_least_two_pairs,
        test_detect_needs_at_least_two_turns,
        test_detect_missing_layer_raises,
        test_calibration_threshold_matches_known_distribution,
        test_large_delta_is_flagged_as_drift,
        test_near_identical_activation_is_not_flagged,
        test_l2_metric_also_works,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)
