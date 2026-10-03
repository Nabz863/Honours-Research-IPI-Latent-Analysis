"""
tests/test_icon_trace_prober.py
----------------------------------
Tests for LatentSpaceTraceProber: slope computed correctly against
known linear/flat trajectories, calibration math, correct flagging of a
trajectory with steep late-layer growth relative to clean calibration,
correct non-flagging of a flat trajectory, and layer-intersection
handling when current/clean activations don't share the exact same
layer set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from defenses.icon_trace_prober import LatentSpaceTraceProber


def test_invalid_metric_raises():
    raised = False
    try:
        LatentSpaceTraceProber(metric="not_real")
    except ValueError:
        raised = True
    assert raised


def test_slope_exact_for_perfectly_linear_trace():
    trace = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    slope = LatentSpaceTraceProber._slope(trace)
    assert abs(slope - 1.0) < 1e-5


def test_slope_zero_for_flat_trace():
    trace = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}
    slope = LatentSpaceTraceProber._slope(trace)
    assert abs(slope - 0.0) < 1e-5


def test_slope_needs_at_least_two_layers():
    raised = False
    try:
        LatentSpaceTraceProber._slope({0: 1.0})
    except ValueError:
        raised = True
    assert raised


def test_calibrate_before_probe_raises():
    prober = LatentSpaceTraceProber()
    current = {l: torch.randn(8) for l in range(4)}
    clean = {l: torch.randn(8) for l in range(4)}
    raised = False
    try:
        prober.probe(current, clean)
    except RuntimeError:
        raised = True
    assert raised


def _make_layer_activations(n_layers, dim, base, growth, noise_seed):
    """Constructs {layer: tensor} where successive layers' activations
    drift further from a fixed reference direction at rate `growth` --
    used to build BOTH the 'current' and 'clean' sides of a comparison
    with a controllable, known resulting slope."""
    torch.manual_seed(noise_seed)
    reference = torch.ones(dim)
    return {
        l: base * reference + growth * l * torch.randn(dim) + growth * l * torch.ones(dim)
        for l in range(n_layers)
    }


def test_over_focusing_flagged_for_steep_late_growth():
    """The actual scientific claim: a trajectory whose distance-to-clean
    GROWS steeply across layers (simulating the paper's 'over-focusing
    signature diverging in a characteristic direction') should be
    flagged, calibrated against flat/bounded clean trajectories."""
    dim = 16
    n_layers = 6

    # Clean calibration: current vs clean activations that stay CLOSE
    # across all layers -> near-flat, near-zero slope traces.
    prober = LatentSpaceTraceProber(metric="l2", k_sigma=2.0)
    clean_traces = []
    for seed in range(10):
        torch.manual_seed(seed)
        clean_by_layer = {l: torch.randn(dim) for l in range(n_layers)}
        nearly_same_by_layer = {l: clean_by_layer[l] + 0.01 * torch.randn(dim) for l in range(n_layers)}
        trace = prober._trace(nearly_same_by_layer, clean_by_layer)
        clean_traces.append(trace)
    prober.calibrate(clean_traces)

    # "Attacked" episode: distance to the SAME clean baseline grows
    # sharply in later layers.
    torch.manual_seed(99)
    clean_by_layer = {l: torch.randn(dim) for l in range(n_layers)}
    attacked_by_layer = {l: clean_by_layer[l] + (l ** 2) * torch.ones(dim) for l in range(n_layers)}

    result = prober.probe(attacked_by_layer, clean_by_layer)
    assert result.is_over_focusing, f"Expected flagged: slope={result.slope}, threshold={result.threshold}"


def test_no_over_focusing_for_flat_trace_similar_to_clean():
    dim = 16
    n_layers = 6
    prober = LatentSpaceTraceProber(metric="l2", k_sigma=2.0)
    clean_traces = []
    for seed in range(10):
        torch.manual_seed(seed)
        clean_by_layer = {l: torch.randn(dim) for l in range(n_layers)}
        nearly_same_by_layer = {l: clean_by_layer[l] + 0.01 * torch.randn(dim) for l in range(n_layers)}
        trace = prober._trace(nearly_same_by_layer, clean_by_layer)
        clean_traces.append(trace)
    prober.calibrate(clean_traces)

    torch.manual_seed(123)
    clean_by_layer = {l: torch.randn(dim) for l in range(n_layers)}
    new_clean_by_layer = {l: clean_by_layer[l] + 0.01 * torch.randn(dim) for l in range(n_layers)}

    result = prober.probe(new_clean_by_layer, clean_by_layer)
    assert not result.is_over_focusing, f"Expected no flag: slope={result.slope}, threshold={result.threshold}"


def test_trace_uses_only_common_layers():
    prober = LatentSpaceTraceProber()
    current = {0: torch.randn(8), 1: torch.randn(8), 5: torch.randn(8)}
    clean = {0: torch.randn(8), 1: torch.randn(8), 2: torch.randn(8)}
    trace = prober._trace(current, clean)
    assert set(trace.keys()) == {0, 1}, f"Expected only common layers {{0,1}}, got {set(trace.keys())}"


if __name__ == "__main__":
    tests = [
        test_invalid_metric_raises,
        test_slope_exact_for_perfectly_linear_trace,
        test_slope_zero_for_flat_trace,
        test_slope_needs_at_least_two_layers,
        test_calibrate_before_probe_raises,
        test_over_focusing_flagged_for_steep_late_growth,
        test_no_over_focusing_for_flat_trace_similar_to_clean,
        test_trace_uses_only_common_layers,
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
