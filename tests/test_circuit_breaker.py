"""
tests/test_circuit_breaker.py
--------------------------------
Tests for CircuitBreakerDirection: direction computed as the correct
normalized difference of means, calibration math, correct rerouting of
an attacked-like activation with verified ablation (zero residual
projection), and correct non-rerouting of a clean-like activation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from defenses.circuit_breaker import CircuitBreakerDirection


def test_fit_direction_before_data_raises():
    cb = CircuitBreakerDirection()
    raised = False
    try:
        cb.fit_direction([], [])
    except ValueError:
        raised = True
    assert raised


def test_calibrate_before_fit_raises():
    cb = CircuitBreakerDirection()
    raised = False
    try:
        cb.calibrate([torch.randn(8) for _ in range(3)])
    except RuntimeError:
        raised = True
    assert raised


def test_maybe_reroute_before_calibrate_raises():
    cb = CircuitBreakerDirection()
    cb.fit_direction([torch.randn(8) for _ in range(3)], [torch.randn(8) for _ in range(3)])
    raised = False
    try:
        cb.maybe_reroute(torch.randn(8))
    except RuntimeError:
        raised = True
    assert raised


def test_direction_is_normalized_difference_of_means():
    torch.manual_seed(0)
    attacked = [torch.tensor([10.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4) for _ in range(20)]
    clean = [torch.tensor([0.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4) for _ in range(20)]

    cb = CircuitBreakerDirection()
    cb.fit_direction(attacked, clean)

    # Direction should point overwhelmingly along the first axis (where
    # attacked and clean means actually differ) and be unit-norm.
    assert abs(cb.direction.norm().item() - 1.0) < 1e-4
    assert cb.direction[0].item() > 0.99
    assert cb.direction[1:].abs().max().item() < 0.05


def test_attacked_like_activation_gets_rerouted_and_ablated():
    torch.manual_seed(1)
    attacked = [torch.tensor([10.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4) for _ in range(20)]
    clean = [torch.tensor([0.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4) for _ in range(20)]

    cb = CircuitBreakerDirection(k_sigma=2.0)
    cb.fit_direction(attacked, clean)
    cb.calibrate(clean)

    attacked_like = torch.tensor([9.5, 0.1, -0.1, 0.0])
    result = cb.maybe_reroute(attacked_like)

    assert result.was_rerouted, f"Expected rerouting: projection={result.projection}, threshold={result.threshold}"
    # The whole point of ablation: after rerouting, the residual
    # projection onto the SAME direction should be ~zero -- checked
    # directly, not assumed just because was_rerouted is True.
    residual_projection = torch.dot(result.rerouted_activation.flatten().float(), cb.direction).item()
    assert abs(residual_projection) < 1e-4, f"Ablation left residual projection {residual_projection}"


def test_clean_like_activation_not_rerouted():
    torch.manual_seed(2)
    attacked = [torch.tensor([10.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4) for _ in range(20)]
    clean = [torch.tensor([0.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4) for _ in range(20)]

    cb = CircuitBreakerDirection(k_sigma=2.0)
    cb.fit_direction(attacked, clean)
    cb.calibrate(clean)

    clean_like = torch.tensor([0.02, -0.01, 0.0, 0.01])
    result = cb.maybe_reroute(clean_like)

    assert not result.was_rerouted, f"Expected no rerouting, got projection={result.projection}, threshold={result.threshold}"
    # Unmodified means literally the same tensor content returned.
    assert torch.equal(result.rerouted_activation, clean_like)


def test_shape_and_dtype_preserved_through_ablation():
    """Ablation flattens internally -- confirm it correctly reshapes back
    for a non-1D activation, and preserves the original dtype."""
    torch.manual_seed(3)
    attacked = [torch.ones(2, 4) * 10.0 + 0.01 * torch.randn(2, 4) for _ in range(10)]
    clean = [torch.zeros(2, 4) + 0.01 * torch.randn(2, 4) for _ in range(10)]

    cb = CircuitBreakerDirection(k_sigma=1.0)
    cb.fit_direction(attacked, clean)
    cb.calibrate(clean)

    attacked_like = (torch.ones(2, 4) * 9.5).half()
    result = cb.maybe_reroute(attacked_like)

    assert result.was_rerouted
    assert result.rerouted_activation.shape == (2, 4)
    assert result.rerouted_activation.dtype == torch.float16


if __name__ == "__main__":
    tests = [
        test_fit_direction_before_data_raises,
        test_calibrate_before_fit_raises,
        test_maybe_reroute_before_calibrate_raises,
        test_direction_is_normalized_difference_of_means,
        test_attacked_like_activation_gets_rerouted_and_ablated,
        test_clean_like_activation_not_rerouted,
        test_shape_and_dtype_preserved_through_ablation,
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
