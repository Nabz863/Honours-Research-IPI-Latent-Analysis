"""
tests/test_attention_tracker.py
----------------------------------
Tests for AttentionTrackerDetector: focus-ratio computation against
known attention distributions, important-head selection correctly
ranking heads by clean-condition instruction focus, calibration math,
and correct distraction flagging/non-flagging.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from defenses.attention_tracker import AttentionTrackerDetector


def test_focus_ratio_uniform_attention():
    """Uniform attention over 20 keys, instruction span [0:5) -- exactly
    5/20 of the mass should land on the span."""
    attn = torch.ones(20) / 20.0
    ratio = AttentionTrackerDetector._focus_ratio(attn, (0, 5))
    assert abs(ratio - 0.25) < 1e-6


def test_focus_ratio_fully_concentrated_on_instruction():
    attn = torch.zeros(20)
    attn[0:5] = 1.0 / 5.0
    ratio = AttentionTrackerDetector._focus_ratio(attn, (0, 5))
    assert abs(ratio - 1.0) < 1e-6


def test_focus_ratio_fully_concentrated_elsewhere():
    attn = torch.zeros(20)
    attn[10:15] = 1.0 / 5.0
    ratio = AttentionTrackerDetector._focus_ratio(attn, (0, 5))
    assert abs(ratio - 0.0) < 1e-6


def _make_attention_sample(n_layers, n_heads, seq_len, high_focus_heads, instruction_span, high=0.85, low=0.05):
    """Builds one [n_layers, n_heads, seq_len] sample where heads in
    high_focus_heads concentrate `high` fraction of mass on the
    instruction span (rest spread uniformly outside it), and all other
    heads concentrate only `low` fraction there."""
    start, end = instruction_span
    span_len = end - start
    outside_len = seq_len - span_len
    sample = torch.zeros(n_layers, n_heads, seq_len)
    for l in range(n_layers):
        for h in range(n_heads):
            frac_on = high if (l, h) in high_focus_heads else low
            sample[l, h, start:end] = frac_on / span_len
            sample[l, h, :start] = (1 - frac_on) / outside_len / 2
            sample[l, h, end:] = (1 - frac_on) / outside_len / 2
    return sample


def test_select_important_heads_picks_highest_focus_heads():
    instruction_span = (0, 3)
    high_focus = {(0, 0), (1, 2)}
    samples = [
        _make_attention_sample(2, 4, 10, high_focus, instruction_span)
        for _ in range(5)
    ]

    detector = AttentionTrackerDetector(top_k_heads=2)
    detector.select_important_heads(samples, instruction_span)

    assert set(detector.important_heads) == high_focus


def test_calibrate_before_select_raises():
    detector = AttentionTrackerDetector()
    samples = [torch.rand(2, 4, 10) for _ in range(5)]
    raised = False
    try:
        detector.calibrate(samples, (0, 3))
    except RuntimeError:
        raised = True
    assert raised


def test_detect_before_calibrate_raises():
    detector = AttentionTrackerDetector()
    instruction_span = (0, 3)
    high_focus = {(0, 0)}
    samples = [_make_attention_sample(2, 4, 10, high_focus, instruction_span) for _ in range(3)]
    detector.select_important_heads(samples, instruction_span)

    raised = False
    try:
        detector.detect(samples[0], instruction_span)
    except RuntimeError:
        raised = True
    assert raised


def test_distraction_flagged_on_large_focus_drop():
    """The actual scientific claim: an attention pattern where the
    important heads have STOPPED focusing on the instruction (simulating
    the paper's 'distraction effect' under attack) should be flagged."""
    instruction_span = (0, 3)
    high_focus = {(0, 0), (1, 1)}
    clean_samples = [
        _make_attention_sample(2, 4, 10, high_focus, instruction_span, high=0.85, low=0.05)
        for _ in range(10)
    ]

    detector = AttentionTrackerDetector(top_k_heads=2, k_sigma=2.0)
    detector.select_important_heads(clean_samples, instruction_span)
    detector.calibrate(clean_samples, instruction_span)

    # "Attacked" sample: the important heads now behave like the LOW-
    # focus heads did in clean data -- attention has moved away from the
    # instruction, exactly the distraction effect this is meant to catch.
    attacked_sample = _make_attention_sample(2, 4, 10, high_focus_heads=set(), instruction_span=instruction_span)

    result = detector.detect(attacked_sample, instruction_span)
    assert result.is_distracted, (
        f"Expected distraction flagged: focus={result.focus_ratio}, threshold={result.threshold}"
    )


def test_no_distraction_when_focus_similar_to_clean():
    instruction_span = (0, 3)
    high_focus = {(0, 0), (1, 1)}
    clean_samples = [
        _make_attention_sample(2, 4, 10, high_focus, instruction_span, high=0.85, low=0.05)
        for _ in range(10)
    ]

    detector = AttentionTrackerDetector(top_k_heads=2, k_sigma=2.0)
    detector.select_important_heads(clean_samples, instruction_span)
    detector.calibrate(clean_samples, instruction_span)

    # A genuinely clean sample NOT in the calibration set, drawn from the
    # same distribution -- should NOT be flagged.
    new_clean_sample = _make_attention_sample(2, 4, 10, high_focus, instruction_span, high=0.85, low=0.05)
    result = detector.detect(new_clean_sample, instruction_span)
    assert not result.is_distracted, f"Expected no distraction, got focus={result.focus_ratio}, threshold={result.threshold}"


if __name__ == "__main__":
    tests = [
        test_focus_ratio_uniform_attention,
        test_focus_ratio_fully_concentrated_on_instruction,
        test_focus_ratio_fully_concentrated_elsewhere,
        test_select_important_heads_picks_highest_focus_heads,
        test_calibrate_before_select_raises,
        test_detect_before_calibrate_raises,
        test_distraction_flagged_on_large_focus_drop,
        test_no_distraction_when_focus_similar_to_clean,
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
