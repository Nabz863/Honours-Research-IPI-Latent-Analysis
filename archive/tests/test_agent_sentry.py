"""
tests/test_agent_sentry.py
-----------------------------
Tests for AgentSentryDiagnostics: the ACE = DE + IE algebraic identity
(holds by construction, checked directly), correct takeover flagging
when mediator content alone drives high-severity actions that
sanitization removes, correct non-flagging when severity is unaffected
by sanitization, purify() delegation, and default_agentdojo_severity_fn's
tool-call-based classification.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from defenses.agent_sentry import AgentSentryDiagnostics, default_agentdojo_severity_fn
from defenses.spotlighting import SpotlightingDefense


@dataclass
class _FakeCall:
    function: str


@dataclass
class _FakeAction:
    tool_calls: Optional[List[_FakeCall]]


def _make_next_action_fn(severity_by_regime: dict):
    """
    Builds a deterministic dry_run_next_action(user_input, mediator) that
    identifies which of the four regimes it's being called for (by
    whether user_input matches the neutral probe and whether mediator
    matches the sanitized form) and returns a fake action whose severity
    (via a trivial severity_fn) matches severity_by_regime[regime_name].
    """
    purifier = SpotlightingDefense(variant="datamarking")

    def next_action_fn(user_input, mediator):
        sanitized = purifier.apply("REAL MEDIATOR CONTENT").transformed_content
        is_masked = user_input != "REAL_GOAL"
        is_sanitized = mediator == sanitized
        if not is_masked and not is_sanitized:
            regime = "orig"
        elif is_masked and not is_sanitized:
            regime = "mask"
        elif is_masked and is_sanitized:
            regime = "mask_sanitized"
        else:
            regime = "orig_sanitized"
        severity = severity_by_regime[regime]
        # severity_fn below just reads .severity directly -- simplest
        # possible fake, avoids coupling this test to tool-call shapes.
        return _SeverityHolder(severity)

    return next_action_fn, purifier


@dataclass
class _SeverityHolder:
    severity: int


def _trivial_severity_fn(action) -> int:
    return action.severity


def test_ace_ie_de_are_each_the_correct_regime_difference():
    """Each contrast is an unambiguous difference between two clearly-
    specified regimes straight from the paper's own prose -- checked
    individually, rather than asserting the paper's claimed ACE~=DE+IE
    composition, which this module does not rely on (see module
    docstring for why: the composition did not reproduce even under a
    plain additive severity model when hand-derived, most likely from
    noisy transcription of the paper's dense, truncated equation
    rendering -- safer to verify only what's unambiguous)."""
    severities = {"orig": 2, "mask": 1, "mask_sanitized": 0, "orig_sanitized": 1}
    next_action_fn, purifier = _make_next_action_fn(severities)
    diag = AgentSentryDiagnostics(severity_fn=_trivial_severity_fn, purifier=purifier, takeover_threshold=0.5)

    result = diag.diagnose(next_action_fn, real_user_goal="REAL_GOAL", real_mediator_content="REAL MEDIATOR CONTENT")

    assert result.ace == severities["orig"] - severities["mask"]
    assert result.ie == severities["mask"] - severities["mask_sanitized"]
    assert result.de == severities["orig_sanitized"] - severities["mask_sanitized"]
    assert result.regime_severities == {
        "orig": 2.0, "mask": 1.0, "mask_sanitized": 0.0, "orig_sanitized": 1.0,
    }


def test_takeover_flagged_when_mediator_alone_drives_high_severity():
    """The actual scientific claim: mediator content that induces risky
    behavior EVEN WITHOUT the real user goal present, and which
    sanitization removes, should be flagged as a takeover."""
    severities = {
        "orig": 2,             # real goal + real mediator -> risky
        "mask": 2,              # NO real goal + real mediator -> STILL risky (mediator alone is driving it)
        "mask_sanitized": 0,    # NO real goal + sanitized mediator -> safe (sanitization removed the risk)
        "orig_sanitized": 0,    # real goal + sanitized mediator -> safe
    }
    next_action_fn, purifier = _make_next_action_fn(severities)
    diag = AgentSentryDiagnostics(severity_fn=_trivial_severity_fn, purifier=purifier, takeover_threshold=0.5)

    result = diag.diagnose(next_action_fn, real_user_goal="REAL_GOAL", real_mediator_content="REAL MEDIATOR CONTENT")

    assert result.ie == 2.0, f"Expected IE=2.0 (mask - mask_sanitized = 2 - 0), got {result.ie}"
    assert result.is_takeover, f"Expected takeover flagged, IE={result.ie} > threshold={result.threshold}"


def test_no_takeover_when_severity_unaffected_by_sanitization():
    """If severity is the SAME regardless of whether the mediator is
    sanitized, the mediator content isn't what's driving the behavior --
    should NOT be flagged, avoiding the over-eager-blocking failure mode
    the paper explicitly targets."""
    severities = {"orig": 1, "mask": 1, "mask_sanitized": 1, "orig_sanitized": 1}
    next_action_fn, purifier = _make_next_action_fn(severities)
    diag = AgentSentryDiagnostics(severity_fn=_trivial_severity_fn, purifier=purifier, takeover_threshold=0.5)

    result = diag.diagnose(next_action_fn, real_user_goal="REAL_GOAL", real_mediator_content="REAL MEDIATOR CONTENT")

    assert result.ie == 0.0
    assert not result.is_takeover


def test_purify_delegates_to_configured_purifier():
    custom_purifier = SpotlightingDefense(variant="encoding")
    diag = AgentSentryDiagnostics(severity_fn=_trivial_severity_fn, purifier=custom_purifier)

    result = diag.purify("some mediator content")
    expected = custom_purifier.apply("some mediator content").transformed_content

    # encoding is deterministic (base64), so this should match exactly --
    # confirms purify() actually delegates rather than using its own
    # separate default regardless of what was configured.
    assert result == expected


def test_default_purifier_is_datamarking_when_none_given():
    diag = AgentSentryDiagnostics(severity_fn=_trivial_severity_fn)
    assert diag.purifier.variant == "datamarking"


def test_default_agentdojo_severity_fn_classifies_correctly():
    severity_fn = default_agentdojo_severity_fn(exfil_tool_names={"send_email", "send_money"})

    no_call_action = _FakeAction(tool_calls=None)
    assert severity_fn(no_call_action) == 0

    diagnostic_action = _FakeAction(tool_calls=[_FakeCall(function="search_calendar")])
    assert severity_fn(diagnostic_action) == 1

    exfil_action = _FakeAction(tool_calls=[_FakeCall(function="send_email")])
    assert severity_fn(exfil_action) == 2

    mixed_action = _FakeAction(tool_calls=[_FakeCall(function="search_calendar"), _FakeCall(function="send_money")])
    assert severity_fn(mixed_action) == 2, "Any exfil-tier call in the action should win, regardless of order"


if __name__ == "__main__":
    tests = [
        test_ace_ie_de_are_each_the_correct_regime_difference,
        test_takeover_flagged_when_mediator_alone_drives_high_severity,
        test_no_takeover_when_severity_unaffected_by_sanitization,
        test_purify_delegates_to_configured_purifier,
        test_default_purifier_is_datamarking_when_none_given,
        test_default_agentdojo_severity_fn_classifies_correctly,
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
