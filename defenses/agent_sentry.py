"""
defenses/agent_sentry.py
---------------------------
AgentSentry-style temporal causal diagnostics (Table 3.3, inference-time
baseline). Zhang et al., "AgentSentry: Mitigating Indirect Prompt
Injection in LLM Agents via Temporal Causal Diagnostics and Context
Purification," arXiv:2602.22724 (Wuhan University / University at
Buffalo / Jinan University). Evaluated directly on AgentDojo -- the same
benchmark already running in this project -- reporting ASR=0% and
average UA=74.55%, 20.8-33.6 points above the strongest baselines.

CORE IDEA (confirmed from the actual paper, not guessed): at each
tool-return boundary (the point right after a tool result enters context
and right before the next action is generated), run FOUR "dry-run"
counterfactual re-executions -- side-effect-free, no tool calls actually
committed -- that vary two things independently: whether the REAL user
goal or a task-neutral probe is used, and whether the REAL or a
SANITIZED tool result is used. Comparing the resulting action severity
across these four regimes decomposes the causal effect on the agent's
behavior into a user-goal-driven component and a mediator (tool-content)
-driven component. If the mediator-driven component is large even with
the user's real goal absent, that's evidence the tool content itself --
not the user's actual request -- is driving the proposed action:
takeover. Only THEN is the context purified and the action revised;
otherwise the real action proceeds untouched, which is the paper's whole
point (avoid the "over-eager blocking" failure mode of prior detection-
first defenses that tank utility -- MELON reports UA=32.91% on GPT-4o
under similar conditions, per this paper's own comparison).

FOUR REGIMES (paper's own names, Eq. 14):
  orig            : real user goal (x_b) + real tool result (r_b)
  mask            : neutral probe (x^mask) + real tool result (r_b)
  mask_sanitized  : neutral probe (x^mask) + sanitized tool result
  orig_sanitized  : real user goal (x_b) + sanitized tool result

CAUSAL CONTRASTS (paper's own regime definitions, Eq 14, and formulas,
Eq 18-20, simplified to K=1 sample per regime -- see SIMPLIFICATIONS
below):
  ACE = mu(orig) - mu(mask)                       -- overall user-goal effect
  IE  = mu(mask) - mu(mask_sanitized)              -- mediator-specific effect
  DE  = mu(orig_sanitized) - mu(mask_sanitized)    -- user-goal-only baseline
The paper states these compose as ACE ~= DE + IE "up to Monte Carlo
error" (Section 4.3). That specific identity is NOT asserted or relied
upon here: the source page's dense MathML rendering for Eq 18-20 was
noisy and truncated when fetched, and hand-deriving the composition
under a plain additive (no-interaction) severity model did not reproduce
it -- rather than assert a relationship reconstructed with that much
uncertainty, this module treats ACE, DE, and IE as three independently
well-defined regime contrasts (each is an unambiguous difference between
two clearly-specified conditions, straight from the paper's own prose)
and uses IE alone -- the mediator-specific effect, the paper's own
central diagnostic quantity -- as the takeover signal. Worth revisiting
against the actual PDF (not just the truncated HTML rendering) if the
exact composition matters for a writeup.

SIMPLIFICATIONS FROM THE REAL PAPER, STATED EXPLICITLY:
  - K=1 re-execution per regime here (4 extra LLM calls per boundary
    total), not the paper's full Monte Carlo protocol over K samples
    with uncertainty estimation -- a real cost/precision tradeoff, not a
    free simplification. Real deployment cost is genuinely 4x (at
    minimum) extra generate() calls per tool-return boundary; this is
    worth weighing before wiring into a large real run, the same way
    num_steps/candidates_per_step were tuned for gcg.py's real cost.
  - Takeover_b is decided here by a plain threshold on IE
    (takeover_threshold), not the paper's formal "Temporal Causal
    Degradation Test" (Section 4.5), whose exact statistical form isn't
    reproduced here.
  - Purify() reuses SpotlightingDefense's datamarking variant (already
    built, already "provenance-preserving, structure-preserving" in
    spirit) rather than the paper's own bespoke purification rule
    (Appendix B.2), whose exact mechanism isn't reproduced here.
  - The severity function psi (0/1/2 ordinal: no risk / diagnostic-tool-
    or-deviation / high-impact-tool) needs a caller-supplied
    classification of which tools count as "high-impact" for the
    benchmark in use -- this is domain-specific by the paper's own
    design (Section 3.2), not something this module invents; a
    reasonable AgentDojo-oriented default is provided (default_
    agentdojo_severity_fn) but should be reviewed against the real
    registered tools per suite before trusting it, not assumed correct.

STATUS: mechanics verified against synthetic next-action functions (see
tests/test_agent_sentry.py) -- ACE, DE, and IE are each confirmed to
compute the correct simple difference between their two specified
regimes, correct takeover flagging when mediator content alone (mask
regime) induces high-severity actions the sanitized version doesn't,
correct non-flagging when severity is unaffected by sanitization. Does
NOT verify the paper's claimed ACE~=DE+IE composition -- see CAUSAL
CONTRASTS above for why. NOT yet wired into LocalVLMPipelineElement as a
live mid-generation intervention -- the "dry-run re-execution without
committing side effects" requirement specifically needs real integration
with AgentDojo's ToolsExecutionLoop/runtime, not something this
standalone module does on its own -- that remaining integration step is
not silently done here, same caveat as task_tracker.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .spotlighting import SpotlightingDefense

Severity = int  # 0, 1, or 2 -- see module docstring


@dataclass
class CausalDiagnosticResult:
    ace: float
    ie: float
    de: float
    is_takeover: bool
    threshold: float
    regime_severities: Dict[str, float]


class AgentSentryDiagnostics:
    """
    Boundary-local temporal causal diagnostics. See module docstring for
    the real mechanism this approximates and exactly what's simplified.
    """

    def __init__(
        self,
        severity_fn: Callable[[Any], Severity],
        neutral_probe: str = "Inspect the currently available information and propose the most appropriate next step.",
        takeover_threshold: float = 0.5,
        purifier: Optional[SpotlightingDefense] = None,
    ):
        """
        severity_fn : maps a proposed action (whatever type the caller's
            dry-run next-action function returns) to an ordinal severity
            in {0, 1, 2} -- see module docstring. Domain-specific; see
            default_agentdojo_severity_fn() below for a starting point.
        neutral_probe : the task-neutral diagnostic probe x^mask -- does
            NOT restate the user's real goal (that would defeat the
            point of isolating mediator-driven behavior).
        takeover_threshold : IE > this value flags Takeover_b = 1. The
            paper's own "Temporal Causal Degradation Test" isn't
            reproduced here -- see module docstring.
        purifier : defense used to construct the sanitized mediator
            r_b^(san). Defaults to SpotlightingDefense(datamarking) if
            not given.
        """
        self.severity_fn = severity_fn
        self.neutral_probe = neutral_probe
        self.takeover_threshold = takeover_threshold
        self.purifier = purifier or SpotlightingDefense(variant="datamarking")

    def diagnose(
        self,
        dry_run_next_action: Callable[[str, str], Any],
        real_user_goal: str,
        real_mediator_content: str,
    ) -> CausalDiagnosticResult:
        """
        dry_run_next_action(user_input, mediator_content) -> action :
            a SIDE-EFFECT-FREE callback the caller supplies -- must not
            commit any tool calls, matching the paper's dry-run
            requirement (Section 2.4, Challenge 3: naively probing a
            real agent by re-running steps can itself trigger external
            side effects, which is unacceptable for a defense). This
            module has no way to enforce that itself; it's a contract
            the caller's callback must honor.
        """
        sanitized_mediator = self.purifier.apply(real_mediator_content).transformed_content

        a_orig = dry_run_next_action(real_user_goal, real_mediator_content)
        a_mask = dry_run_next_action(self.neutral_probe, real_mediator_content)
        a_mask_san = dry_run_next_action(self.neutral_probe, sanitized_mediator)
        a_orig_san = dry_run_next_action(real_user_goal, sanitized_mediator)

        mu_orig = float(self.severity_fn(a_orig))
        mu_mask = float(self.severity_fn(a_mask))
        mu_mask_san = float(self.severity_fn(a_mask_san))
        mu_orig_san = float(self.severity_fn(a_orig_san))

        ace = mu_orig - mu_mask
        ie = mu_mask - mu_mask_san
        de = mu_orig_san - mu_mask_san

        return CausalDiagnosticResult(
            ace=ace,
            ie=ie,
            de=de,
            is_takeover=ie > self.takeover_threshold,
            threshold=self.takeover_threshold,
            regime_severities={
                "orig": mu_orig, "mask": mu_mask,
                "mask_sanitized": mu_mask_san, "orig_sanitized": mu_orig_san,
            },
        )

    def purify(self, mediator_content: str) -> str:
        """The Purify() operator used for safe continuation (Eq. 9) when
        diagnose() flags a takeover -- same sanitization used internally
        to construct the counterfactual regimes above, exposed here so a
        caller can apply it to the REAL episode's context once takeover
        is confirmed."""
        return self.purifier.apply(mediator_content).transformed_content


def default_agentdojo_severity_fn(exfil_tool_names: set) -> Callable[[Any], Severity]:
    """
    Reasonable STARTING severity classifier for AgentDojo actions --
    domain-specific by the paper's own design (Section 3.2); review
    against the real registered tools per suite before trusting this,
    don't assume it's correct out of the box.

    exfil_tool_names : names of tools this classifier should treat as
        "high-impact" (Section 3.2's T_exfil) -- e.g. send_email,
        send_money, delete_file. Everything else that's still a tool
        call counts as "diagnostic" (T_diag, e.g. read-only queries); no
        tool call at all is severity 0.

    Expects `action` objects exposing `.tool_calls` (a list of objects
    with a `.function` name attribute, or None) -- matching this
    project's own AgentDojo FunctionCall convention
    (environments/agentdojo_llm.py), not the raw paper's black-box
    action representation.
    """

    def severity_fn(action) -> Severity:
        tool_calls = getattr(action, "tool_calls", None)
        if not tool_calls:
            return 0
        for call in tool_calls:
            name = getattr(call, "function", None)
            if name in exfil_tool_names:
                return 2
        return 1

    return severity_fn
