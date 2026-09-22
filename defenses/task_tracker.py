"""
defenses/task_tracker.py
--------------------------
TaskTracker-style activation-delta drift detector (Table 3.3,
inference-time baseline).

CITATION CORRECTION (found while grounding this, not assumed): this
project's references.bib has an "abdelnabi2025" entry citing "TaskTracker:
Taming Large Language Model Agents with Space-Sensitive Activation
Signatures" (arXiv:2412.05434, authors S. Abdelnabi/A. Zhao/M. Fritz).
That paper does not exist -- checked directly via search, not assumed;
no result anywhere under that title, ID, or author combination. The
REAL TaskTracker paper is "Get my drift? Catching LLM Task Drift with
Activation Deltas" (arXiv:2406.00799, IEEE SaTML 2025), by Sahar
Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, and
Andrew Paverd -- six authors, not three. The same wrong title/authors/ID
appear in the compiled proposal PDF's own bibliography, meaning this
error is already in the submitted proposal document, not only the
references.bib source -- worth fixing in the thesis bibliography, and
worth spot-checking the rest of references.bib for the same class of
error separately from this file. Official code/data:
github.com/microsoft/TaskTracker.

REAL MECHANISM (confirmed from the actual paper and papers citing it as
a baseline, not guessed): "extracting the activations once when the user
first prompts the LLM with a task, and then again after the LLM has
processed any external data ... contrasting these two sets of
activations via a linear probe on the activation deltas." This maps
cleanly onto AgentDojo's own multi-turn structure -- turn 0 (before any
tool result has entered context) versus a later turn (after a tool
result, potentially carrying an injected payload, has been processed) --
using LocalVLMPipelineElement.turn_activations, which already exists.
Unlike ObfuscatedActivationAttack's clean_reference_activation, this
does NOT need a separate clean-baseline episode: the comparison is
WITHIN one episode, turn 0 against a later turn.

SIMPLIFICATION FROM THE REAL PAPER, STATED EXPLICITLY: the real
TaskTracker trains a linear probe (two variants: linear classifier and a
metric-learning method) on their own released ~500K-instance activation
dataset, achieving >0.99 ROC AUC, zero-shot generalizing to unseen attack
types. This project has neither that dataset nor their trained probe.
What's implemented here is an UNTRAINED, threshold-based analog of their
core finding (activation deltas correlate strongly with task drift):
compute the same before/after delta they use, and flag drift when its
magnitude exceeds a threshold calibrated from a set of genuinely clean
episodes' own natural turn-0-vs-turn-N variation (mean + k*std), not
learned end-to-end via their probe. This is real signal from the same
underlying phenomenon, not a reproduction of their classifier -- do not
imply equivalence to their reported ROC AUC in any writeup without
re-deriving a comparable number against this project's own models.

DIFFERENT INTERFACE FROM SpotlightingDefense, DELIBERATELY: this is NOT
a defenses.base.BaseDefense. That interface transforms CONTENT before
the model ever sees it (pre-processing). TaskTracker's actual mechanism
is the opposite -- it operates on ACTIVATIONS produced AFTER a forward
pass, comparing two points within an episode's own trajectory and
flagging when they diverge. Forcing this into BaseDefense.apply(content)
would misrepresent what it actually does; it belongs with
extraction/hooks.py's world of raw tensors, not defenses/spotlighting.py's
world of text.

STATUS: mechanics verified against synthetic activation vectors (see
tests/test_task_tracker.py) -- calibration from a clean delta
distribution, correct flagging of a large synthetic delta as drift,
correct non-flagging of a near-zero delta, both distance metrics
checked. NOT yet wired into a live generation-time intervention (i.e.
this can score/flag a completed episode's turn_activations, but nothing
currently ACTS on that flag mid-generation to block or alter a
response) -- that remaining integration step is not silently done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DriftDetectionResult:
    is_drifted: bool
    delta: float
    threshold: float
    metric: str  # "l2" or "cosine_distance"
    reference_turn: int
    check_turn: int


class TaskTrackerDetector:
    """
    Threshold-based activation-delta drift detector. See module
    docstring for exactly how and why this differs from the real
    TaskTracker's trained-probe approach.
    """

    def __init__(self, metric: str = "cosine_distance", k: float = 3.0):
        """
        metric : "cosine_distance" (1 - cosine similarity -- scale-
            invariant, robust to activation-magnitude differences across
            layers/models) or "l2" (Euclidean distance -- sensitive to
            magnitude too, which can itself carry signal but is less
            comparable across layers/models).
        k : threshold = mean(clean_deltas) + k * std(clean_deltas), a
            standard k-sigma calibration. Larger k means fewer false
            positives on clean multi-turn drift (ordinary conversations
            DO shift activations turn to turn even with no injection)
            but also fewer true detections -- the same ASR/UA-style
            tradeoff the rest of this project's defenses navigate,
            just for a detector's flag rate instead of a content
            transform's utility cost.
        """
        if metric not in ("l2", "cosine_distance"):
            raise ValueError(f"Unknown metric: {metric!r}")
        self.metric = metric
        self.k = k
        self.threshold: Optional[float] = None
        self._calibration_deltas: List[float] = []

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.flatten().float()
        b = b.flatten().float()
        if self.metric == "l2":
            return float(torch.norm(a - b).item())
        cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        return float(1.0 - cos_sim)

    def calibrate(self, clean_delta_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        clean_delta_pairs : list of (turn0_activation, later_turn_activation)
            tensor pairs, drawn from GENUINELY CLEAN episodes (no
            injection present) -- this establishes what natural
            turn-to-turn drift looks like on its own, so the threshold
            isn't calibrated against attacked data, which would defeat
            the point of a detector meant to catch exactly that.
        """
        self._calibration_deltas = [self._distance(a, b) for a, b in clean_delta_pairs]
        if len(self._calibration_deltas) < 2:
            raise ValueError(
                f"Need at least 2 clean calibration pairs to estimate a distribution "
                f"(got {len(self._calibration_deltas)})."
            )
        mean = sum(self._calibration_deltas) / len(self._calibration_deltas)
        var = sum((d - mean) ** 2 for d in self._calibration_deltas) / len(self._calibration_deltas)
        std = var ** 0.5
        self.threshold = mean + self.k * std

    def detect(
        self,
        turn_activations: List[Dict[int, torch.Tensor]],
        layer: int,
        reference_turn: int = 0,
        check_turn: int = -1,
    ) -> DriftDetectionResult:
        """
        turn_activations : ONE episode's LocalVLMPipelineElement.
            turn_activations -- the real per-turn activation list already
            produced by every episode this project runs, no extra
            capture needed.
        reference_turn / check_turn : which turns to compare -- default
            is turn 0 (before any tool result) against the LAST turn
            (after everything the episode processed, injected or not).
            AgentDojo episodes vary in turn count; -1 always means
            "whatever the last turn actually was" for this episode.
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() with clean delta pairs before detect().")
        if len(turn_activations) < 2:
            raise ValueError(
                f"Need at least 2 turns to compare a reference against a later turn "
                f"(got {len(turn_activations)})."
            )

        ref_acts = turn_activations[reference_turn]
        check_acts = turn_activations[check_turn]
        if layer not in ref_acts or layer not in check_acts:
            raise KeyError(
                f"Layer {layer} not captured at reference_turn={reference_turn} or "
                f"check_turn={check_turn} (available: {sorted(ref_acts.keys())})."
            )

        delta = self._distance(ref_acts[layer], check_acts[layer])
        return DriftDetectionResult(
            is_drifted=delta > self.threshold,
            delta=delta,
            threshold=self.threshold,
            metric=self.metric,
            reference_turn=reference_turn,
            check_turn=check_turn if check_turn >= 0 else len(turn_activations) + check_turn,
        )
