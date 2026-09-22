"""
defenses/circuit_breaker.py
------------------------------
Inference-time analog of Circuit Breaking / Representation Rerouting
(Table 3.3, "train/inference" baseline). Zou et al., "Improving
Alignment and Robustness with Circuit Breakers," arXiv:2406.04313
(NeurIPS 2024 poster) -- correctly cited in this project's own
references.bib as zou2024.

REAL MECHANISM (confirmed from the actual paper): via a LoRA adapter
fine-tuned with "Low-Rank Representation Adaptation" (LoRRA) over 500+
training steps on paired "rerouting" (harmful) and "retain" (harmless)
datasets, the model is trained to push harmful-content representations
ORTHOGONAL to their representation in the original, unmodified model --
"catapulting" them off the data manifold, which causes generation to
short-circuit into incoherent output (repeated <EOS> tokens, per the
paper's own Figure 1) rather than complete the harmful request. This is
genuinely TRAINING-TIME: the intervention is baked into fine-tuned
weights, not something applied at inference without first training.

WHY THIS ISN'T THAT: this project has no fine-tuning infrastructure, and
building one is a materially different undertaking from everything else
in this repo (real gradient updates to model weights, a labeled
rerouting/retain dataset, hundreds of training steps, checkpointing) --
not attempted here, and not silently faked as something smaller.

INFERENCE-TIME ANALOG BUILT INSTEAD: representation engineering has a
well-precedented training-free sibling technique -- computing a
"direction" via the difference of means between two activation
distributions (e.g. Arditi et al.'s refusal-direction work, cited by
papers building on Zou et al.'s own), then intervening on that direction
at inference via projection/ablation, no gradient step required. This
project already collects paired clean vs. attacked activations for
exactly this kind of comparison (the same mechanism
obfuscated_activation.py and task_tracker.py both already use) --
reusing that data to compute a "circuit-breaker direction" (difference
of means between attacked and clean activations at a chosen layer) and,
at inference, projecting a CURRENT activation's component along that
direction toward zero whenever it exceeds a threshold, is a direct,
honest, training-free analog of "reroute the representation associated
with the bad behavior" -- not a reproduction of the paper's own
LoRA-trained mechanism, and weaker in a specific, nameable way: a single
global linear direction from mean differences is a much cruder
intervention than a fine-tuned adapter that can learn a much richer
rerouting function across the model's actual weights.

STATUS: mechanics verified against synthetic activation vectors (see
tests/test_circuit_breaker.py) -- direction computed correctly as the
normalized difference of means, projection/ablation correctly removes
the component along that direction (checked directly, not assumed),
thresholded intervention correctly triggers only when projection exceeds
calibration, correctly leaves an activation unmodified when its
projection is small. NOT yet wired into a live generation loop -- the
actual intervention point (REROUTING the activation such that it
genuinely changes what the model subsequently generates) needs a forward
hook that MODIFIES a live activation mid-forward-pass, not just analyzes
a post-hoc saved tensor; not attempted here, same "not yet a live
intervention" caveat as task_tracker.py and agent_sentry.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class RerouteResult:
    was_rerouted: bool
    projection: float
    threshold: float
    rerouted_activation: torch.Tensor


class CircuitBreakerDirection:
    """
    Training-free analog of a circuit-breaker's learned rerouting
    behavior: a single global direction (difference of means between
    attacked and clean activations), with threshold-gated
    projection/ablation at inference. See module docstring for exactly
    how and why this differs from the real paper's LoRA-trained
    mechanism.
    """

    def __init__(self, k_sigma: float = 2.0):
        """
        k_sigma : threshold = mean(clean_projections) + k_sigma *
            std(clean_projections) -- same k-sigma calibration pattern
            as task_tracker.py, applied to "how strongly does this
            activation project onto the bad-behavior direction" instead
            of an activation delta.
        """
        self.k_sigma = k_sigma
        self.direction: Optional[torch.Tensor] = None
        self.threshold: Optional[float] = None

    def fit_direction(
        self,
        attacked_activations: List[torch.Tensor],
        clean_activations: List[torch.Tensor],
    ) -> None:
        """
        Computes the circuit-breaker direction as the normalized
        difference of means: mean(attacked) - mean(clean). This is the
        SAME quantity difference-of-means/refusal-direction techniques
        in the representation-engineering literature use -- a real,
        precedented training-free technique, just not Zou et al.'s own
        (which learns a richer rerouting function via gradient descent,
        not a single linear direction).
        """
        if not attacked_activations or not clean_activations:
            raise ValueError("Need at least one attacked and one clean activation sample.")

        attacked_stack = torch.stack([a.flatten().float() for a in attacked_activations])
        clean_stack = torch.stack([a.flatten().float() for a in clean_activations])

        mean_attacked = attacked_stack.mean(dim=0)
        mean_clean = clean_stack.mean(dim=0)
        direction = mean_attacked - mean_clean
        norm = direction.norm().clamp_min(1e-12)
        self.direction = direction / norm

    def calibrate(self, clean_activations: List[torch.Tensor]) -> None:
        """
        Must be called AFTER fit_direction(). Establishes the natural
        clean-condition distribution of projection onto the
        circuit-breaker direction, for k-sigma thresholding -- a
        genuinely clean activation should project only weakly onto a
        direction defined by attacked-minus-clean means; a large
        projection at inference indicates the CURRENT activation looks
        more like the attacked distribution than the clean one.
        """
        if self.direction is None:
            raise RuntimeError("Call fit_direction() before calibrate().")

        projections = [float(torch.dot(a.flatten().float(), self.direction).item()) for a in clean_activations]
        if len(projections) < 2:
            raise ValueError(f"Need at least 2 clean calibration samples (got {len(projections)}).")
        mean = sum(projections) / len(projections)
        var = sum((p - mean) ** 2 for p in projections) / len(projections)
        std = var ** 0.5
        self.threshold = mean + self.k_sigma * std

    def maybe_reroute(self, activation: torch.Tensor) -> RerouteResult:
        """
        If the activation's projection onto the circuit-breaker
        direction exceeds the calibrated threshold, ABLATE that
        component (project it out) -- a crude, linear analog of
        "rerouting the representation off the harmful-behavior
        manifold." Otherwise returns the activation unmodified.
        """
        if self.direction is None or self.threshold is None:
            raise RuntimeError("Call fit_direction() and calibrate() before maybe_reroute().")

        flat = activation.flatten().float()
        projection = float(torch.dot(flat, self.direction).item())

        if projection <= self.threshold:
            return RerouteResult(
                was_rerouted=False, projection=projection, threshold=self.threshold,
                rerouted_activation=activation,
            )

        ablated = flat - projection * self.direction
        rerouted = ablated.reshape(activation.shape).to(activation.dtype)
        return RerouteResult(
            was_rerouted=True, projection=projection, threshold=self.threshold,
            rerouted_activation=rerouted,
        )
