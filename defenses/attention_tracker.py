"""
defenses/attention_tracker.py
--------------------------------
Attention Tracker-style distraction detector (Table 3.3, inference-time
baseline). Hung et al., "Attention Tracker: Detecting Prompt Injection
Attacks in LLMs," arXiv:2411.00348 (NAACL 2025).

CORE IDEA (confirmed from the paper's own abstract and secondary
sources): under prompt injection, specific attention heads ("important
heads") exhibit a "distraction effect" -- they shift focus AWAY from the
original instruction TOWARD the injected content. The paper's detector
tracks attention-based statistics (entropy, KL-divergence, per secondary
sources) on these important heads as a TRAINING-FREE detection signal,
reporting >95% detection accuracy and up to 10 AUROC points over prior
methods -- notably WITHOUT needing any additional LLM inference (unlike
TaskTracker's or AgentSentry's extra forward/dry-run passes here, this
only needs attention weights the real forward pass already produces, a
genuinely cheap signal by comparison).

SIMPLIFICATIONS FROM THE REAL PAPER, STATED EXPLICITLY:
  - "Important head" selection here is NOT the paper's own algorithm --
    the exact selection criterion wasn't confidently extractable from
    available sources without risking the same kind of equation-
    transcription error that had to be corrected in agent_sentry.py, so
    it wasn't attempted from a partially-rendered source. Instead,
    important heads are selected by a literal, simple proxy for their
    own stated concept: the (layer, head) pairs that most CONSISTENTLY
    concentrate attention on the instruction span under CLEAN
    calibration data -- heads that reliably "pay attention to the
    instruction" when nothing is wrong, which the distraction effect
    predicts should degrade specifically under attack.
  - The detection statistic is instruction-focus RATIO (fraction of a
    head's attention mass landing on the designated instruction token
    span) rather than the paper's own entropy/KL-divergence formulas,
    which were likewise not confidently reproducible from available
    sources. A large DROP in this ratio relative to a clean calibration
    distribution is the anomaly signal -- same k-sigma calibration
    pattern as task_tracker.py, applied to a different underlying
    quantity and on the opposite tail (distraction is a DROP in focus,
    not a rise).
  - Requires the caller to supply attention weight tensors AND the
    instruction token span explicitly -- this module has no way to
    obtain either on its own. Getting attention weights out of a live
    model needs output_attentions=True (or attention-module hooks), a
    different mechanism from extraction.hooks.ActivationExtractor's
    hidden-state hooks -- NOT wired into LocalVLMPipelineElement here.

STATUS: mechanics verified against synthetic attention tensors (see
tests/test_attention_tracker.py) -- calibration from a clean
instruction-focus distribution, correct flagging of a large synthetic
drop in focus (simulated distraction), correct non-flagging when focus
is unaffected, important-head selection ranks heads correctly by
clean-condition focus. NOT yet wired into a live model's attention
output -- that remaining integration step (output_attentions=True or
attention-module hooks, plus locating the real instruction token span
inside AgentDojo's actual chat-template-formatted prompt) is not
silently done here, same caveat as task_tracker.py and agent_sentry.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class DistractionResult:
    is_distracted: bool
    focus_ratio: float
    threshold: float
    heads_used: List[Tuple[int, int]]


class AttentionTrackerDetector:
    """
    Distraction-effect detector over attention weights. See module
    docstring for the real mechanism this approximates and exactly what
    is simplified.
    """

    def __init__(self, top_k_heads: int = 5, k_sigma: float = 3.0):
        """
        top_k_heads : how many (layer, head) pairs to select as
            "important" via select_important_heads().
        k_sigma : threshold = mean(clean_focus) - k_sigma * std(clean_focus)
            -- same k-sigma calibration philosophy as
            task_tracker.py.TaskTrackerDetector, mirrored onto the
            opposite tail since distraction is a DROP in instruction
            focus, not a rise in some distance metric.
        """
        self.top_k_heads = top_k_heads
        self.k_sigma = k_sigma
        self.important_heads: Optional[List[Tuple[int, int]]] = None
        self.threshold: Optional[float] = None
        self._calibration_ratios: List[float] = []

    @staticmethod
    def _focus_ratio(attn: torch.Tensor, instruction_span: Tuple[int, int]) -> float:
        """
        attn : 1-D attention weights over key positions for ONE head at
            the generating (last query) position -- how much of THIS
            head's attention, right now, lands on the instruction span
            versus everywhere else.
        instruction_span : (start, end) token indices (end exclusive)
            marking the original user instruction within the key
            sequence.
        """
        start, end = instruction_span
        total = attn.sum().clamp_min(1e-12)
        on_instruction = attn[start:end].sum()
        return float((on_instruction / total).item())

    def select_important_heads(
        self,
        clean_attention_samples: List[torch.Tensor],
        instruction_span: Tuple[int, int],
    ) -> None:
        """
        clean_attention_samples : list of attention tensors from CLEAN
            episodes, each shaped [num_layers, num_heads, seq_len] --
            already reduced to the generating position's attention row
            per (layer, head) by the caller, not the full [seq, seq]
            matrix -- keeps this interface shape-agnostic across
            architectures with different sequence-length conventions.

        Selects the top_k_heads (layer, head) pairs with the HIGHEST
        mean instruction-focus ratio under clean conditions -- see
        module docstring for why this is a simplified proxy for the
        paper's own "important head" concept, not their algorithm.
        """
        if not clean_attention_samples:
            raise ValueError("Need at least one clean attention sample to select important heads.")

        n_layers, n_heads, _ = clean_attention_samples[0].shape
        mean_focus = torch.zeros(n_layers, n_heads)
        for sample in clean_attention_samples:
            for l in range(n_layers):
                for h in range(n_heads):
                    mean_focus[l, h] += self._focus_ratio(sample[l, h], instruction_span)
        mean_focus /= len(clean_attention_samples)

        flat = mean_focus.flatten()
        k = min(self.top_k_heads, flat.numel())
        top_indices = torch.topk(flat, k).indices
        self.important_heads = [(int(idx) // n_heads, int(idx) % n_heads) for idx in top_indices]

    def calibrate(
        self,
        clean_attention_samples: List[torch.Tensor],
        instruction_span: Tuple[int, int],
    ) -> None:
        """
        Must be called AFTER select_important_heads(). Establishes the
        natural clean-condition distribution of average focus ratio
        across the selected important heads, for k-sigma thresholding.
        """
        if self.important_heads is None:
            raise RuntimeError("Call select_important_heads() before calibrate().")

        self._calibration_ratios = [
            self._average_focus_over_important_heads(sample, instruction_span)
            for sample in clean_attention_samples
        ]
        if len(self._calibration_ratios) < 2:
            raise ValueError(
                f"Need at least 2 clean calibration samples to estimate a distribution "
                f"(got {len(self._calibration_ratios)})."
            )
        mean = sum(self._calibration_ratios) / len(self._calibration_ratios)
        var = sum((r - mean) ** 2 for r in self._calibration_ratios) / len(self._calibration_ratios)
        std = var ** 0.5
        self.threshold = mean - self.k_sigma * std

    def _average_focus_over_important_heads(
        self, attention_sample: torch.Tensor, instruction_span: Tuple[int, int]
    ) -> float:
        ratios = [self._focus_ratio(attention_sample[l, h], instruction_span) for l, h in self.important_heads]
        return sum(ratios) / len(ratios)

    def detect(
        self,
        current_attention: torch.Tensor,
        instruction_span: Tuple[int, int],
    ) -> DistractionResult:
        """
        current_attention : same [num_layers, num_heads, seq_len] shape
            as calibration samples, for the episode/turn being checked.
        """
        if self.important_heads is None or self.threshold is None:
            raise RuntimeError("Call select_important_heads() and calibrate() before detect().")

        focus = self._average_focus_over_important_heads(current_attention, instruction_span)
        return DistractionResult(
            is_distracted=focus < self.threshold,
            focus_ratio=focus,
            threshold=self.threshold,
            heads_used=list(self.important_heads),
        )
