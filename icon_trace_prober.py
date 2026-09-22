"""
defenses/icon_trace_prober.py
--------------------------------
ICON-style Latent Space Trace Prober (Table 3.3, inference-time
baseline; DETECTION half only, see SIMPLIFICATIONS). Wang, "ICON:
Indirect Prompt Injection Defense for Agents based on Inference-Time
Correction," arXiv:2602.20708 (Feb 2026 preprint).

CORE IDEA (confirmed from the real paper's abstract): IPI attacks leave
distinct "over-focusing signatures" in the LATENT SPACE -- ICON's Latent
Space Trace Prober (LSTP) detects these via high-intensity scores; a
paired Mitigating Rectifier then performs "surgical attention steering"
that suppresses adversarial query-key dependencies while reinforcing
task-relevant features. Reported: 0.4% ASR (matching commercial-grade
detectors) with over 50% task utility GAIN versus prior defenses, plus
OOD generalization and extension to multimodal agents.

This maps directly onto language already in this project's own
literature review (chapter2.tex, Activation-Based Methods): "Latent
space trace probing extends this by examining activation trajectories
across multiple layers, identifying over-focusing signatures in which
the trajectory diverges from the clean-input baseline in a
characteristic direction" -- and onto infrastructure this project
already has: extraction.hooks.ActivationExtractor already captures
MULTIPLE layers per episode (e.g. LLaVA-NeXT's layers 16-27), and
clean_activations_by_task (experiments/run_attack_matrix.py) already
gives a per-task clean baseline at every one of those layers, meaning a
genuine multi-layer TRAJECTORY of distance-to-clean-baseline is already
sitting in data this project produces -- not something requiring new
capture machinery the way Attention Tracker's raw attention weights did.

SIMPLIFICATIONS FROM THE REAL PAPER, STATED EXPLICITLY:
  - This module implements the DETECTION half only (the Latent Space
    Trace Prober's core idea: a multi-layer distance-to-clean-baseline
    trajectory, checked for characteristic divergence). The Mitigating
    Rectifier's "surgical attention steering" is NOT reproduced -- that
    needs live attention-weight access and a mid-generation steering
    intervention, the same category of not-yet-attempted integration
    work flagged in attention_tracker.py.
  - The exact LSTP scoring function ("high-intensity scores") isn't
    reproduced -- not confidently extractable from available sources.
    Instead, this module summarizes the per-layer distance trajectory
    with its LINEAR SLOPE across layer index (least-squares fit) as a
    single scalar "growth rate" statistic, and calibrates a threshold on
    that statistic from genuinely clean trajectories -- same k-sigma
    calibration pattern as task_tracker.py and attention_tracker.py,
    applied to a per-episode multi-layer TRAJECTORY summary rather than
    a single-layer delta or a single attention-focus ratio.

STATUS: mechanics verified against synthetic multi-layer trajectories
(see tests/test_icon_trace_prober.py) -- slope computed correctly
against known linear trajectories, correct flagging of a trajectory with
a steep late-layer growth pattern relative to clean calibration, correct
non-flagging of a flat/bounded trajectory. NOT yet wired into a live
episode (needs clean_activations_by_task and a current episode's
turn_activations from the same real run, not synthesized here) and does
NOT implement the Mitigating Rectifier's mitigation half at all -- both
are real remaining steps, not silently done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class TraceProbeResult:
    is_over_focusing: bool
    slope: float
    threshold: float
    per_layer_distance: Dict[int, float]


class LatentSpaceTraceProber:
    """
    Multi-layer trajectory analog of ICON's Latent Space Trace Prober.
    See module docstring for exactly what's simplified.
    """

    def __init__(self, metric: str = "cosine_distance", k_sigma: float = 2.0):
        if metric not in ("l2", "cosine_distance"):
            raise ValueError(f"Unknown metric: {metric!r}")
        self.metric = metric
        self.k_sigma = k_sigma
        self.threshold: Optional[float] = None
        self._calibration_slopes: List[float] = []

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.flatten().float()
        b = b.flatten().float()
        if self.metric == "l2":
            return float(torch.norm(a - b).item())
        cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        return float(1.0 - cos_sim)

    def _trace(
        self, current_by_layer: Dict[int, torch.Tensor], clean_by_layer: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        common_layers = sorted(set(current_by_layer) & set(clean_by_layer))
        if not common_layers:
            raise ValueError("No overlapping layers between current and clean activations.")
        return {l: self._distance(current_by_layer[l], clean_by_layer[l]) for l in common_layers}

    @staticmethod
    def _slope(per_layer_distance: Dict[int, float]) -> float:
        """
        Least-squares slope of distance vs layer index -- the "growth
        rate" summary statistic this module uses in place of the real
        paper's own LSTP scoring function (see module docstring).
        """
        layers = sorted(per_layer_distance.keys())
        n = len(layers)
        if n < 2:
            raise ValueError("Need at least 2 layers to compute a trajectory slope.")
        xs = torch.tensor(layers, dtype=torch.float32)
        ys = torch.tensor([per_layer_distance[l] for l in layers], dtype=torch.float32)
        x_mean, y_mean = xs.mean(), ys.mean()
        numerator = ((xs - x_mean) * (ys - y_mean)).sum()
        denominator = ((xs - x_mean) ** 2).sum().clamp_min(1e-12)
        return float((numerator / denominator).item())

    def calibrate(self, clean_episode_traces: List[Dict[int, float]]) -> None:
        """
        clean_episode_traces : list of per-layer distance dicts from
            GENUINELY CLEAN episode comparisons (e.g. one clean task's
            activations compared against ANOTHER clean task's, or a
            clean episode's own early-vs-late-turn activations) --
            establishes the natural slope distribution with no attack
            present.
        """
        self._calibration_slopes = [self._slope(trace) for trace in clean_episode_traces]
        if len(self._calibration_slopes) < 2:
            raise ValueError(
                f"Need at least 2 clean calibration traces (got {len(self._calibration_slopes)})."
            )
        mean = sum(self._calibration_slopes) / len(self._calibration_slopes)
        var = sum((s - mean) ** 2 for s in self._calibration_slopes) / len(self._calibration_slopes)
        std = var ** 0.5
        self.threshold = mean + self.k_sigma * std

    def probe(
        self,
        current_by_layer: Dict[int, torch.Tensor],
        clean_by_layer: Dict[int, torch.Tensor],
    ) -> TraceProbeResult:
        """
        current_by_layer / clean_by_layer : {layer_idx: activation}
            dicts -- directly the shape extraction.hooks.
            ActivationExtractor.get_activations() and
            clean_activations_by_task's per-task entries already use, so
            wiring this against real data needs no reshaping.
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before probe().")

        trace = self._trace(current_by_layer, clean_by_layer)
        slope = self._slope(trace)
        return TraceProbeResult(
            is_over_focusing=slope > self.threshold,
            slope=slope,
            threshold=self.threshold,
            per_layer_distance=trace,
        )
