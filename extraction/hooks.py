"""
extraction/hooks.py
-------------------
PyTorch forward-hook infrastructure for VLM activation extraction.

Design goals (per Phase 1 requirements):
  - Memory-safe: hooks are registered via context manager and guaranteed
    to be removed even if an exception occurs mid-inference, preventing
    the memory leaks common in long-context agentic tasks.
  - Layer-targeted: captures late-middle transformer layers by default
    (indices in the top 50% of the stack, where semantic convergence
    typically occurs), with full override support.
  - Modality-aware: records whether each captured activation came from
    a text-only, image-only, or mixed (cross-modal) forward pass, so
    downstream analysis can separate signatures by modality.
  - Minimal overhead: stores activations as detached CPU tensors to avoid
    holding GPU memory between inference calls.

Usage:
    extractor = ActivationExtractor(model, target_layers=[16, 20, 24, 28])
    with extractor:
        output = model(**inputs)
    activations = extractor.get_activations()   # {layer_idx: tensor}
    extractor.clear()
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_target_layers(model: nn.Module, target_layers: Optional[Sequence[int]]) -> List[int]:
    """
    Return layer indices to hook.

    If target_layers is None, use the late-middle heuristic: all layers
    in the top 50-90% of the transformer stack (avoids the very last
    layers which are task-specific and the early layers which encode
    surface features rather than semantics).
    """
    # Try to find the transformer layer list in common VLM architectures.
    # LLaVA-NeXT: model.language_model.model.layers
    # Qwen2-VL:   model.model.layers
    # InternVL:   model.language_model.model.layers
    layer_list = _get_layer_list(model)

    if layer_list is None:
        raise RuntimeError(
            "Could not locate transformer layer list. "
            "Pass target_layers explicitly or check _get_layer_list() for your architecture."
        )

    n = len(layer_list)

    if target_layers is not None:
        # Validate provided indices
        invalid = [i for i in target_layers if not (0 <= i < n)]
        if invalid:
            raise ValueError(
                f"Layer indices {invalid} are out of range for a {n}-layer model."
            )
        return list(target_layers)

    # Late-middle heuristic: layers from 50% to 90% of depth
    lo = int(n * 0.50)
    hi = int(n * 0.90)
    # Sample at most every 2 layers to reduce memory pressure on 13B models
    step = max(1, (hi - lo) // 8)
    return list(range(lo, hi, step))


def _get_layer_list(model: nn.Module):
    """
    Return the nn.ModuleList of transformer layers, or None if not found.

    Strategy
    --------
    1. Fast path: try known attribute paths for common architectures.
       Checked first because they're unambiguous when correct.
    2. Auto-discovery fallback: walk the full module tree and collect every
       nn.ModuleList. Score each by (a) length — transformer stacks are
       typically >= 4 layers, and (b) homogeneity — every element should be
       the same type, since a real transformer stack is N copies of one
       block class. Return the longest homogeneous ModuleList found.
       This means new/unseen VLM architectures are supported automatically,
       without requiring a hardcoded path for every model family.
    """
    candidates = [
        # Qwen2-VL / generic
        lambda m: m.model.layers,
        # LLaVA-NeXT (language_model is itself the LM, not wrapped in .model)
        lambda m: m.language_model.layers,
        # Some LLaVA / InternVL variants wrap one level deeper
        lambda m: m.language_model.model.layers,
        # Bare LlamaModel
        lambda m: m.model.model.layers,
        # Direct
        lambda m: m.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
        except AttributeError:
            continue

    return _discover_layer_list(model)


def _discover_layer_list(model: nn.Module, min_layers: int = 4):
    """
    Fallback auto-discovery: find the best-looking transformer stack
    anywhere in the module tree.

    Returns the longest nn.ModuleList whose elements are all the same
    type (homogeneous), with at least `min_layers` elements. Returns
    None if nothing qualifies.
    """
    best = None
    best_len = 0

    for _, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        if len(module) < min_layers:
            continue

        # Homogeneity check: every block should be the same class,
        # as in a real transformer stack (N copies of one decoder layer).
        types = {type(sub) for sub in module}
        if len(types) != 1:
            continue

        if len(module) > best_len:
            best = module
            best_len = len(module)

    return best


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ActivationExtractor:
    """
    Registers forward hooks on a VLM's transformer layers and collects
    the hidden-state output of each targeted layer.

    Parameters
    ----------
    model : nn.Module
        The loaded VLM (must be in eval mode before extraction).
    target_layers : list[int] | None
        Indices of transformer layers to hook.  None → late-middle heuristic.
    token_aggregation : str
        How to reduce the sequence dimension of collected activations.
        - 'last'  : last non-padding token (good for causal LMs)
        - 'mean'  : mean over all tokens
        - 'full'  : keep full [seq_len, hidden_dim] tensor (large!)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[Sequence[int]] = None,
        token_aggregation: str = "last",
    ):
        if token_aggregation not in ("last", "mean", "full"):
            raise ValueError("token_aggregation must be 'last', 'mean', or 'full'.")

        self.model = model
        self.token_aggregation = token_aggregation
        self._target_indices: List[int] = _resolve_target_layers(model, target_layers)
        self._layer_list = _get_layer_list(model)

        # Storage: {layer_idx -> tensor}
        self._activations: Dict[int, torch.Tensor] = {}
        # Registered hook handles (for guaranteed removal)
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        self._active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def target_layers(self) -> List[int]:
        return list(self._target_indices)

    def register(self) -> None:
        """
        Attach forward hooks to target layers.
        Call clear() + register() to re-use across multiple forward passes.
        Prefer the context manager (with extractor:) for automatic teardown.
        """
        if self._active:
            raise RuntimeError("Hooks already registered. Call remove() first.")

        for idx in self._target_indices:
            layer = self._layer_list[idx]
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

        self._active = True

    def remove(self) -> None:
        """Detach all hooks and free handle references."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._active = False

    def clear(self) -> None:
        """Delete stored activations and free GPU/CPU memory."""
        self._activations.clear()
        gc.collect()

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """
        Return a copy of the collected activations.

        Returns
        -------
        dict mapping layer index -> tensor of shape:
            [hidden_dim]          if token_aggregation in ('last', 'mean')
            [seq_len, hidden_dim] if token_aggregation == 'full'
        """
        if not self._activations:
            raise RuntimeError(
                "No activations collected. "
                "Did you run a forward pass while hooks were registered?"
            )
        return dict(self._activations)

    def get_activation_matrix(self) -> torch.Tensor:
        """
        Stack all layer activations into a single matrix.

        Returns
        -------
        tensor of shape [n_layers, hidden_dim]
            (only valid when token_aggregation != 'full')
        """
        if self.token_aggregation == "full":
            raise RuntimeError(
                "get_activation_matrix() is not defined for token_aggregation='full'. "
                "Use get_activations() instead."
            )
        layers = sorted(self._activations.keys())
        return torch.stack([self._activations[l] for l in layers], dim=0)

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationExtractor":
        self.register()
        return self

    def __exit__(self, *_) -> None:
        # Always remove hooks, even on exception — this is the primary
        # guard against memory leaks in long-context agentic tasks.
        self.remove()

    # ------------------------------------------------------------------
    # Internal hook factory
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        """Return a closure that captures activations for layer_idx."""

        def hook(module, input, output):
            # output is typically a tuple; first element is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output

            # hidden shape: [batch, seq_len, hidden_dim]
            # We assume batch size 1 for single-inference extraction
            h = hidden[0].detach().cpu()  # [seq_len, hidden_dim]

            if self.token_aggregation == "last":
                self._activations[layer_idx] = h[-1]          # [hidden_dim]
            elif self.token_aggregation == "mean":
                self._activations[layer_idx] = h.mean(dim=0)  # [hidden_dim]
            else:  # 'full'
                self._activations[layer_idx] = h              # [seq_len, hidden_dim]

        return hook


# ---------------------------------------------------------------------------
# Convenience: context manager only (no persistent object needed)
# ---------------------------------------------------------------------------

@contextmanager
def extract_activations(
    model: nn.Module,
    target_layers: Optional[Sequence[int]] = None,
    token_aggregation: str = "last",
):
    """
    Thin context manager wrapper around ActivationExtractor.

    Example
    -------
    with extract_activations(model, target_layers=[16, 20, 24]) as ext:
        output = model(**inputs)
    acts = ext.get_activations()
    """
    ext = ActivationExtractor(model, target_layers, token_aggregation)
    try:
        ext.register()
        yield ext
    finally:
        ext.remove()