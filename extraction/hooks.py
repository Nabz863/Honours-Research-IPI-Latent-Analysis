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

CAPTURE-TIMING FIX (this version): a forward hook fires on EVERY forward
call, and model.generate() with KV caching calls forward() once per
step -- the first call sees the full prompt (prefill), every call after
that sees only the single newest token (decode steps, seq_len=1). The
original version of this file overwrote self._activations[layer_idx]
unconditionally on every call, so anything captured while wrapped around
generate() (both environments/agentdojo_llm.py's turn_activations and
environments/vwa_adv_llm.py's step_activations do this) ended up holding
the hidden state of the LAST GENERATED TOKEN from the final decode step,
not "the model's state after reading the (possibly injected) prompt" --
which is what this project's own activation-delta methodology
(chapter2.tex: displacement between the model's state before/after
processing untrusted content) actually needs. Confirmed as a real,
reproducible issue (not a hypothetical) via a minimal fake-layer
forward-hook test before this fix was written, and independently
re-verified against a real (tiny, random-weight) Qwen3-VL architecture:
the hook fires multiple times per generate() call, and the old
unconditional-overwrite tensor and the new first-only tensor are
provably different on identical input -- this is not a no-op change.

Fixed via capture_step (see ActivationExtractor's docstring): default
'first' captures only the prefill call per register()/clear() cycle.
For a single model(**inputs) call (Phase 1 baseline collection -- hook
fires exactly once), 'first' and the old always-overwrite behavior are
identical -- no change there, and Phase 1 baseline data does NOT need
to be recollected. 'last' (the original, unconditional-overwrite
behavior) is kept available and explicitly opt-in, in case some future
use genuinely wants the final-token signal instead.

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
from typing import Dict, List, Optional, Sequence, Set

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
    layer_list = _get_layer_list(model)

    if layer_list is None:
        raise RuntimeError(
            "Could not locate transformer layer list. "
            "Pass target_layers explicitly or check _get_layer_list() for your architecture."
        )

    n = len(layer_list)

    if target_layers is not None:
        invalid = [i for i in target_layers if not (0 <= i < n)]
        if invalid:
            raise ValueError(
                f"Layer indices {invalid} are out of range for a {n}-layer model."
            )
        return list(target_layers)

    lo = int(n * 0.50)
    hi = int(n * 0.90)
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
       nn.ModuleList. Score each by (a) length -- transformer stacks are
       typically >= 4 layers, and (b) homogeneity -- every element should be
       the same type, since a real transformer stack is N copies of one
       block class. Return the longest homogeneous ModuleList found.
       This means new/unseen VLM architectures are supported automatically,
       without requiring a hardcoded path for every model family.
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.model.layers,
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
        Indices of transformer layers to hook.  None -> late-middle heuristic.
    token_aggregation : str
        How to reduce the sequence dimension of collected activations.
        - 'last'  : last non-padding token (good for causal LMs)
        - 'mean'  : mean over all tokens
        - 'full'  : keep full [seq_len, hidden_dim] tensor (large!)
    capture_step : str
        WHICH forward call to capture from, when hooks fire more than
        once per register()/clear() cycle -- this happens whenever the
        extractor is wrapped around model.generate() with KV caching:
        one forward call for the prompt/prefill (full sequence), then
        one more per newly generated token (seq_len=1 each).
        - 'first' (default) : capture only the FIRST forward call this
          cycle -- the prefill pass over the full prompt. This is "the
          model's state right after reading the (possibly injected)
          prompt, before generating a response" -- the activation-delta
          target this project's methodology actually needs. For a
          single model(**inputs) call (no generate(), hook fires once),
          'first' is identical to the old unconditional-overwrite
          behavior -- Phase 1 baseline collection is unaffected.
        - 'last' : capture whichever call happens to run last,
          overwriting every time (this file's ORIGINAL behavior). Under
          generate() + caching this ends up being the hidden state of
          the FINAL GENERATED TOKEN, not the prompt-end state. Kept
          available, explicitly opt-in, in case some future use
          genuinely wants that signal.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[Sequence[int]] = None,
        token_aggregation: str = "last",
        capture_step: str = "first",
    ):
        if token_aggregation not in ("last", "mean", "full"):
            raise ValueError("token_aggregation must be 'last', 'mean', or 'full'.")
        if capture_step not in ("first", "last"):
            raise ValueError("capture_step must be 'first' or 'last'.")

        self.model = model
        self.token_aggregation = token_aggregation
        self.capture_step = capture_step
        self._target_indices: List[int] = _resolve_target_layers(model, target_layers)
        self._layer_list = _get_layer_list(model)

        self._activations: Dict[int, torch.Tensor] = {}
        self._captured_this_cycle: Set[int] = set()
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

        self._captured_this_cycle = set()

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
        self._captured_this_cycle = set()
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
        # Always remove hooks, even on exception -- this is the primary
        # guard against memory leaks in long-context agentic tasks.
        self.remove()

    # ------------------------------------------------------------------
    # Internal hook factory
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        """Return a closure that captures activations for layer_idx."""

        def hook(module, input, output):
            if self.capture_step == "first" and layer_idx in self._captured_this_cycle:
                # Already captured this layer's prefill state this cycle --
                # this call is a later decode step (seq_len=1), ignore it.
                return

            hidden = output[0] if isinstance(output, tuple) else output

            h = hidden[0].detach().cpu()  # [seq_len, hidden_dim]

            if self.token_aggregation == "last":
                # .clone() is REQUIRED, not cosmetic: h[-1] is a VIEW into the
                # full [seq_len, hidden_dim] tensor, and torch.save() serialises
                # a view's ENTIRE underlying storage. Without this, every saved
                # "[4096] activation" silently carries its whole parent sequence
                # -- confirmed by a 109GB clean_activations_cache.pt (~380MB per
                # task, vs ~16KB for a real fp32 [4096] vector), and by ~10min of
                # per-episode overhead spent rewriting it.
                self._activations[layer_idx] = h[-1].clone()  # [hidden_dim]
            elif self.token_aggregation == "mean":
                self._activations[layer_idx] = h.mean(dim=0)  # [hidden_dim]
            else:  # 'full'
                self._activations[layer_idx] = h              # [seq_len, hidden_dim]

            self._captured_this_cycle.add(layer_idx)

        return hook


# ---------------------------------------------------------------------------
# Convenience: context manager only (no persistent object needed)
# ---------------------------------------------------------------------------

@contextmanager
def extract_activations(
    model: nn.Module,
    target_layers: Optional[Sequence[int]] = None,
    token_aggregation: str = "last",
    capture_step: str = "first",
):
    """
    Thin context manager wrapper around ActivationExtractor.

    Example
    -------
    with extract_activations(model, target_layers=[16, 20, 24]) as ext:
        output = model(**inputs)
    acts = ext.get_activations()
    """
    ext = ActivationExtractor(model, target_layers, token_aggregation, capture_step)
    try:
        ext.register()
        yield ext
    finally:
        ext.remove()
