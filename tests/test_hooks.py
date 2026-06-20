"""
tests/test_hooks.py
-------------------
Unit tests for the ActivationExtractor.

These tests use a tiny synthetic transformer (2 layers, 16-dim hidden state)
so they run in under a second on CPU — no HPC or model weights required.

Run with:
    python -m pytest tests/test_hooks.py -v
or:
    python tests/test_hooks.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make local imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.hooks import ActivationExtractor, extract_activations, _get_layer_list


# ---------------------------------------------------------------------------
# Minimal transformer stub
# ---------------------------------------------------------------------------

class _FakeTransformerLayer(nn.Module):
    """Single linear layer standing in for a transformer block."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return (self.linear(x),)   # Return tuple, as real transformer layers do


class _FakeVLM(nn.Module):
    """Minimal VLM stub with a .model.layers attribute."""
    def __init__(self, n_layers: int = 4, dim: int = 16):
        super().__init__()

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [_FakeTransformerLayer(dim) for _ in range(n_layers)]
                )
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)[0]
                return x

        self.model = _Inner()
        self.n_layers = n_layers
        self.dim = dim

    def forward(self, x):
        return self.model(x)


def _make_input(batch=1, seq=5, dim=16):
    return torch.randn(batch, seq, dim)


class _FakeVLM_UnconventionalPath(nn.Module):
    """
    VLM stub whose layers live somewhere NOT covered by the hardcoded
    candidate paths (mimics discovering a brand-new architecture).
    Tests that _discover_layer_list() finds it anyway.
    """
    def __init__(self, n_layers: int = 6, dim: int = 16):
        super().__init__()

        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder_blocks = nn.ModuleList(
                    [_FakeTransformerLayer(dim) for _ in range(n_layers)]
                )

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_backbone = _Backbone()

        self.core = _Wrapper()
        # Also add a smaller, irrelevant ModuleList (e.g. a vision tower)
        # to confirm discovery picks the LONGEST homogeneous list, not
        # just the first one found.
        self.vision_stub = nn.ModuleList([_FakeTransformerLayer(dim) for _ in range(2)])

    def forward(self, x):
        for layer in self.core.text_backbone.decoder_blocks:
            x = layer(x)[0]
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_layer_discovery():
    """_get_layer_list should find the layers in our stub."""
    model = _FakeVLM(n_layers=4)
    layers = _get_layer_list(model)
    assert layers is not None, "Layer list not found"
    assert len(layers) == 4


def test_auto_discovery_unconventional_path():
    """
    _get_layer_list should fall back to auto-discovery and still find
    the transformer stack even when it's at a path none of the
    hardcoded candidates check (core.text_backbone.decoder_blocks).
    """
    model = _FakeVLM_UnconventionalPath(n_layers=6)
    layers = _get_layer_list(model)
    assert layers is not None, "Auto-discovery failed to find layer list"
    assert len(layers) == 6, f"Expected 6 layers, found {len(layers)}"


def test_auto_discovery_picks_longest_homogeneous_list():
    """
    When multiple ModuleLists exist (e.g. a small vision tower stub
    alongside the real decoder stack), auto-discovery should pick the
    longest homogeneous one — i.e. the actual transformer stack, not
    an incidental shorter list.
    """
    model = _FakeVLM_UnconventionalPath(n_layers=6)
    layers = _get_layer_list(model)
    assert len(layers) == 6, "Did not select the longer of the two ModuleLists"


def test_explicit_target_layers():
    """Extractor respects explicitly provided layer indices."""
    model = _FakeVLM(n_layers=4)
    ext = ActivationExtractor(model, target_layers=[0, 2])
    assert ext.target_layers == [0, 2]


def test_heuristic_target_layers():
    """Default heuristic selects a subset of late-middle layers."""
    model = _FakeVLM(n_layers=32)
    ext = ActivationExtractor(model)
    # All selected layers should be in the 50-90% range
    n = 32
    lo, hi = int(n * 0.50), int(n * 0.90)
    for idx in ext.target_layers:
        assert lo <= idx < hi, f"Layer {idx} outside expected range [{lo}, {hi})"


def test_activation_shape_last():
    """'last' aggregation yields [hidden_dim] tensors."""
    model = _FakeVLM(n_layers=4, dim=16)
    x = _make_input(seq=5, dim=16)

    with ActivationExtractor(model, target_layers=[1, 3], token_aggregation="last") as ext:
        _ = model(x)

    acts = ext.get_activations()
    assert set(acts.keys()) == {1, 3}
    for idx, t in acts.items():
        assert t.shape == (16,), f"Layer {idx}: expected (16,), got {t.shape}"


def test_activation_shape_mean():
    """'mean' aggregation also yields [hidden_dim] tensors."""
    model = _FakeVLM(n_layers=4, dim=16)
    x = _make_input(seq=5, dim=16)

    with ActivationExtractor(model, target_layers=[0], token_aggregation="mean") as ext:
        _ = model(x)

    acts = ext.get_activations()
    assert acts[0].shape == (16,)


def test_activation_shape_full():
    """'full' aggregation preserves sequence dimension."""
    model = _FakeVLM(n_layers=4, dim=16)
    seq = 7
    x = _make_input(seq=seq, dim=16)

    with ActivationExtractor(model, target_layers=[2], token_aggregation="full") as ext:
        _ = model(x)

    acts = ext.get_activations()
    assert acts[2].shape == (seq, 16)


def test_hooks_removed_after_context():
    """Hooks must be removed after the context manager exits."""
    model = _FakeVLM(n_layers=4)
    ext = ActivationExtractor(model, target_layers=[1])

    with ext:
        pass  # No forward pass needed to test cleanup

    assert not ext._active
    assert len(ext._handles) == 0


def test_hooks_removed_on_exception():
    """Hooks must be removed even if forward pass raises an exception."""
    model = _FakeVLM(n_layers=4)
    ext = ActivationExtractor(model, target_layers=[1])

    try:
        with ext:
            raise RuntimeError("Simulated inference failure")
    except RuntimeError:
        pass

    assert not ext._active, "Hooks not removed after exception"
    assert len(ext._handles) == 0


def test_clear_frees_memory():
    """clear() removes stored activations."""
    model = _FakeVLM(n_layers=4, dim=16)
    x = _make_input()

    with ActivationExtractor(model, target_layers=[0, 1]) as ext:
        _ = model(x)

    assert len(ext._activations) == 2
    ext.clear()
    assert len(ext._activations) == 0


def test_get_activation_matrix():
    """get_activation_matrix() stacks layers correctly."""
    model = _FakeVLM(n_layers=4, dim=16)
    x = _make_input()

    with ActivationExtractor(model, target_layers=[0, 1, 2], token_aggregation="last") as ext:
        _ = model(x)

    mat = ext.get_activation_matrix()
    assert mat.shape == (3, 16), f"Expected (3, 16), got {mat.shape}"


def test_no_double_registration():
    """Registering hooks twice should raise RuntimeError."""
    model = _FakeVLM(n_layers=4)
    ext = ActivationExtractor(model, target_layers=[0])
    ext.register()
    try:
        raised = False
        try:
            ext.register()
        except RuntimeError:
            raised = True
        assert raised, "Expected RuntimeError on double registration"
    finally:
        ext.remove()


def test_extract_activations_context_manager():
    """Convenience extract_activations() wrapper works correctly."""
    model = _FakeVLM(n_layers=4, dim=16)
    x = _make_input()

    with extract_activations(model, target_layers=[0, 3]) as ext:
        _ = model(x)

    acts = ext.get_activations()
    assert 0 in acts and 3 in acts


def test_cpu_tensors():
    """Activations are always stored on CPU."""
    model = _FakeVLM(n_layers=4, dim=16)
    x = _make_input()

    with ActivationExtractor(model, target_layers=[1]) as ext:
        _ = model(x)

    acts = ext.get_activations()
    for t in acts.values():
        assert t.device.type == "cpu", f"Expected CPU tensor, got {t.device}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_layer_discovery,
        test_auto_discovery_unconventional_path,
        test_auto_discovery_picks_longest_homogeneous_list,
        test_explicit_target_layers,
        test_heuristic_target_layers,
        test_activation_shape_last,
        test_activation_shape_mean,
        test_activation_shape_full,
        test_hooks_removed_after_context,
        test_hooks_removed_on_exception,
        test_clear_frees_memory,
        test_get_activation_matrix,
        test_no_double_registration,
        test_extract_activations_context_manager,
        test_cpu_tensors,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)