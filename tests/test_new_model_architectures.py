"""
tests/test_new_model_architectures.py
-----------------------------------------
Confidence tests for InternVL3-8B-hf and Qwen3-VL-8B-Instruct, built
specifically to answer "will these actually work when their v2 matrix
job's turn comes up" as concretely as possible WITHOUT real downloaded
weights (this environment has no huggingface.co access -- see
model_loader.py's own docstring on this exact limitation).

What these tests actually verify, and why it's stronger evidence than a
generic fake nn.Module: every model here is built from the REAL
transformers classes (Qwen3VLForConditionalGeneration,
InternVLForConditionalGeneration) with tiny configs (hidden_size=16,
a handful of layers) -- genuinely random weights, but the REAL module
hierarchy, REAL attention/MLP structure, REAL forward-pass wiring. This
means extraction/hooks.py's target-layer auto-discovery and
ActivationExtractor's hook mechanism are tested against each
architecture's ACTUAL structure, not a generic stand-in -- a real
mismatch in how Qwen3-VL or InternVL nests its decoder layers would
show up here, the same way it would on the full-size real checkpoint.

What these tests can NOT verify, and this is a real, stated limit, not
an oversight: whether the actual pretrained weights download
correctly, whether the real tokenizer/chat template round-trips
correctly, whether a real generate() call on full-size weights produces
sane text. That needs the cluster's real HF access -- see
hpc_model_preflight.sh for the fast, bigbatch-targeted check that
actually answers those questions, deliberately NOT queued behind the
71-hour biggpu matrix jobs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.hooks import ActivationExtractor, _resolve_target_layers


# ---------------------------------------------------------------------------
# Real, tiny Qwen3-VL -- built once per test via these helpers, not at
# module import time, so a transformers version without these classes
# fails inside a specific test (clear signal) rather than breaking
# every test in this file via a top-level import error.
# ---------------------------------------------------------------------------
def _build_tiny_qwen3vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig, Qwen3VLVisionConfig,
    )

    text_cfg = Qwen3VLTextConfig(
        vocab_size=100, hidden_size=16, intermediate_size=32,
        num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=1,
    )
    vision_cfg = Qwen3VLVisionConfig(
        depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
        patch_size=14, in_channels=3,
    )
    config = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    return model


def _build_tiny_internvl():
    from transformers import InternVLConfig, InternVLForConditionalGeneration, InternVLVisionConfig

    vision_cfg = InternVLVisionConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, image_size=64, patch_size=16,
    )
    text_cfg = {
        "model_type": "qwen2", "vocab_size": 100, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 8,
        "num_attention_heads": 2, "num_key_value_heads": 1,
    }
    config = InternVLConfig(vision_config=vision_cfg, text_config=text_cfg)
    model = InternVLForConditionalGeneration(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Layer auto-discovery against REAL architecture
# ---------------------------------------------------------------------------
def test_qwen3vl_layer_discovery_finds_real_decoder_layers():
    model = _build_tiny_qwen3vl()
    layers = _resolve_target_layers(model, None)
    assert len(layers) > 0, "found no layers at all on real Qwen3-VL architecture"
    assert all(isinstance(l, int) for l in layers)
    # 8 real decoder layers -- late-middle heuristic should land inside
    # that range, not spill outside it or degenerate to layer 0.
    assert all(0 <= l < 8 for l in layers), f"layers {layers} outside the real 8-layer decoder stack"


def test_internvl_layer_discovery_finds_real_decoder_layers():
    model = _build_tiny_internvl()
    layers = _resolve_target_layers(model, None)
    assert len(layers) > 0, "found no layers at all on real InternVL architecture"
    assert all(isinstance(l, int) for l in layers)
    assert all(0 <= l < 8 for l in layers), f"layers {layers} outside the real 8-layer decoder stack"


# ---------------------------------------------------------------------------
# ActivationExtractor hooks actually firing on a REAL forward pass
# ---------------------------------------------------------------------------
def test_qwen3vl_hooks_capture_real_activations():
    model = _build_tiny_qwen3vl()
    extractor = ActivationExtractor(model, target_layers=None, token_aggregation="last")
    input_ids = torch.randint(1, 99, (1, 10))

    with extractor:
        with torch.no_grad():
            model(input_ids=input_ids)
        activations = extractor.get_activations()

    assert set(activations.keys()) == set(extractor.target_layers)
    assert all(v.numel() > 0 and torch.isfinite(v).all() for v in activations.values())


def test_internvl_hooks_capture_real_activations():
    model = _build_tiny_internvl()
    extractor = ActivationExtractor(model, target_layers=None, token_aggregation="last")
    input_ids = torch.randint(1, 99, (1, 10))

    with extractor:
        with torch.no_grad():
            model(input_ids=input_ids)
        activations = extractor.get_activations()

    assert set(activations.keys()) == set(extractor.target_layers)
    assert all(v.numel() > 0 and torch.isfinite(v).all() for v in activations.values())


# ---------------------------------------------------------------------------
# model_loader.py's loader functions call the real from_pretrained API
# with the right arguments -- mocked, so no network/weights needed, but
# genuinely checks OUR code's call shape against what the real classes
# expect (confirmed earlier this session by reaching the real network
# wall cleanly, not a code error -- these tests lock that in).
# ---------------------------------------------------------------------------
def test_load_qwen3vl_calls_real_api_correctly():
    from extraction.model_loader import load_qwen3vl_8b

    fake_model = MagicMock()
    fake_processor = MagicMock()
    with patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=fake_model) as mock_model, \
         patch("transformers.AutoProcessor.from_pretrained", return_value=fake_processor) as mock_proc:
        model, processor = load_qwen3vl_8b(hpc=True)

    assert model is fake_model and processor is fake_processor
    call_args, call_kwargs = mock_model.call_args
    assert call_args[0] == "Qwen/Qwen3-VL-8B-Instruct"
    assert call_kwargs["device_map"] == "auto"
    assert call_kwargs["attn_implementation"] == "eager"
    assert call_kwargs["torch_dtype"] == torch.bfloat16
    mock_proc.assert_called_once_with("Qwen/Qwen3-VL-8B-Instruct")


def test_load_internvl_calls_real_api_correctly():
    from extraction.model_loader import load_internvl_8b

    fake_model = MagicMock()
    fake_processor = MagicMock()
    with patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=fake_model) as mock_model, \
         patch("transformers.AutoProcessor.from_pretrained", return_value=fake_processor) as mock_proc:
        model, processor = load_internvl_8b(hpc=True)

    assert model is fake_model and processor is fake_processor
    call_args, call_kwargs = mock_model.call_args
    assert call_args[0] == "OpenGVLab/InternVL3-8B-hf"
    assert call_kwargs["device_map"] == "auto"
    assert call_kwargs["attn_implementation"] == "eager"
    mock_proc.assert_called_once_with("OpenGVLab/InternVL3-8B-hf")


def test_run_attack_matrix_dispatch_routes_to_correct_loaders():
    from experiments.run_attack_matrix import _MODEL_LOADER_NAMES
    import extraction.model_loader as ml

    assert _MODEL_LOADER_NAMES["internvl"] == "load_internvl_8b"
    assert _MODEL_LOADER_NAMES["qwen3vl"] == "load_qwen3vl_8b"
    assert hasattr(ml, _MODEL_LOADER_NAMES["internvl"])
    assert hasattr(ml, _MODEL_LOADER_NAMES["qwen3vl"])


if __name__ == "__main__":
    tests = [
        test_qwen3vl_layer_discovery_finds_real_decoder_layers,
        test_internvl_layer_discovery_finds_real_decoder_layers,
        test_qwen3vl_hooks_capture_real_activations,
        test_internvl_hooks_capture_real_activations,
        test_load_qwen3vl_calls_real_api_correctly,
        test_load_internvl_calls_real_api_correctly,
        test_run_attack_matrix_dispatch_routes_to_correct_loaders,
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
