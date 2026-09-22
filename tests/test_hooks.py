"""
tests/test_hooks.py
-----------------------
Regression tests for the capture-timing fix in extraction/hooks.py.

Background: a forward hook fires on EVERY forward call, and
model.generate() with KV caching calls forward() once for the prefill
(full prompt) and once more per newly generated token (decode steps,
seq_len=1 each). The original implementation overwrote
self._activations[layer_idx] unconditionally on every hook firing, so
after a full generate() call the stored activation held the hidden
state of the LAST GENERATED TOKEN, not the prompt-end/prefill state
this project's activation-delta methodology actually needs (chapter2.tex:
displacement between the model's state before/after processing
untrusted content).

These tests use a REAL (if tiny, random-weight) Qwen3-VL architecture
-- not a generic fake nn.Module -- specifically so a real mismatch in
how a real VLM nests its decoder layers or handles KV-cached generate()
would show up here, the same way it would on the full-size checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.hooks import ActivationExtractor


def _build_tiny_qwen3vl(num_hidden_layers=6):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig, Qwen3VLVisionConfig,
    )

    text_cfg = Qwen3VLTextConfig(
        vocab_size=100, hidden_size=16, intermediate_size=32,
        num_hidden_layers=num_hidden_layers, num_attention_heads=2, num_key_value_heads=1,
    )
    vision_cfg = Qwen3VLVisionConfig(
        depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
        patch_size=14, in_channels=3,
    )
    config = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    return model


def test_hook_fires_multiple_times_under_generate():
    """Confirms the actual mechanism this fix addresses is real: a hook
    on a real decoder layer fires more than once for a single generate()
    call with max_new_tokens > 1, not just once."""
    model = _build_tiny_qwen3vl()
    layer = model.model.language_model.layers[2]
    fire_count = [0]

    def counting_hook(module, inp, out):
        fire_count[0] += 1

    handle = layer.register_forward_hook(counting_hook)
    input_ids = torch.randint(1, 99, (1, 5))
    with torch.no_grad():
        model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
    handle.remove()

    assert fire_count[0] > 1, (
        f"hook fired only {fire_count[0]} time(s) -- expected multiple firings "
        "(1 prefill + N decode steps) under generate() with KV caching"
    )


def test_first_and_last_capture_step_genuinely_differ():
    """capture_step='first' and 'last' must produce DIFFERENT tensors on
    the same generate() call -- if they're ever identical, the fix has
    regressed into a no-op."""
    model = _build_tiny_qwen3vl()
    input_ids = torch.randint(1, 99, (1, 5))

    ext_first = ActivationExtractor(model, target_layers=[2], token_aggregation="last", capture_step="first")
    with ext_first:
        with torch.no_grad():
            model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
        first_act = ext_first.get_activations()[2].clone()

    ext_last = ActivationExtractor(model, target_layers=[2], token_aggregation="last", capture_step="last")
    with ext_last:
        with torch.no_grad():
            model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
        last_act = ext_last.get_activations()[2].clone()

    assert not torch.equal(first_act, last_act), (
        "capture_step='first' and 'last' produced identical tensors on identical "
        "input -- the prefill-vs-final-decode-step distinction has been lost"
    )


def test_multiturn_register_clear_cycle_resets_correctly():
    """The real usage pattern: LocalVLMPipelineElement.query() reuses ONE
    extractor across multiple turns via register()/clear() cycles (the
    context manager does register()/remove(), clear() is called
    separately between turns). Each turn must capture ITS OWN prefill
    state, not silently skip because a stale _captured_this_cycle set
    leaked from a previous turn."""
    model = _build_tiny_qwen3vl()
    extractor = ActivationExtractor(model, target_layers=[2], token_aggregation="last", capture_step="first")

    turn1_ids = torch.randint(1, 99, (1, 5))
    with extractor:
        with torch.no_grad():
            model.generate(input_ids=turn1_ids, max_new_tokens=4, do_sample=False)
        turn1_act = extractor.get_activations()[2].clone()
    extractor.clear()

    turn2_ids = torch.randint(1, 99, (1, 5))
    with extractor:
        with torch.no_grad():
            model.generate(input_ids=turn2_ids, max_new_tokens=4, do_sample=False)
        turn2_act = extractor.get_activations()[2].clone()
    extractor.clear()

    assert not torch.equal(turn1_act, turn2_act), (
        "turn 2 captured the same activation as turn 1 despite different input -- "
        "the per-cycle capture set is not resetting correctly between turns"
    )


def test_single_forward_call_unaffected_by_fix():
    """Phase 1 baseline collection uses a single model(**inputs) call, no
    generate() -- the hook fires exactly once, so 'first' and the old
    unconditional-overwrite behavior must be IDENTICAL here. This is the
    concrete guarantee that Phase 1 baseline data does NOT need to be
    recollected because of this fix."""
    model = _build_tiny_qwen3vl()
    input_ids = torch.randint(1, 99, (1, 8))

    extractor = ActivationExtractor(model, target_layers=[2], token_aggregation="last", capture_step="first")
    with extractor:
        with torch.no_grad():
            model(input_ids=input_ids)
        act = extractor.get_activations()[2].clone()

    # Independently recompute via a raw hook with unconditional overwrite
    # (the OLD behavior) and confirm it lands on the exact same tensor.
    raw = {}
    def old_style_hook(module, inp, out):
        h = (out[0] if isinstance(out, tuple) else out)[0].detach().cpu()
        raw["h"] = h[-1]
    layer = model.model.language_model.layers[2]
    handle = layer.register_forward_hook(old_style_hook)
    with torch.no_grad():
        model(input_ids=input_ids)
    handle.remove()

    assert torch.equal(act, raw["h"]), (
        "single forward-call capture differs from old unconditional-overwrite "
        "behavior -- Phase 1 baseline data would need recollection, which the "
        "fix's docstring claims should NOT be the case"
    )


if __name__ == "__main__":
    tests = [
        test_hook_fires_multiple_times_under_generate,
        test_first_and_last_capture_step_genuinely_differ,
        test_multiturn_register_clear_cycle_resets_correctly,
        test_single_forward_call_unaffected_by_fix,
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
