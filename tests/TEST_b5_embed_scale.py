#!/usr/bin/env python3
"""
TEST_b5_embed_scale.py
----------------------
Settles audit item B5 (GCG suffix gradient under-scaled on Gemma) WITHOUT a
biggpu slot. The behaviour is architectural -- Gemma 3's word-embedding module
multiplies the lookup by embed_scale = hidden_size**0.5 inside its forward, so
`one_hot @ embed.weight` (raw weight) bypasses a scale that `embed(ids)` (module
forward) applies. That holds for a TINY Gemma config too, so this runs on CPU
on bigbatch.

It checks three things:
  1. Does embed_layer(ids) differ from one_hot @ embed_layer.weight, and by what
     factor? (equals embed_scale when the scaled-embedding bug is present)
  2. Does multiplying by getattr(embed_layer, "embed_scale", 1.0) reconcile them?
  3. Loss parity: model(input_ids) vs model(inputs_embeds=embed_layer(ids)) --
     confirms the model expects ALREADY-SCALED inputs_embeds, i.e. the eval path
     is right and the gradient path must match it.

Usage A (preferred -- your real loader, tiny or full, CPU is fine):
    from extraction.model_loader import <your_gemma_loader>
    model, processor = <loader>(...)          # on CPU
    from TEST_b5_embed_scale import check
    check(model, processor)

Usage B (no loader -- build a tiny Gemma3 text config locally):
    python -I TEST_b5_embed_scale.py           # tries a tiny Gemma3 config

CPU-only, loads no checkpoint in usage B. Safe on the login node or bigbatch.
"""

from __future__ import annotations

import torch


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def check(model, processor) -> bool:
    """Returns True if the embed-scale fix is needed for this model."""
    model.eval()
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight
    device = embed_weight.device
    vocab = embed_weight.shape[0]

    tok = _tokenizer(processor)
    try:
        ids = tok.encode("balance transfer to account", add_special_tokens=False)
    except TypeError:
        ids = tok.encode("balance transfer to account")
    ids = torch.tensor(ids[:8] or [1, 2, 3], device=device, dtype=torch.long)

    # (1) module-forward lookup vs raw-weight one-hot product
    with torch.no_grad():
        via_module = embed_layer(ids)                       # scaled iff scaled-embedding module
        one_hot = torch.zeros(len(ids), vocab, device=device, dtype=embed_weight.dtype)
        one_hot.scatter_(1, ids.unsqueeze(1), 1.0)
        via_weight = one_hot @ embed_weight                 # raw weight, never scaled

    same = torch.allclose(via_module, via_weight, atol=1e-4, rtol=1e-3)
    declared = float(getattr(embed_layer, "embed_scale", 1.0))
    # empirical factor (median of elementwise ratio on non-tiny entries)
    mask = via_weight.abs() > 1e-6
    factor = float((via_module[mask] / via_weight[mask]).median()) if mask.any() else 1.0

    print("=" * 64)
    print("B5 embed-scale check")
    print("=" * 64)
    print(f"declared embed_scale attr : {declared:.6g}")
    print(f"hidden_size**0.5          : {embed_weight.shape[1] ** 0.5:.6g}")
    print(f"embed(ids) == one_hot@W   : {same}")
    print(f"empirical scale factor    : {factor:.6g}")

    # (3) loss parity: does the model expect scaled inputs_embeds?
    full = ids.unsqueeze(0)
    labels = ids.unsqueeze(0).clone()
    with torch.no_grad():
        loss_id = float(model(input_ids=full, labels=labels).loss)
        loss_emb_scaled = float(model(inputs_embeds=embed_layer(full), labels=labels).loss)
        loss_emb_raw = float(model(inputs_embeds=(one_hot @ embed_weight).unsqueeze(0),
                                   labels=labels).loss)
    print(f"loss(input_ids)                 : {loss_id:.5f}")
    print(f"loss(inputs_embeds=embed(ids))  : {loss_emb_scaled:.5f}   <- must match input_ids")
    print(f"loss(inputs_embeds=one_hot@W)   : {loss_emb_raw:.5f}   <- wrong if it differs")

    needs_fix = (not same) and abs(factor - 1.0) > 1e-3
    print("-" * 64)
    if needs_fix:
        print(f"VERDICT: fix NEEDED. Suffix embeds must be scaled by {factor:.4g}.")
        print("         Apply:  suffix_embeds = (one_hot @ embed_weight) * embed_scale")
        print("         in _suffix_gradient (gcg.py) and _combined_gradient (obfuscated).")
    else:
        print("VERDICT: no scale divergence for this model; fix is a no-op here.")
    print("Expected: NEEDED on Gemma 3; no-op on Qwen/InternVL/LLaVA.")
    return needs_fix


def _tiny_gemma():
    """Build a tiny random-weight Gemma3 text model on CPU (no checkpoint)."""
    from transformers import AutoTokenizer
    try:
        from transformers import Gemma3TextConfig, Gemma3TextModel  # text-only
        from transformers import Gemma3ForCausalLM  # noqa: F401
        cfg = Gemma3TextConfig(
            vocab_size=512, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
            head_dim=16, sliding_window=32, max_position_embeddings=128,
        )
        from transformers import Gemma3ForCausalLM as _LM
        model = _LM(cfg)
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Could not build a tiny Gemma3 text model from the installed "
            f"transformers ({type(e).__name__}: {e}). Use usage A: pass your own "
            "loaded (model, processor) to check()."
        )
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", use_fast=True)
    return model, tok


if __name__ == "__main__":
    m, p = _tiny_gemma()
    check(m, p)
