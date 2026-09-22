"""
tests/test_obfuscated_activation.py
--------------------------------------
Mechanics-only smoke test for ObfuscatedActivationAttack against a fake
HF-convention text model with a REAL extraction.hooks.ActivationExtractor
(not a stand-in) -- same pattern as test_agentdojo_llm.py using the real
extractor while faking only the model. Does NOT verify against real model
weights, real tokenizer behavior, or a real Phase 1 baseline activation --
run the real thing via sbatch before trusting this for actual attacks,
same caveat as test_gcg.py / test_pixel_perturbation.py.
"""

from __future__ import annotations

import string as string_lib
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.hooks import ActivationExtractor
from attacks.gcg import GCGAttack
from attacks.obfuscated_activation import ObfuscatedActivationAttack


class _FakeLayer(nn.Module):
    """Includes cross-position mixing (mean-pool residual), same fix
    test_gcg.py needed -- without it, activations at any position would
    be mathematically independent of the suffix positions, making both
    the behavioral AND geometry loss terms meaningless regardless of
    whether the real attack code is correct."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        h = torch.tanh(self.linear(x))
        context = h.mean(dim=1, keepdim=True)
        return (h + context,)


class _FakeInner(nn.Module):
    def __init__(self, n_layers, dim):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


class _FakeCausalLM(nn.Module):
    def __init__(self, vocab_size=60, dim=16, n_layers=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.model = _FakeInner(n_layers, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds=None, labels=None, **kwargs):
        hidden = self.model(inputs_embeds)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

        class _Out:
            pass

        out = _Out()
        out.loss = loss
        return out


class _FakeTokenizer:
    def __init__(self, vocab_size=60):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [min(abs(hash(w)) % (self.vocab_size - 1), self.vocab_size - 2) + 1 for w in text.split()] or [1]

    def decode(self, ids):
        return " ".join(f"tok{i}" for i in ids)


class _FakeProcessor:
    def __init__(self, vocab_size=60):
        self.tokenizer = _FakeTokenizer(vocab_size)


def _make_attack(seed, geometry_weight=1.0, target_layer=1, **overrides):
    torch.manual_seed(seed)
    model = _FakeCausalLM(vocab_size=60, dim=16, n_layers=4)
    processor = _FakeProcessor(vocab_size=60)
    extractor = ActivationExtractor(model, target_layers=[0, 1, 2], token_aggregation="last")
    kwargs = dict(num_steps=10, suffix_length=6, topk=15, candidates_per_step=12)
    kwargs.update(overrides)
    attack = ObfuscatedActivationAttack(
        model, processor, extractor, geometry_weight=geometry_weight, target_layer=target_layer, **kwargs
    )
    return attack, model, processor, extractor


def test_requires_last_token_aggregation():
    model = _FakeCausalLM()
    extractor = ActivationExtractor(model, target_layers=[0], token_aggregation="mean")
    raised = False
    try:
        ObfuscatedActivationAttack(model, _FakeProcessor(), extractor)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for token_aggregation != 'last'"


def test_gradient_flows_through_both_terms():
    attack, model, processor, extractor = _make_attack(seed=0)
    prefix_ids = torch.tensor(processor.tokenizer.encode("context here"), dtype=torch.long)
    target_ids = torch.tensor(processor.tokenizer.encode("target text"), dtype=torch.long)
    suffix_ids = torch.full((6,), 3, dtype=torch.long)
    clean_ref = torch.randn(16)

    grad = attack._combined_gradient(model.get_input_embeddings(), prefix_ids, suffix_ids, target_ids, clean_ref)
    assert grad.shape == (6, 60)
    assert torch.any(grad != 0), "Gradient is all-zero -- suggests a detached graph somewhere"


def test_geometry_weight_zero_matches_plain_gcg_exactly():
    """The real correctness check for the 'off switch': with
    geometry_weight=0.0, the combined gradient must be numerically
    IDENTICAL to plain GCGAttack's gradient -- not just similar -- proving
    the geometry term is a true additive zero, not a small unintended
    perturbation to the objective."""
    torch.manual_seed(42)
    model = _FakeCausalLM(vocab_size=60, dim=16, n_layers=4)
    processor = _FakeProcessor(vocab_size=60)
    extractor = ActivationExtractor(model, target_layers=[0, 1, 2], token_aggregation="last")

    prefix_ids = torch.tensor(processor.tokenizer.encode("context here"), dtype=torch.long)
    target_ids = torch.tensor(processor.tokenizer.encode("target text"), dtype=torch.long)
    suffix_ids = torch.full((6,), 3, dtype=torch.long)
    clean_ref = torch.randn(16)

    obf = ObfuscatedActivationAttack(
        model, processor, extractor, geometry_weight=0.0, target_layer=1,
        num_steps=10, suffix_length=6, topk=15, candidates_per_step=12,
    )
    plain = GCGAttack(model, processor, num_steps=10, suffix_length=6, topk=15, candidates_per_step=12)

    grad_obf = obf._combined_gradient(
        model.get_input_embeddings(), prefix_ids, suffix_ids.clone(), target_ids, clean_ref
    )
    grad_plain = plain._suffix_gradient(
        model.get_input_embeddings(), prefix_ids, suffix_ids.clone(), target_ids
    )

    assert torch.allclose(grad_obf, grad_plain, atol=1e-5), (
        "geometry_weight=0.0 should produce an IDENTICAL gradient to plain GCG, "
        f"max diff = {(grad_obf - grad_plain).abs().max().item()}"
    )


def test_large_geometry_weight_measurably_reduces_distance_to_clean_reference():
    """The actual scientific claim this attack exists to support: does
    optimizing WITH the geometry term genuinely end up closer to the
    clean reference than optimizing WITHOUT it -- not just plumbed
    through, actually numerically true."""
    torch.manual_seed(7)
    model = _FakeCausalLM(vocab_size=60, dim=16, n_layers=4)
    processor = _FakeProcessor(vocab_size=60)

    # A real forward pass on some clean-looking input, to get a genuine
    # (if synthetic) reference activation rather than pure random noise.
    extractor_for_ref = ActivationExtractor(model, target_layers=[0, 1, 2], token_aggregation="last")
    clean_ids = torch.tensor(processor.tokenizer.encode("a perfectly normal clean sentence"), dtype=torch.long)
    with extractor_for_ref:
        with torch.no_grad():
            _ = model(inputs_embeds=model.get_input_embeddings()(clean_ids).unsqueeze(0))
        clean_ref = extractor_for_ref.get_activations()[1].clone()
    extractor_for_ref.clear()

    base_input = {"page_text": "Some tool output content.", "injectable_field": "page_text"}

    extractor_hi = ActivationExtractor(model, target_layers=[0, 1, 2], token_aggregation="last")
    attack_hi = ObfuscatedActivationAttack(
        model, processor, extractor_hi, geometry_weight=10.0, target_layer=1,
        num_steps=15, suffix_length=6, topk=20, candidates_per_step=16,
    )
    result_hi = attack_hi.generate(base_input, target_behavior="do the thing", clean_reference_activation=clean_ref)

    extractor_lo = ActivationExtractor(model, target_layers=[0, 1, 2], token_aggregation="last")
    attack_lo = ObfuscatedActivationAttack(
        model, processor, extractor_lo, geometry_weight=0.0, target_layer=1,
        num_steps=15, suffix_length=6, topk=20, candidates_per_step=16,
    )
    result_lo = attack_lo.generate(base_input, target_behavior="do the thing", clean_reference_activation=clean_ref)

    assert result_hi.metadata["final_geometry_distance"] < result_lo.metadata["final_geometry_distance"], (
        f"High geometry_weight ({result_hi.metadata['final_geometry_distance']}) should end up closer to "
        f"the clean reference than geometry_weight=0 ({result_lo.metadata['final_geometry_distance']})"
    )


class _MixedSafetyTokenizer:
    """Same fake as test_gcg.py's -- roughly half the vocabulary decodes
    to YAML-unsafe characters, reproducing the conditions of the real
    GPU crash (job 23406) this test guards against. See test_gcg.py for
    the full rationale."""

    UNSAFE_CHARS = ["!", ":", '"', "\n", "&", "*", "[", "]", "{", "}", "|", ">", "`", "#", "?"]
    SAFE_CHARS = list(string_lib.ascii_lowercase)[: len(UNSAFE_CHARS)]

    def __init__(self):
        self._id_to_char = {}
        for i, (safe, unsafe) in enumerate(zip(self.SAFE_CHARS, self.UNSAFE_CHARS)):
            self._id_to_char[2 * i] = safe
            self._id_to_char[2 * i + 1] = unsafe
        self.vocab_size = len(self._id_to_char)

    def encode(self, text):
        ids = [abs(hash(w)) % self.vocab_size for w in text.split()]
        return ids or [0]

    def decode(self, ids):
        return "".join(self._id_to_char.get(i, "?") for i in ids)


def test_generated_suffix_never_contains_unsafe_characters():
    """Regression test for the real GPU crash (job 23406): AgentDojo's
    yaml.safe_load() broke on a suffix containing '!' and other
    YAML-structural characters. This class has its OWN independent
    candidate-selection/seed-validation logic (doesn't call
    super().generate()), so it needs its own copy of this test too --
    the fix living in GCGAttack alone wouldn't protect this class."""
    tokenizer = _MixedSafetyTokenizer()

    class _ProcessorWithMixedTokenizer:
        def __init__(self):
            self.tokenizer = tokenizer

    for seed in range(5):
        torch.manual_seed(seed)
        model = _FakeCausalLM(vocab_size=tokenizer.vocab_size, dim=16, n_layers=4)
        extractor = ActivationExtractor(model, target_layers=[0, 1], token_aggregation="last")
        attack = ObfuscatedActivationAttack(
            model, _ProcessorWithMixedTokenizer(), extractor,
            geometry_weight=1.0, target_layer=1,
            num_steps=10, suffix_length=8, topk=6, candidates_per_step=10,
        )
        base_input = {"page_text": "some content here", "injectable_field": "page_text"}
        clean_ref = torch.randn(16)
        result = attack.generate(base_input, target_behavior="do a thing", clean_reference_activation=clean_ref)

        suffix_text = result.adversarial_text.split(" ", len(base_input["page_text"].split()))[-1]
        for char in tokenizer.UNSAFE_CHARS:
            assert char not in suffix_text, (
                f"seed={seed}: unsafe character {char!r} leaked into generated suffix "
                f"{suffix_text!r} -- the safety mask did not hold"
            )


if __name__ == "__main__":
    tests = [
        test_requires_last_token_aggregation,
        test_gradient_flows_through_both_terms,
        test_geometry_weight_zero_matches_plain_gcg_exactly,
        test_large_geometry_weight_measurably_reduces_distance_to_clean_reference,
        test_generated_suffix_never_contains_unsafe_characters,
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
