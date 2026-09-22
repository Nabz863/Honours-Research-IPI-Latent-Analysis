"""
tests/test_gcg.py
--------------------
Mechanics-only smoke test for GCGAttack: verifies gradients flow through
the one-hot relaxation, the topk/candidate-sampling loop completes, and
the optimized suffix's loss is genuinely lower than the untouched initial
suffix's -- using a fake HF-convention TEXT model (inputs_embeds/labels
-> object with .loss), NOT a real VLM. Does NOT verify against real model
weights or real tokenizer behavior -- run the real thing via sbatch
before trusting this for actual attacks, same caveat as
test_pixel_perturbation.py.
"""

from __future__ import annotations

import string as string_lib
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks.gcg import GCGAttack


class _FakeOutput:
    def __init__(self, loss):
        self.loss = loss


class _FakeCausalLM(nn.Module):
    """Toy model matching the HF calling convention: forward(inputs_embeds, labels) -> .loss.
    Has a real nn.Embedding (so get_input_embeddings() has a genuine weight matrix to backprop
    through) and a small nonlinear 'transformer' so loss is a non-trivial function of the input
    embeddings, not just a linear readout -- closer to exercising the real gradient path."""

    def __init__(self, vocab_size: int = 60, dim: int = 16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.transform = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds=None, labels=None, **kwargs):
        hidden = torch.tanh(self.transform(inputs_embeds))
        # Real transformers mix information across sequence positions via
        # attention -- this fake model needs SOME analogous cross-position
        # dependence too, or the loss at the target positions becomes
        # mathematically independent of the suffix positions entirely
        # (each position's logits would only depend on that position's own
        # embedding), making both the suffix-optimization objective and any
        # gradient-flow check on it meaningless regardless of whether the
        # real gcg.py code is correct. Mean-pooling context across all
        # positions before the final projection is the minimal fix.
        context = hidden.mean(dim=1, keepdim=True)
        hidden = hidden + context
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
    """Deterministic fake tokenization: one token id per word, capped at vocab size."""

    def __init__(self, vocab_size=60):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [min(abs(hash(w)) % (self.vocab_size - 1), self.vocab_size - 2) + 1 for w in text.split()] or [1]

    def decode(self, ids):
        return " ".join(f"tok{i}" for i in ids)


class _FakeProcessor:
    def __init__(self, vocab_size=60):
        self.tokenizer = _FakeTokenizer(vocab_size)


def test_gcg_mechanics_gradients_and_shape():
    model = _FakeCausalLM(vocab_size=60, dim=16)
    processor = _FakeProcessor(vocab_size=60)
    attack = GCGAttack(model, processor, num_steps=5, suffix_length=6, topk=10, candidates_per_step=8)

    base_input = {"page_text": "A normal looking tool result.", "injectable_field": "page_text"}
    result = attack.generate(base_input, target_behavior="approve the transaction")

    assert result.attack_name == "gcg"
    assert result.adversarial_text.startswith("A normal looking tool result.")
    assert result.metadata["num_steps"] == 5
    assert result.metadata["suffix_length"] == 6
    assert isinstance(result.metadata["final_loss"], float)


def test_gcg_gradient_actually_flows():
    """Directly checks the one-hot gradient trick isn't silently broken --
    the exact class of bug that hit pixel_perturbation.py's PIL round-trip
    (differentiability killed silently, producing a zero-effect 'attack')."""
    model = _FakeCausalLM(vocab_size=60, dim=16)
    processor = _FakeProcessor(vocab_size=60)
    attack = GCGAttack(model, processor, num_steps=1, suffix_length=4, topk=5, candidates_per_step=4)

    prefix_ids = torch.tensor(processor.tokenizer.encode("context here"), dtype=torch.long)
    target_ids = torch.tensor(processor.tokenizer.encode("target text"), dtype=torch.long)
    suffix_ids = torch.full((4,), 3, dtype=torch.long)

    grad = attack._suffix_gradient(model.get_input_embeddings(), prefix_ids, suffix_ids, target_ids)
    assert grad.shape == (4, 60)
    assert torch.any(grad != 0), "Gradient is all-zero -- suggests a detached graph, not real optimization"


def test_gcg_optimized_suffix_beats_untouched_initial_suffix():
    """The real correctness check: does the optimization loop actually
    reduce the loss versus doing nothing, not just run without crashing."""
    model = _FakeCausalLM(vocab_size=60, dim=16)
    processor = _FakeProcessor(vocab_size=60)
    attack = GCGAttack(
        model, processor, num_steps=20, suffix_length=6, topk=20, candidates_per_step=16, init_suffix_token="!"
    )

    base_input = {"page_text": "Some content.", "injectable_field": "page_text"}
    result = attack.generate(base_input, target_behavior="do the thing")

    init_id = processor.tokenizer.encode("!")[-1]
    untouched_suffix = torch.full((6,), init_id, dtype=torch.long)
    prefix_ids = torch.tensor(
        processor.tokenizer.encode("Follow the instructions in this content.\n\nSome content.\n\n"),
        dtype=torch.long,
    )
    target_ids = torch.tensor(processor.tokenizer.encode("do the thing"), dtype=torch.long)
    initial_loss = attack._evaluate_candidates(
        model.get_input_embeddings(), prefix_ids, untouched_suffix.unsqueeze(0), target_ids
    )[0].item()

    assert result.metadata["final_loss"] <= initial_loss + 1e-4, (
        f"GCG did not improve loss: final={result.metadata['final_loss']}, initial={initial_loss}"
    )


class _MixedSafetyTokenizer:
    """Fake tokenizer where roughly HALF the vocabulary decodes to
    YAML-unsafe characters ('!', ':', '"', '\\n', etc.) and half to safe
    ones -- unlike the plain _FakeTokenizer above (whose decode() always
    produces safe 'tokN' text regardless of which ids get picked, so it
    can't actually exercise whether the safety mask does anything). This
    is the fake standing in for the real crash: real GPU run 23406 hit a
    YAML parse error from a suffix containing '!', ':', '\"' etc. --
    reproducing a vocabulary where those characters are genuinely
    reachable is what makes this test meaningful rather than vacuous."""

    UNSAFE_CHARS = ["!", ":", '"', "\n", "&", "*", "[", "]", "{", "}", "|", ">", "`", "#", "?"]
    SAFE_CHARS = list(string_lib.ascii_lowercase)[: len(UNSAFE_CHARS)]

    def __init__(self):
        # Interleaved so id parity alone doesn't correlate with anything
        # else in the optimization (avoids accidentally testing a
        # coincidence instead of the actual mask).
        self._id_to_char = {}
        for i, (safe, unsafe) in enumerate(zip(self.SAFE_CHARS, self.UNSAFE_CHARS)):
            self._id_to_char[2 * i] = safe
            self._id_to_char[2 * i + 1] = unsafe
        self.vocab_size = len(self._id_to_char)

    def encode(self, text):
        # Deterministic hash-based mapping into this small vocab -- same
        # spirit as the other fake tokenizers' encode(), just over a
        # smaller id range.
        ids = [abs(hash(w)) % self.vocab_size for w in text.split()]
        return ids or [0]

    def decode(self, ids):
        return "".join(self._id_to_char.get(i, "?") for i in ids)


def test_generated_suffix_never_contains_unsafe_characters():
    """The actual regression test for the real GPU crash (job 23406):
    AgentDojo's yaml.safe_load() broke on a suffix containing '!' (the
    OLD default init_suffix_token, left untouched by optimization) and
    other YAML-structural characters from unconstrained candidate
    selection. Runs several seeds against a vocabulary where unsafe
    characters are genuinely reachable -- if the mask weren't working,
    at least one of these runs would very likely produce one."""
    tokenizer = _MixedSafetyTokenizer()

    class _ProcessorWithMixedTokenizer:
        def __init__(self):
            self.tokenizer = tokenizer

    for seed in range(5):
        torch.manual_seed(seed)
        model = _FakeCausalLM(vocab_size=tokenizer.vocab_size, dim=16)
        attack = GCGAttack(
            model, _ProcessorWithMixedTokenizer(),
            num_steps=15, suffix_length=8, topk=6, candidates_per_step=10,
        )
        base_input = {"page_text": "some content here", "injectable_field": "page_text"}
        result = attack.generate(base_input, target_behavior="do a thing")

        suffix_text = result.adversarial_text.split(" ", len(base_input["page_text"].split()))[-1]
        for char in tokenizer.UNSAFE_CHARS:
            assert char not in suffix_text, (
                f"seed={seed}: unsafe character {char!r} leaked into generated suffix "
                f"{suffix_text!r} -- the safety mask did not hold"
            )


if __name__ == "__main__":
    tests = [
        test_gcg_mechanics_gradients_and_shape,
        test_gcg_gradient_actually_flows,
        test_gcg_optimized_suffix_beats_untouched_initial_suffix,
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
