"""
tests/test_agentdojo_llm.py
------------------------------
Mechanics-only smoke test for LocalVLMPipelineElement: verifies the
5-tuple contract, tool-call parsing, conversation building (including
tool-result folding), and -- critically -- that ActivationExtractor's
hooks actually fire during the model's .generate() call. Uses a fake
generative model + fake chat processor, but the REAL
extraction.hooks.ActivationExtractor (not a stand-in), so the hook-firing
claim is genuinely verified, not just assumed.

Does NOT verify against the real agentdojo package, a real loaded VLM,
or a real chat template's exact behavior -- see the STATUS note in
environments/agentdojo_llm.py for what remains unverified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.hooks import ActivationExtractor
from environments.agentdojo_llm import LocalVLMPipelineElement, _build_conversation, _parse_model_output


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return (self.linear(x),)  # tuple output, like real transformer blocks


class _FakeInner(nn.Module):
    def __init__(self, n_layers, dim):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


class _FakeGenerativeVLM(nn.Module):
    """Fake model with a real .model.layers stack (so ActivationExtractor's
    auto-discovery finds it exactly like a real HF model) and a .generate()
    that runs an actual forward pass through those layers, so hooks fire
    for real rather than being bypassed."""

    def __init__(self, n_layers=4, dim=8, vocab_size=50, response_ids=(1, 2, 3)):
        super().__init__()
        self.model = _FakeInner(n_layers, dim)
        self.embed = nn.Embedding(vocab_size, dim)
        self._response_ids = list(response_ids)

    def generate(self, input_ids=None, max_new_tokens=512, **kwargs):
        x = self.embed(input_ids)
        _ = self.model(x)  # exercises the hooked layers for real
        batch = input_ids.shape[0]
        response = torch.tensor([self._response_ids] * batch, dtype=input_ids.dtype)
        return torch.cat([input_ids, response], dim=1)


class _FakeChatProcessor:
    """Fakes apply_chat_template + tokenizer __call__ + decode. decode()
    returns prompt+completion (mirroring HF's convention of returning the
    full sequence), using a test-controlled completion string, so the
    "strip the echoed prompt" and tag-parsing logic get genuinely exercised."""

    def __init__(self, completion_text: str):
        self._last_prompt = None
        self._completion_text = completion_text

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
        parts = [f"<{turn['role']}>{turn['content'][0]['text']}</{turn['role']}>" for turn in conversation]
        prompt = "\n".join(parts)
        self._last_prompt = prompt
        return prompt

    def __call__(self, text=None, return_tensors="pt"):
        ids = [min(abs(hash(w)) % 48, 47) + 1 for w in text.split()] or [1]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=True):
        return f"{self._last_prompt}{self._completion_text}"


class _FakeParams:
    def model_json_schema(self):
        return {"type": "object", "properties": {"to": {"type": "string"}}}


class _FakeFunction:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.parameters = _FakeParams()


class _FakeRuntime:
    def __init__(self):
        self.functions = {"send_email": _FakeFunction("send_email", "Sends an email")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_query_produces_tool_call_and_fires_real_hooks():
    model = _FakeGenerativeVLM(n_layers=4, dim=8)
    processor = _FakeChatProcessor(completion_text='<function=send_email>{"to": "a@b.com"}</function>')
    extractor = ActivationExtractor(model, target_layers=[0, 2], token_aggregation="last")
    runtime = _FakeRuntime()
    element = LocalVLMPipelineElement(model, processor, extractor)

    messages = [
        {"role": "system", "content": [{"content": "You are an assistant."}]},
        {"role": "user", "content": [{"content": "Please email a@b.com"}]},
    ]

    query, runtime_out, env, new_messages, extra_args = element.query(
        "Please email a@b.com", runtime, env=None, messages=messages, extra_args={}
    )

    # Hooks genuinely fired during generate() -- not assumed, checked.
    # (extractor itself is cleared internally after each turn now; check
    # the snapshot in turn_activations instead.)
    assert len(element.turn_activations) == 1
    acts = element.turn_activations[0]
    assert set(acts.keys()) == {0, 2}
    for t in acts.values():
        assert t.shape == (8,)

    assert new_messages[-1]["role"] == "assistant"
    tool_calls = new_messages[-1]["tool_calls"]
    assert tool_calls is not None
    assert tool_calls[0].function == "send_email"
    assert tool_calls[0].args == {"to": "a@b.com"}


def test_query_plain_text_answer_no_tool_call():
    model = _FakeGenerativeVLM(n_layers=2, dim=8)
    processor = _FakeChatProcessor(completion_text="The weather is sunny today.")
    extractor = ActivationExtractor(model, target_layers=[0], token_aggregation="last")
    runtime = _FakeRuntime()
    element = LocalVLMPipelineElement(model, processor, extractor)

    messages = [{"role": "user", "content": [{"content": "What's the weather?"}]}]
    _, _, _, new_messages, _ = element.query("What's the weather?", runtime, messages=messages)

    assert new_messages[-1]["tool_calls"] is None
    assert "sunny" in new_messages[-1]["content"][0]["content"]


def test_tool_result_folded_into_conversation():
    messages = [
        {"role": "system", "content": [{"content": "sys"}]},
        {"role": "user", "content": [{"content": "do the thing"}]},
        {"role": "assistant", "content": [{"content": "calling tool"}], "tool_calls": [{"function": "x", "args": {}}]},
        {"role": "tool", "content": [{"content": "42"}], "error": None},
    ]
    conversation = _build_conversation(messages)
    # tool message should have become a user turn, not been dropped
    assert conversation[-1]["role"] == "user"
    assert "TOOL RESULT" in conversation[-1]["content"][0]["text"]
    assert "42" in conversation[-1]["content"][0]["text"]


def test_parse_model_output_no_match_returns_plain_text():
    text, calls = _parse_model_output("just a plain answer, no tags here")
    assert calls == []
    assert text == "just a plain answer, no tags here"


def test_multi_turn_activations_captured_separately():
    """Simulates two turns (as ToolsExecutionLoop would produce by calling
    query() twice) and confirms both turns' activations survive distinctly,
    rather than the second overwriting the first."""
    model = _FakeGenerativeVLM(n_layers=2, dim=8)
    processor = _FakeChatProcessor(completion_text="turn output")
    extractor = ActivationExtractor(model, target_layers=[0], token_aggregation="last")
    runtime = _FakeRuntime()
    element = LocalVLMPipelineElement(model, processor, extractor)

    messages = [{"role": "user", "content": [{"content": "start"}]}]
    _, _, _, messages, _ = element.query("start", runtime, messages=messages)
    messages = [*messages, {"role": "tool", "content": [{"content": "result"}], "error": None}]
    _, _, _, messages, _ = element.query("start", runtime, messages=messages)

    assert len(element.turn_activations) == 2, "Expected one activation snapshot per turn"
    # Each turn's tensors should be independent objects, not the same
    # dict reused/overwritten across turns.
    assert element.turn_activations[0] is not element.turn_activations[1]
    assert 0 in element.turn_activations[0] and 0 in element.turn_activations[1]


if __name__ == "__main__":
    tests = [
        test_query_produces_tool_call_and_fires_real_hooks,
        test_query_plain_text_answer_no_tool_call,
        test_tool_result_folded_into_conversation,
        test_parse_model_output_no_match_returns_plain_text,
        test_multi_turn_activations_captured_separately,
    ]
    passed = failed = 0
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
