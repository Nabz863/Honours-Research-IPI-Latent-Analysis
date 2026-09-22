"""
tests/test_vwa_adv_llm.py
---------------------------
Integration-level tests for environments/vwa_adv_llm.py, run against the
REAL ChenWu98/visualwebarena agent.agent / agent.prompts.prompt_constructor
/ llms.lm_config classes (not stand-ins) -- requires visualwebarena's repo
root on sys.path/PYTHONPATH (see conftest-style setup at the bottom of this
file / hpc_vwa_adv_smoketest.sh for the real invocation). Only the model
itself is faked, exactly like test_agentdojo_llm.py fakes the model while
using the real extraction.hooks.ActivationExtractor.

Requires the same placeholder env vars documented in vwa_adv_llm.py's
module docstring (REDDIT/SHOPPING/WIKIPEDIA/HOMEPAGE/CLASSIFIEDS/
CLASSIFIEDS_RESET_TOKEN/OPENAI_API_KEY) -- browser_env and
llms.providers.openai_utils both assert/construct things at import time
regardless of whether those providers are actually used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# --- Make visualwebarena importable, and satisfy its import-time env-var
# --- requirements, before importing anything that touches it. ---
_VWA_ROOT = os.environ.get("VWA_ROOT", "/home/claude/vwa_work/visualwebarena")
sys.path.insert(0, _VWA_ROOT)
for _var in ("REDDIT", "SHOPPING", "WIKIPEDIA", "HOMEPAGE", "CLASSIFIEDS",
             "REDDIT_RESET_URL", "SHOPPING_ADMIN", "GITLAB", "MAP"):
    os.environ.setdefault(_var, "http://localhost:9999")
os.environ.setdefault("CLASSIFIEDS_RESET_TOKEN", "dummy")
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# VWA's own instruction JSONs (e.g. the openai/anthropic few-shot example
# image paths) are relative to the visualwebarena repo root -- real
# eval_pipeline.py scripts are always run with that as cwd, so tests match.
os.chdir(_VWA_ROOT)

from extraction.hooks import ActivationExtractor
from environments.vwa_adv_llm import (
    LocalVLMAgent,
    LocalVLMBackend,
    LocalVLMPromptConstructor,
    HFTokenizerAdapter,
    construct_local_vlm_agent,
    install_local_vlm_provider,
    uninstall_local_vlm_provider,
    register_local_vlm_backend,
    unregister_local_vlm_backend,
    _dispatching_call_llm,
)

import agent.agent as vwa_agent_module
from llms.lm_config import LMConfig

_SOM_INSTRUCTION_PATH = Path(vwa_agent_module.__file__).resolve().parent / "prompts" / "jsons" / "p_som_cot_id_actree_3s.json"


# ---------------------------------------------------------------------------
# Fakes (model side only -- everything else below is real VWA code)
# ---------------------------------------------------------------------------
class _FakeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return (self.linear(x),)


class _FakeInner(nn.Module):
    def __init__(self, n_layers, dim):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


class _FakeGenerativeVLM(nn.Module):
    """Real .model.layers stack (so ActivationExtractor's auto-discovery
    finds it like a real HF model) + a .generate() that runs an actual
    forward pass through those layers, so hooks fire for real."""

    def __init__(self, n_layers=4, dim=8, vocab_size=50, response_ids=(1, 2, 3)):
        super().__init__()
        self.model = _FakeInner(n_layers, dim)
        self.embed = nn.Embedding(vocab_size, dim)
        self._response_ids = list(response_ids)

    def generate(self, input_ids=None, max_new_tokens=512, **kwargs):
        x = self.embed(input_ids)
        _ = self.model(x)
        batch = input_ids.shape[0]
        response = torch.tensor([self._response_ids] * batch, dtype=input_ids.dtype)
        return torch.cat([input_ids, response], dim=1)


class _FakeInnerTokenizer:
    """Backs HFTokenizerAdapter -- deterministic word-count tokenization."""

    def encode(self, text):
        return [min(abs(hash(w)) % 100, 99) + 1 for w in text.split()] or [1]

    def decode(self, ids):
        return " ".join(f"tok{i}" for i in ids)


class _FakeChatProcessor:
    """Fakes apply_chat_template + __call__ + decode, mirroring
    test_agentdojo_llm.py's _FakeChatProcessor. Exposes .tokenizer for
    HFTokenizerAdapter, and records the images it was called with so tests
    can confirm ordering."""

    def __init__(self, completion_text: str, response_ids=(1, 2, 3)):
        self.tokenizer = _FakeInnerTokenizer()
        self._last_prompt = None
        self._completion_text = completion_text
        self._response_ids = list(response_ids)
        self.last_images = None

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
        parts = []
        for turn in conversation:
            text_bits = [b["text"] for b in turn["content"] if b.get("type") == "text"]
            parts.append(f"<{turn['role']}>{' '.join(text_bits)}</{turn['role']}>")
        prompt = "\n".join(parts)
        self._last_prompt = prompt
        return prompt

    def __call__(self, images=None, text=None, return_tensors="pt"):
        self.last_images = images
        ids = [min(abs(hash(w)) % 48, 47) + 1 for w in text.split()] or [1]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=True):
        ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if ids_list == self._response_ids:
            # Caller sliced to JUST the newly generated tokens (the
            # correct, robust way -- see vwa_adv_llm.py's
            # LocalVLMBackend.generate()) -- a real tokenizer decodes
            # only what it's given, so this fake does the same instead
            # of always returning the same hardcoded prompt+completion
            # regardless of input, which was masking a real bug (see
            # agentdojo_llm.py's identical fix and its own discovery).
            return self._completion_text
        return f"{self._last_prompt}{self._completion_text}"


# ---------------------------------------------------------------------------
# get_lm_api_input: local_vlm branch (real MultimodalCoTPromptConstructor)
# ---------------------------------------------------------------------------
def test_get_lm_api_input_local_vlm_branch_zero_shot():
    lm_config = LMConfig(
        provider="local_vlm", model="fake/vlm-7b", mode="chat",
        gen_config={"max_obs_length": 0, "max_retry": 1},
    )
    tokenizer = HFTokenizerAdapter(_FakeChatProcessor("unused"))
    pc = LocalVLMPromptConstructor(str(_SOM_INSTRUCTION_PATH), lm_config=lm_config, tokenizer=tokenizer)

    trajectory = [{
        "observation": {"text": "[1] link 'Home'\n[2] button 'Submit'", "image": np.zeros((32, 32, 3), dtype=np.uint8)},
        "info": {"page": type("P", (), {"url": "http://localhost:9999/f/test"})()},
    }]
    meta_data = {"action_history": ["None"]}
    screenshot = Image.new("RGB", (32, 32))
    extra_image = Image.new("RGB", (16, 16))

    prompt = pc.construct(trajectory, "find the thing", screenshot, [extra_image], meta_data)

    assert isinstance(prompt, dict)
    assert set(prompt.keys()) == {"conversation", "images"}
    conv = prompt["conversation"]
    assert conv[0]["role"] == "system"
    assert conv[1]["role"] == "user"

    # Zero-shot: only the live turn, no per-example system turns injected.
    assert len(conv) == 2

    # One {"type": "image"} placeholder per collected image, in order.
    image_blocks = [b for b in conv[1]["content"] if b["type"] == "image"]
    assert len(image_blocks) == 2
    assert prompt["images"] == [screenshot, extra_image]


def test_get_lm_api_input_falls_back_to_super_for_other_providers():
    """Pointing this at a non-local_vlm provider must reproduce VWA's own
    (unmodified) openai branch -- proves the override doesn't break the
    class for real API models."""
    lm_config = LMConfig(
        provider="openai", model="gpt-4-vision-preview", mode="chat",
        gen_config={"max_obs_length": 0, "max_retry": 1},
    )
    tokenizer = HFTokenizerAdapter(_FakeChatProcessor("unused"))
    pc = LocalVLMPromptConstructor(str(_SOM_INSTRUCTION_PATH), lm_config=lm_config, tokenizer=tokenizer)

    trajectory = [{
        "observation": {"text": "[1] link 'Home'", "image": np.zeros((32, 32, 3), dtype=np.uint8)},
        "info": {"page": type("P", (), {"url": "http://localhost:9999/f/test"})()},
    }]
    meta_data = {"action_history": ["None"]}
    screenshot = Image.new("RGB", (32, 32))

    prompt = pc.construct(trajectory, "find the thing", screenshot, [], meta_data)

    # VWA's own openai branch returns a list of role/content dicts, not our dict shape.
    assert isinstance(prompt, list)
    assert prompt[0]["role"] == "system"
    assert prompt[0]["content"][0]["text"] == pc.instruction["intro"]


# ---------------------------------------------------------------------------
# call_llm dispatch
# ---------------------------------------------------------------------------
def test_dispatching_call_llm_routes_local_vlm_to_backend():
    class _StubBackend:
        def __init__(self):
            self.received = None

        def generate(self, prompt):
            self.received = prompt
            return "stub response"

    stub = _StubBackend()
    register_local_vlm_backend("stub-model", stub)
    try:
        lm_config = LMConfig(provider="local_vlm", model="stub-model", mode="chat", gen_config={})
        result = _dispatching_call_llm(lm_config, {"conversation": [], "images": []})
        assert result == "stub response"
        assert stub.received == {"conversation": [], "images": []}
    finally:
        unregister_local_vlm_backend("stub-model")


def test_dispatching_call_llm_missing_backend_raises():
    lm_config = LMConfig(provider="local_vlm", model="nobody-registered", mode="chat", gen_config={})
    raised = False
    try:
        _dispatching_call_llm(lm_config, {"conversation": [], "images": []})
    except RuntimeError:
        raised = True
    assert raised


def test_install_local_vlm_provider_patches_real_agent_module():
    """The actual integration point: after install_local_vlm_provider(),
    calling agent.agent.call_llm (the name PromptAgent.next_action() calls)
    with a local_vlm config must reach our backend."""
    class _StubBackend:
        def generate(self, prompt):
            return "```stop [ok]```"

    register_local_vlm_backend("patched-model", _StubBackend())
    install_local_vlm_provider()
    try:
        lm_config = LMConfig(provider="local_vlm", model="patched-model", mode="chat", gen_config={})
        result = vwa_agent_module.call_llm(lm_config, {"conversation": [], "images": []})
        assert result == "```stop [ok]```"
    finally:
        unregister_local_vlm_backend("patched-model")
        uninstall_local_vlm_provider()


# ---------------------------------------------------------------------------
# multimodal_inputs override
# ---------------------------------------------------------------------------
def test_multimodal_inputs_forced_true_for_local_vlm():
    lm_config = LMConfig(provider="local_vlm", model="llava-hf/llava-v1.6-mistral-7b-hf", mode="chat", gen_config={"max_retry": 1})
    tokenizer = HFTokenizerAdapter(_FakeChatProcessor("unused"))
    pc = LocalVLMPromptConstructor(str(_SOM_INSTRUCTION_PATH), lm_config=lm_config, tokenizer=tokenizer)
    agent = LocalVLMAgent(action_set_tag="som", lm_config=lm_config, prompt_constructor=pc, captioning_fn=None)
    assert agent.multimodal_inputs is True


def test_multimodal_inputs_preserves_original_behavior_for_other_providers():
    """Base PromptAgent behavior (False unless gemini/gpt-4/claude is in
    the model name) must be untouched for non-local_vlm providers."""
    lm_config = LMConfig(provider="openai", model="gpt-3.5-turbo", mode="chat", gen_config={"max_retry": 1})
    tokenizer = HFTokenizerAdapter(_FakeChatProcessor("unused"))
    pc = LocalVLMPromptConstructor(str(_SOM_INSTRUCTION_PATH), lm_config=lm_config, tokenizer=tokenizer)
    agent = LocalVLMAgent(action_set_tag="som", lm_config=lm_config, prompt_constructor=pc, captioning_fn=None)
    assert agent.multimodal_inputs is False


# ---------------------------------------------------------------------------
# LocalVLMBackend.generate(): real hook-firing
# ---------------------------------------------------------------------------
def test_backend_generate_fires_real_hooks_and_strips_prompt():
    model = _FakeGenerativeVLM(n_layers=4, dim=8)
    processor = _FakeChatProcessor(completion_text='some reasoning ```stop [done]```')
    extractor = ActivationExtractor(model, target_layers=[0, 2], token_aggregation="last")
    backend = LocalVLMBackend(model, processor, extractor, max_new_tokens=16)

    prompt = {
        "conversation": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "do the thing"}, {"type": "image"}]},
        ],
        "images": [Image.new("RGB", (8, 8))],
    }

    completion = backend.generate(prompt)

    assert len(backend.step_activations) == 1
    acts = backend.step_activations[0]
    assert set(acts.keys()) == {0, 2}
    for t in acts.values():
        assert t.shape == (8,)

    assert completion == 'some reasoning ```stop [done]```'
    assert processor.last_images == prompt["images"]


def test_backend_generate_two_steps_do_not_overwrite_each_other():
    model = _FakeGenerativeVLM(n_layers=2, dim=8)
    processor = _FakeChatProcessor(completion_text='```stop [x]```')
    extractor = ActivationExtractor(model, target_layers=[0], token_aggregation="last")
    backend = LocalVLMBackend(model, processor, extractor, max_new_tokens=8)

    prompt = {"conversation": [{"role": "user", "content": [{"type": "text", "text": "go"}, {"type": "image"}]}], "images": [Image.new("RGB", (4, 4))]}
    backend.generate(prompt)
    backend.generate(prompt)

    assert len(backend.step_activations) == 2
    assert backend.step_activations[0] is not backend.step_activations[1]


# ---------------------------------------------------------------------------
# Full chain: construct_local_vlm_agent -> real PromptAgent.next_action()
# ---------------------------------------------------------------------------
def test_construct_local_vlm_agent_end_to_end_next_action():
    """Exercises the entire real chain unmodified: multimodal detection ->
    prompt_constructor.construct() -> local_vlm get_lm_api_input branch ->
    monkeypatched call_llm -> LocalVLMBackend.generate() (real hooks
    firing) -> VWA's own extract_action()/create_id_based_action() parsing
    the returned action. Only the model is fake."""
    model = _FakeGenerativeVLM(n_layers=2, dim=8)
    processor = _FakeChatProcessor(
        completion_text='I will finish now. In summary, the next action I will perform is ```stop [test answer]```'
    )
    extractor = ActivationExtractor(model, target_layers=[0], token_aggregation="last")

    agent, backend = construct_local_vlm_agent(
        model, processor, extractor,
        model_id="fake/e2e-vlm",
        instruction_path=str(_SOM_INSTRUCTION_PATH),
        action_set_tag="som",
        max_new_tokens=16,
        max_retry=1,
    )
    try:
        trajectory = [{
            "observation": {"text": "[1] link 'Home'\n[2] button 'Submit'", "image": np.zeros((32, 32, 3), dtype=np.uint8)},
            "info": {"page": type("P", (), {"url": "http://localhost:9999/f/test"})()},
        }]
        meta_data = {"action_history": ["None"]}

        action = agent.next_action(trajectory, "find the thing", meta_data, images=[])

        assert action["answer"] == "test answer" or "test answer" in str(action)
        assert len(backend.step_activations) == 1
    finally:
        unregister_local_vlm_backend("fake/e2e-vlm")
        uninstall_local_vlm_provider()


if __name__ == "__main__":
    tests = [
        test_get_lm_api_input_local_vlm_branch_zero_shot,
        test_get_lm_api_input_falls_back_to_super_for_other_providers,
        test_dispatching_call_llm_routes_local_vlm_to_backend,
        test_dispatching_call_llm_missing_backend_raises,
        test_install_local_vlm_provider_patches_real_agent_module,
        test_multimodal_inputs_forced_true_for_local_vlm,
        test_multimodal_inputs_preserves_original_behavior_for_other_providers,
        test_backend_generate_fires_real_hooks_and_strips_prompt,
        test_backend_generate_two_steps_do_not_overwrite_each_other,
        test_construct_local_vlm_agent_end_to_end_next_action,
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
