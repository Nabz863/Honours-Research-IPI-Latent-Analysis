"""
environments/vwa_adv_llm.py
-----------------------------
VisualWebArena-Adversarial (VWA-Adv) integration for THIS PROJECT'S own
locally-loaded VLMs (via extraction/model_loader.py), following the same
approach as environments/agentdojo_llm.py: run generation in-process so
ActivationExtractor can hook a live model object during actual agent
inference, rather than routing through any out-of-process API/server.

Why VWA's own "huggingface" provider (llms/providers/hf_utils.py) doesn't
work for us: despite the name, it isn't local at all -- it's a
text_generation.Client pointed at a separate HuggingFace Text-Generation-
Inference server via model_endpoint. Same fundamental issue as AgentDojo's
LocalLLM: there's no live model object in this process to attach forward
hooks to if generation happens behind an HTTP server in another process.
It's also text-only (call_llm asserts isinstance(prompt, str) on that
branch), so it wouldn't carry the page screenshot either way.

Three integration gaps found by reading the real ChenWu98/visualwebarena
source directly (not assumed by analogy to AgentDojo -- that assumption is
what caused the FunctionCall/dict bug in the AgentDojo integration):

1. `llms.utils.call_llm` dispatches on `lm_config.provider` across a fixed
   openai/anthropic/google/huggingface elif chain and raises
   NotImplementedError for anything else. Handled here by monkeypatching
   the `call_llm` name inside `agent.agent`'s namespace (NOT editing the
   vendored file) so PromptAgent.next_action's unmodified retry/parsing
   loop transparently reaches our backend for provider=="local_vlm" and
   falls through to the original implementation for every other provider.

2. `agent.prompts.prompt_constructor.MultimodalCoTPromptConstructor.
   get_lm_api_input` only has branches for openai/anthropic/google, each
   producing an API-specific payload (base64 data URLs for OpenAI, etc.).
   LocalVLMPromptConstructor below adds a "local_vlm" branch that instead
   returns {"conversation": [...], "images": [...]} -- HF chat-template-
   ready turns with bare {"type": "image"} placeholders, plus the actual
   PIL.Image objects in call order -- since generation happens in-process
   via processor.apply_chat_template()/processor(...), never through an
   API that needs base64 encoding.

3. `agent.agent.PromptAgent.__init__` decides self.multimodal_inputs by
   checking whether "gemini"/"gpt-4"/"claude" appears in lm_config.model.
   None of this project's model IDs (llava-hf/..., Qwen/..., OpenGVLab/...)
   match, so the base class would silently treat every one of our VLMs as
   text-only and never construct/pass the page screenshot. LocalVLMAgent
   below forces multimodal_inputs=True whenever provider=="local_vlm".

Deliberate scoping choice: the few-shot examples in VWA's own instruction
JSONs (e.g. p_som_cot_id_actree_3s.json) each carry a reference screenshot.
For a 7-8B open VLM, encoding 3 example screenshots plus the live turn's
screenshot in one call roughly quadruples vision-token cost, for a benefit
(GPT-4V-style few-shot transfer) that isn't established for models this
size. LocalVLMPromptConstructor's "local_vlm" branch therefore builds a
ZERO-SHOT prompt by default (intro + current turn only; example text/images
dropped). If this measurably underperforms in the smoketest, revisit --
e.g. text-only descriptions of the examples instead of re-encoding their
images -- rather than silently assuming zero-shot is fine.

Requires visualwebarena's own repo root on sys.path/PYTHONPATH (this module
does bare `from agent...` / `from llms...` imports exactly like
ChenWu98/visualwebarena's own scripts/eval_pipeline.py does -- it is not a
pip-installed package). Importing browser_env also requires REDDIT /
SHOPPING / WIKIPEDIA / HOMEPAGE / CLASSIFIEDS / CLASSIFIEDS_RESET_TOKEN env
vars to be non-empty (browser_env/env_config.py asserts ALL of them are
set, not just the ones actually used under the Reddit-first scoping
decision -- placeholder values for the unused sites are fine) and
OPENAI_API_KEY (llms/providers/openai_utils.py constructs an OpenAI client
at import time regardless of which provider you actually use -- a
placeholder string is fine, no request is ever sent through it here).

STATUS: get_lm_api_input's local_vlm branch, the call_llm monkeypatch
dispatch, and the multimodal_inputs override are verified against the REAL
agent.agent / agent.prompts.prompt_constructor / llms.lm_config / llms.utils
classes from ChenWu98/visualwebarena (not stand-ins) -- see
tests/test_vwa_adv_llm.py. LocalVLMBackend.generate()'s hook-firing is
verified against the real extraction.hooks.ActivationExtractor with a fake
generative model, same pattern as agentdojo_llm.py. NOT yet verified:
against a real loaded VLM, a real running Postmill/Reddit instance, or
genuine end-to-end task execution through browser_env.ScriptBrowserEnv --
that needs postmill.sif confirmed working on the cluster first.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re

import torch
from PIL import Image

try:
    import agent.agent as _vwa_agent_module
    from agent.agent import PromptAgent
    from agent.prompts.prompt_constructor import MultimodalCoTPromptConstructor
    from llms.lm_config import LMConfig
    from llms.utils import call_llm as _ORIGINAL_CALL_LLM
except ImportError:  # keeps this module importable/testable without visualwebarena on sys.path
    _vwa_agent_module = None
    PromptAgent = object  # type: ignore
    MultimodalCoTPromptConstructor = object  # type: ignore
    LMConfig = None  # type: ignore
    _ORIGINAL_CALL_LLM = None


# ---------------------------------------------------------------------------
# Backend registry: call_llm(lm_config, prompt) only receives lm_config +
# prompt, not our backend object, so registered backends are looked up by
# lm_config.model at dispatch time.
# ---------------------------------------------------------------------------
_ACTIVE_BACKENDS: Dict[str, "LocalVLMBackend"] = {}


def register_local_vlm_backend(model_id: str, backend: "LocalVLMBackend") -> None:
    """Make `backend` reachable from the monkeypatched call_llm whenever lm_config.model == model_id."""
    _ACTIVE_BACKENDS[model_id] = backend


def unregister_local_vlm_backend(model_id: str) -> None:
    _ACTIVE_BACKENDS.pop(model_id, None)


def _dispatching_call_llm(lm_config, prompt, num_outputs: int = 1) -> str:
    if lm_config.provider == "local_vlm":
        backend = _ACTIVE_BACKENDS.get(lm_config.model)
        if backend is None:
            raise RuntimeError(
                f"No LocalVLMBackend registered for model {lm_config.model!r}. "
                "Call register_local_vlm_backend(model_id, backend) before running the agent."
            )
        return backend.generate(prompt)
    return _ORIGINAL_CALL_LLM(lm_config, prompt, num_outputs)


_PATCHED = False


def install_local_vlm_provider() -> None:
    """
    Monkeypatch the `call_llm` name inside agent.agent's namespace (NOT the
    vendored llms/utils.py file) so PromptAgent.next_action's inherited,
    UNMODIFIED retry/parsing loop transparently reaches LocalVLMBackend for
    provider=="local_vlm" and behaves exactly as before for every other
    provider. Idempotent -- safe to call more than once.
    """
    global _PATCHED
    if _PATCHED:
        return
    if _vwa_agent_module is None:
        raise ImportError("visualwebarena's agent.agent module is not importable -- check sys.path/PYTHONPATH.")
    _vwa_agent_module.call_llm = _dispatching_call_llm
    _PATCHED = True


def uninstall_local_vlm_provider() -> None:
    """Restore the original call_llm. Mainly for test teardown."""
    global _PATCHED
    if _vwa_agent_module is not None and _PATCHED:
        _vwa_agent_module.call_llm = _ORIGINAL_CALL_LLM
    _PATCHED = False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
class LocalVLMPromptConstructor(MultimodalCoTPromptConstructor):
    """
    Adds a "local_vlm" branch to get_lm_api_input(), alongside VWA's own
    openai/anthropic/google branches (left untouched via super() -- this
    class still works normally if pointed at a real API model instead).
    See module docstring for the zero-shot scoping rationale and the
    {"conversation", "images"} return shape.
    """

    def get_lm_api_input(
        self,
        intro: str,
        examples: Any,
        current: str,
        page_screenshot_img: "Image.Image",
        images: List["Image.Image"],
    ) -> Any:
        if self.lm_config.provider != "local_vlm":
            return super().get_lm_api_input(intro, examples, current, page_screenshot_img, images)

        # PromptAgent.next_action's own signature defaults images to None
        # (not []); VWA's own openai/anthropic branches would crash on that
        # too (bare `for ... in images`), so this isn't purely defensive --
        # it's a real gotcha hit during testing, not a hypothetical one.
        images = images or []

        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": intro}]},
        ]
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": current},
            {"type": "text", "text": "IMAGES: (1) current page screenshot"},
            {"type": "image"},
        ]
        collected_images = [page_screenshot_img]
        for image_i, image in enumerate(images):
            content.append({"type": "text", "text": f"({image_i + 2}) input image {image_i + 1}"})
            content.append({"type": "image"})
            collected_images.append(image)
        conversation.append({"role": "user", "content": content})

        return {"conversation": conversation, "images": collected_images}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class LocalVLMAgent(PromptAgent):
    """
    Same as VWA's PromptAgent, except self.multimodal_inputs is forced True
    for provider=="local_vlm" (see module docstring, gap 3). next_action()
    itself is NOT overridden -- the actual model call is redirected via
    install_local_vlm_provider()'s call_llm monkeypatch, so this class's
    only job is fixing the multimodal-detection gap; every other API
    provider still behaves exactly like the base PromptAgent.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.lm_config.provider == "local_vlm":
            self.multimodal_inputs = True


# ---------------------------------------------------------------------------
# Tokenizer adapter -- PromptConstructor.construct() calls
# self.tokenizer.encode()/.decode() purely to cap the accessibility-tree
# observation to max_obs_length tokens. VWA's own Tokenizer class
# (llms/tokenizers.py) raises NotImplementedError for any provider outside
# openai/huggingface/anthropic/google, so "local_vlm" needs its own here.
# ---------------------------------------------------------------------------
class HFTokenizerAdapter:
    """Duck-types llms.tokenizers.Tokenizer's .encode()/.decode() using the
    already-loaded model's own processor.tokenizer, so observation-length
    truncation uses the real tokenizer for the model actually being run
    instead of an unrelated approximation (e.g. tiktoken)."""

    def __init__(self, processor: Any):
        self.tokenizer = getattr(processor, "tokenizer", processor)

    # add_special_tokens=False: VWA truncates the observation with
    # decode(encode(obs)[:max_obs_length]). A tokenizer that adds BOS on
    # encode (LLaVA-NeXT's '<s>') would otherwise put a literal BOS string
    # in the middle of the prompt (audit 11.10; Qwen3-VL, InternVL3 and
    # Gemma 4 add none, so this changes nothing for them).
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)


# ---------------------------------------------------------------------------
# Generation backend
# ---------------------------------------------------------------------------
class LocalVLMBackend:
    """
    Does the actual in-process generation for one VWA agent step, wrapped
    in ActivationExtractor's hook context -- same lifecycle as
    LocalVLMPipelineElement in environments/agentdojo_llm.py: enter the
    extractor, generate, snapshot this step's activations into
    self.step_activations (a list, index = step number, so no step
    silently overwrites the last), clear, exit. Caller reads
    backend.step_activations after the episode (e.g. after the browser_env
    rollout loop ends) to decide which step(s) matter for analysis.
    """

    def __init__(self, model: Any, processor: Any, extractor: Any, max_new_tokens: int = 512,
                 defense: Any = None, ecso: Any = None, task_intent: Optional[str] = None,
                 payload_markers: Optional[List[str]] = None, icon: Any = None,
                 capture_dir: Optional[str] = None):
        self.model = model
        self.processor = processor
        self.extractor = extractor
        self.max_new_tokens = max_new_tokens
        self.step_activations: List[Dict[int, torch.Tensor]] = []
        # ICON (P2a): probe attention over a short generation, then generate
        # under the Mitigating Rectifier when flagged -- the same flow as the
        # AgentDojo element. One info dict per agent step.
        self.icon = icon
        self.icon_info: List[Dict[str, Any]] = []
        # Per-step text (P2a). Each agent decision records the observation as
        # the model saw it (after any defence), the undefended one, every
        # generation the decision took (ECSO takes up to four) and the
        # response. Message de-duplication as in agentdojo_llm.
        from environments.agentdojo_llm import TranscriptRecorder
        self.transcript = TranscriptRecorder()
        self._calls: List[Dict[str, Any]] = []
        # Capture mode (off by default): the images EXACTLY as given to the
        # model at every generation (the screenshot after any image defence),
        # written as PNGs, for figures and for replaying a decision.
        self.capture_dir = Path(capture_dir) if capture_dir else None
        # DEFENSE SCOPE, and how it differs from AgentDojo's.
        # In AgentDojo the untrusted channel is cleanly delimited: only
        # role="tool" messages carry retrieved content, so the defense
        # wraps exactly those. Here the observation is a single blob --
        # accessibility tree, SoM annotations and the injected caption
        # all concatenated, with no boundary between them. So the whole
        # observation is treated as untrusted and defended in full.
        # That is defensible (it IS all retrieved from the web) but it
        # is a STRONGER intervention than AgentDojo's, so ASR/UA under
        # defense are not directly comparable across the two
        # environments. Say so when reporting.
        self.defense = defense
        self.defense_addenda: List[str] = []
        # Set only when a defence actually transformed an image, so a
        # run labelled with a visual defence that silently did nothing
        # is distinguishable from one that worked.
        self.defense_image_meta: Dict[str, Any] = {}
        # ECSO restructures generation rather than transforming
        # content, so like ICON it rides its own parameter rather
        # than the defense= path, where apply() is a no-op.
        self.ecso = ecso
        self.ecso_info: List[Dict[str, Any]] = []
        self._defense_cache: Dict[str, Any] = {}
        # The user's own task, for ECSO's {query} slot (audit 13.2): the
        # last user message is the whole observation, injection included.
        self.task_intent = task_intent
        # Reachability, per agent step (audit 11.11): does the injected
        # caption text appear in the prompt the model is given? Checked on
        # the prompt BEFORE any defence transforms it.
        # Whitespace-normalised on both sides (review 3, #14).
        self.payload_markers = [" ".join(m.split()) for m in (payload_markers or []) if m and m.strip()]
        self.payload_in_prompt: List[bool] = []
        # Greedy decoding makes a parse-failure retry identical to the call
        # it retries; VWA's PromptAgent re-calls with the SAME prompt
        # object. Returning the previous response keeps step_activations one
        # entry per decision and saves the compute (audit 11.9).
        self._last_prompt: Any = None
        self._last_response: Optional[str] = None
        self.retries_deduplicated = 0
        self.defense_unmatched_steps = 0
        self.prev_action_defended = 0
        self.double_bos: Optional[bool] = None

    def generate(self, prompt: Dict[str, Any]) -> str:
        """Apply defences, then generate once.

        Split from _raw_generate so ECSO can reuse the generation path
        for each of its four steps without re-applying defences or
        re-deriving the decode logic.
        """
        if prompt is self._last_prompt and self._last_response is not None:
            self.retries_deduplicated += 1
            return self._last_response

        conversation = prompt["conversation"]
        images = prompt["images"]
        raw_conversation = conversation
        self._calls = []

        if self.payload_markers:
            raw_text = " ".join(
                b.get("text", "") for m in conversation for b in (m.get("content") or [])
                if isinstance(b, dict) and b.get("type") == "text")
            raw_text = " ".join(raw_text.split())
            self.payload_in_prompt.append(any(mk in raw_text for mk in self.payload_markers))

        if self.defense is not None:
            if not getattr(self.defense, "page_level", False):
                conversation = self._defend(conversation)
            # Visual defences transform the IMAGE, not the text. The
            # default apply_image() added by defenses.visual is the
            # identity, so text defences pass through unchanged and this
            # needs no branching on defence type. A PAGE-LEVEL visual
            # defence (blur on page images, audit 13.4 / D3) is applied in
            # the browser and the captioner instead, never to the whole
            # screenshot here.
            ai = getattr(self.defense, "apply_image", None)
            if getattr(self.defense, "page_level", False):
                ai = None
            if ai is not None and images:
                defended = []
                for im in images:
                    r = ai(im)
                    defended.append(getattr(r, "transformed_image", im))
                    meta = getattr(r, "metadata", None) or {}
                    if meta.get("applied"):
                        self.defense_image_meta = meta
                images = defended

        if getattr(self, "ecso", None) is not None:
            out = self._ecso_generate(conversation, images)
        else:
            out = self._raw_generate(conversation, images)
        self.transcript.add(
            step=len(self.transcript.steps),
            raw_input=self.transcript.ids(raw_conversation),
            calls=self._calls,
            response=out,
            ecso=(self.ecso_info[-1] if getattr(self, "ecso", None) is not None and self.ecso_info else None),
            icon=(self.icon_info[-1] if self.icon is not None and self.icon_info else None),
        )
        self._last_prompt, self._last_response = prompt, out
        return out

    def _raw_generate(self, conversation, images, max_new_tokens=None) -> str:
        """One generation with activation capture. max_new_tokens is
        overridable because ECSO's detect and caption steps are far
        shorter than a full agent response.

        Appends ONE entry to step_activations per call. ECSO calls this
        up to four times per agent step and keeps only the last, so that
        step_activations stays one-entry-per-step and remains comparable
        with every other condition -- see _ecso_generate.
        """
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        device = next(self.model.parameters()).device
        inputs = self.processor(images=images, text=text, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        if self.double_bos is None:
            from environments.agentdojo_llm import _has_double_bos
            self.double_bos = _has_double_bos(self.processor, inputs["input_ids"])

        image_files = self._capture_images(images)

        # ICON: detection pass OUTSIDE the extractor (capture_step='first'
        # would otherwise keep the detection pass's activations), then the
        # agent's generation, rectified when flagged.
        rectify = False
        if self.icon is not None:
            from defenses.icon_runtime import icon_detect
            info = icon_detect(self.model, inputs, self.icon, detect_tokens=32)
            rectify = bool(info.get("icon_is_attacked"))
            info["icon_rectified"] = rectify
            self.icon_info.append(info)
        if rectify:
            from defenses.icon_runtime import _rectifying
            ctx = _rectifying(self.icon.rectifier, self.model)
        else:
            ctx = contextlib.nullcontext()

        with self.extractor:
            with torch.no_grad(), ctx:
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
            # Snapshot THIS step's activations before the next generate()
            # call would overwrite them, then clear for the next step.
            self.step_activations.append(self.extractor.get_activations())
            torch.cuda.empty_cache()
            self.extractor.clear()

        # Decode ONLY the newly generated tokens (sliced by the known
        # input length), NOT the whole sequence followed by
        # string-matching the prompt back out. This is the exact same
        # bug, found and fixed in environments/agentdojo_llm.py's
        # query() after it caused a real, confirmed problem there: the
        # skip_special_tokens decode essentially never matches the
        # literal apply_chat_template(tokenize=False) string for a
        # ChatML-style template, so completion.startswith(text) silently
        # fails and the ENTIRE prompt stays attached to every
        # completion. In AgentDojo this caused the tag parser to grab
        # the system prompt's own illustrative example of the tag format
        # instead of the model's real tool call -- discarding a
        # genuinely correct call as if the model had done nothing. This
        # backend feeds into the real, unmodified visualwebarena
        # PromptAgent.next_action() parsing -- NOT verified here whether
        # that specific parser is equally vulnerable to a prompt-
        # contaminated completion, but there's no reason to find out the
        # hard way when the fix is this direct and already proven.
        completion = self.processor.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
        self._calls.append({
            "input": self.transcript.ids(conversation),
            # identifies the images given to the model even without capture
            # mode, so a capture re-run can be checked against the original
            "image_sha256": [hashlib.sha256(im.tobytes()).hexdigest()[:16] for im in (images or [])],
            "prompt_sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
            "n_input_tokens": int(input_length), "max_new_tokens": int(max_new_tokens),
            "n_images": len(images or []), "image_files": image_files,
            "completion": completion,
        })
        return completion

    def _capture_images(self, images) -> Optional[List[str]]:
        """Capture mode only: write this generation's images as given to the
        model. Names are step/call indices, so a retried episode overwrites
        its own files."""
        if self.capture_dir is None or not images:
            return None
        from experiments.episode_utils import atomic_save_image
        names = []
        for j, im in enumerate(images):
            name = f"step{len(self.transcript.steps):02d}_call{len(self._calls)}_img{j}.png"
            atomic_save_image(im, self.capture_dir / name)
            names.append(name)
        return names

    def _ecso_generate(self, conversation, images) -> str:
        """ECSO replaces generation rather than transforming content, so
        it cannot ride the self.defense path. It calls back into
        _raw_generate for each of its four steps; activations are
        captured on every one of them, and then reduced to ONE entry per
        step: the activation of the generation whose output is returned
        (see below)."""
        mark = len(self.step_activations)

        def gen(conv, imgs, n):
            return self._raw_generate(conv, imgs, max_new_tokens=n)

        r = self.ecso.run(gen, conversation, images, query=self.task_intent)
        # Keep the activations of the generation whose output is RETURNED
        # as the agent's decision -- one entry per step, like every other
        # condition. Unflagged: that is the initial answer (entry `mark`);
        # keeping the last entry, as before, kept the DETECTOR prompt's
        # activation (audit 13.1). Flagged: the image-free regeneration
        # (the last entry).
        entries = self.step_activations[mark:]
        if entries:
            keep = entries[-1] if r.flagged else entries[0]
            self.step_activations[mark:] = [keep]
        meta = dict(r.metadata)
        meta["activation_kept"] = "regeneration" if r.flagged else "initial_answer"
        self.ecso_info.append(meta)
        return r.response

    def _defend_text(self, raw: str) -> Tuple[str, List[str], bool]:
        """Defends ONLY the observation inside VWA's template block
        ('OBSERVATION: ... \nURL: ...\nOBJECTIVE: ...'), leaving the URL
        and the user's objective untouched (audit 11.3). In PREVIOUS
        ACTION, only the page-derived element text is defended (review 3,
        #5); the action verb and element id stay readable.
        Returns (text, addenda, matched_template)."""
        # The observation ends at the LAST "\nURL: " -- the tail (URL,
        # OBJECTIVE, PREVIOUS ACTION) is written by the harness, whereas
        # page text (including an attacker's caption) can itself contain
        # "\nURL:" and would otherwise end the defended region early
        # (review C7).
        head = "OBSERVATION:"
        if not raw.startswith(head):
            return raw, [], False
        cut = raw.rfind("\nURL:")
        if cut == -1:
            return raw, [], False
        start = len(head)
        while start < len(raw) and raw[start] in " \t":
            start += 1
        if start < len(raw) and raw[start] == "\n":
            start += 1
        obs = raw[start:cut]
        r = self._defend_cached(obs)
        tail, tail_add = self._defend_prev_action(raw[cut:])
        adds = [a for a in (r.system_prompt_addendum, tail_add) if a]
        return raw[:start] + r.transformed_content + tail, adds, True

    def _defend_cached(self, text: str):
        if text not in self._defense_cache:
            self._defense_cache[text] = self.defense.apply(text)
        return self._defense_cache[text]

    # VWA describes a click/hover/type on an element as
    #   "<action> [<id>] where [<id>] is <element text>"
    # and <element text> is PAGE-derived: after the agent clicks the victim
    # image it carries the (possibly adversarial) caption into PREVIOUS
    # ACTION, outside the OBSERVATION block (review 3, #5).
    _PREV_ACTION_RE = re.compile(r"^(\w+ \[[^\]]*\] where \[[^\]]*\] is )(.+)$", re.DOTALL)

    def _defend_prev_action(self, tail: str):
        """Defends the page-derived element text inside PREVIOUS ACTION,
        leaving the agent's own action verb and element id readable.
        Returns (tail, addendum or None) -- the delimiting variant's
        addendum names its boundary, so this block needs its own."""
        key = "\nPREVIOUS ACTION:"
        i = tail.rfind(key)
        if i == -1:
            return tail, None
        j = i + len(key)
        while j < len(tail) and tail[j] == " ":
            j += 1
        value = tail[j:]
        m = self._PREV_ACTION_RE.match(value)
        if not m:
            return tail, None
        self.prev_action_defended += 1
        r = self._defend_cached(m.group(2))
        return tail[:j] + m.group(1) + r.transformed_content, (r.system_prompt_addendum or None)

    def _defend(self, conversation):
        """Defends the untrusted observation and tells the model how
        (audits 11.2, 11.3). Only the OBSERVATION section of the user turn
        is transformed; the objective is the user's own and is left as is.
        The defence's system_prompt_addendum is appended to the system
        message -- previously it was collected and never delivered, so on
        this benchmark content was marked but the model was never told
        what the marking meant.

        Previous docstring, kept for the caching rationale:
        Transforms every user-role text block through the defense.
        Cached by raw text for the same reason agentdojo_llm.py caches:
        the conversation is rebuilt every step, and a defense with any
        per-call randomness (spotlighting's delimiting variant generates
        a fresh boundary token each call) would otherwise re-mark the
        same historical observation differently on every step, so the
        model would see its own history appear to change."""
        out = []
        addenda: List[str] = []
        matched_any = False
        for msg in conversation:
            if msg.get("role") != "user":
                out.append(msg)
                continue
            blocks = []
            for b in msg.get("content", []):
                if isinstance(b, dict) and b.get("type") == "text":
                    text, adds, matched = self._defend_text(b.get("text", ""))
                    matched_any = matched_any or matched
                    for add in adds:
                        if add not in addenda:
                            addenda.append(add)
                    blocks.append({"type": "text", "text": text})
                else:
                    blocks.append(b)
            out.append({**msg, "content": blocks})
        for a in addenda:
            if a not in self.defense_addenda:
                self.defense_addenda.append(a)
        if not matched_any:
            # A text defence that found no OBSERVATION block defended
            # nothing. Counted and recorded on the row, never silent.
            self.defense_unmatched_steps += 1
        if not addenda:
            return out
        prov = "\n\n## Content Provenance\n\n" + "\n\n".join(addenda)
        final = []
        patched = False
        for msg in out:
            if msg.get("role") == "system" and not patched:
                blocks = list(msg.get("content") or [])
                for i, b in enumerate(blocks):
                    if isinstance(b, dict) and b.get("type") == "text":
                        blocks[i] = {**b, "text": b.get("text", "") + prov}
                        patched = True
                        break
                final.append({**msg, "content": blocks})
            else:
                final.append(msg)
        if not patched:
            final.insert(0, {"role": "system", "content": [{"type": "text", "text": prov.strip()}]})
        return final


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------
def construct_local_vlm_agent(
    model: Any,
    processor: Any,
    extractor: Any,
    model_id: str,
    instruction_path: str,
    action_set_tag: str = "som",
    max_new_tokens: int = 512,
    max_obs_length: int = 3840,
    max_retry: int = 3,
    temperature: float = 1.0,
    top_p: float = 0.9,
    defense: Any = None,
    ecso: Any = None,
    task_intent: Optional[str] = None,
    payload_markers: Optional[List[str]] = None,
    icon: Any = None,
    capture_dir: Optional[str] = None,
):
    """
    Builds and registers everything needed for one (model, task) run:
    LMConfig, tokenizer adapter, LocalVLMPromptConstructor, LocalVLMBackend,
    and the LocalVLMAgent itself -- then installs the call_llm monkeypatch
    if not already installed.

    instruction_path : reuse one of VWA's own instruction JSONs, e.g.
        agent/prompts/jsons/p_som_cot_id_actree_3s.json -- only
        get_lm_api_input's FORMATTING differs per provider; the task intro/
        template/answer_phrase/action_splitter content is shared, so there
        is no reason to fork it.

    Returns (agent, backend). Keep `backend` around after the episode to
    read backend.step_activations -- `agent` alone doesn't expose them.
    """
    if LMConfig is None:
        raise ImportError("visualwebarena is not importable -- check sys.path/PYTHONPATH.")

    llm_config = LMConfig(
        provider="local_vlm",
        model=model_id,
        mode="chat",
        gen_config={
            "temperature": temperature,
            "top_p": top_p,
            "context_length": 0,
            "max_tokens": max_new_tokens,
            "stop_token": None,
            "max_obs_length": max_obs_length,
            "max_retry": max_retry,
        },
    )

    tokenizer = HFTokenizerAdapter(processor)
    prompt_constructor = LocalVLMPromptConstructor(instruction_path, lm_config=llm_config, tokenizer=tokenizer)
    backend = LocalVLMBackend(model, processor, extractor,
                              max_new_tokens=max_new_tokens, defense=defense,
                              ecso=ecso, task_intent=task_intent,
                              payload_markers=payload_markers, icon=icon,
                              capture_dir=capture_dir)

    register_local_vlm_backend(model_id, backend)
    install_local_vlm_provider()

    agent = LocalVLMAgent(
        action_set_tag=action_set_tag,
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
        captioning_fn=None,
    )
    return agent, backend
