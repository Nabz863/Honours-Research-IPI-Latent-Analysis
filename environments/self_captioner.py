"""
environments/self_captioner.py
------------------------------
Self-captioning for VWA-Adv: the agent model captions the page's images
itself, and those captions appear in its observation as
", description: <caption>" -- the VWA-Adv paper's "self-captioning"
configuration (Wu et al., ICLR 2025). Decision: Nabeel, 30 Sep 2026.

WHY
    VWA-Adv's image attacks were designed for captioner-augmented agents:
    the attack changes the caption. With no captioner (as run until now),
    an image attack reaches the agent only as a thumbnail in the
    screenshot -- 19 of the 40 tasks show the victim image at ~64 px. With
    self-captioning, a pixel attack is a white-box attack on OUR model in
    its captioning role, and the paper's own adversarial images (optimised
    against LLaVA-1.5) become a transfer condition.

WHAT MATCHES THE PAPER
    Prompt text: "Provide a short image description." -- agent-attack's
    LLaVa.get_captioning_prompt_fn(). Here it goes through each model's own
    chat template rather than LLaVA's vicuna_v1 template, since the
    captioner IS the agent model. Greedy decoding.

WHAT IS DELIBERATELY NOT DONE HERE
    No activation capture. Captioning runs inside env.reset()/env.step(),
    outside the backend's `with extractor:` block, so the extractor's hooks
    are not registered and caption passes cannot contaminate the agent's
    per-step activations. check_not_capturing() makes that an assertion
    rather than an assumption.

BLUR
    set_blur(defence) makes every caption read the blurred image (page-level
    blur, audit 13.4). Captions depend on it, so the environment's caption
    cache must be cleared whenever blur changes -- see clear_caption_cache().
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch

logger = logging.getLogger(__name__)

CAPTION_PROMPT = "Provide a short image description."


class SelfCaptioner:
    """Callable with VWA's captioning_fn contract: list[PIL] -> list[str]."""

    def __init__(self, model, processor, max_new_tokens: int = 64,
                 prompt: str = CAPTION_PROMPT, extractor: Any = None):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.extractor = extractor
        self.blur = None              # a GaussianBlurDefense, or None
        self.n_calls = 0
        self.n_images = 0

    def set_blur(self, defence) -> None:
        self.blur = defence

    def check_not_capturing(self) -> None:
        ex = self.extractor
        if ex is None:
            return
        for attr in ("_handles", "handles", "_hooks"):
            h = getattr(ex, attr, None)
            if h:
                raise RuntimeError(
                    "self-captioner called while the ActivationExtractor has live hooks -- "
                    "caption passes would be recorded as agent activations")

    @torch.no_grad()
    def caption(self, image) -> str:
        img = image.convert("RGB")
        if self.blur is not None:
            img = self.blur.blur_pil(img)
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        device = next(self.model.parameters()).device
        inputs = self.processor(images=[img], text=text, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        n_in = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None)
        cap = self.processor.decode(out[0][n_in:], skip_special_tokens=True)
        # One line, always. VWA writes the caption into the image's alt text
        # as "description: <caption>, url: ..." and later swaps in the
        # adversarial caption with re.sub(r"description: (.+?), url", ...),
        # whose `.` does not match a newline: a multi-line clean caption
        # would make the substitution silently fail and the attack never
        # land (processors.py, read 30 Sep).
        return " ".join(cap.split())

    def __call__(self, images: List[Any], prompts: Optional[List[str]] = None, **kwargs) -> List[str]:
        if prompts is not None:
            # VWA also uses captioning_fn for image-question evaluators.
            # None of the 40 tasks use one (env audit S6); refuse rather
            # than silently answering a different question.
            raise NotImplementedError("SelfCaptioner does not answer VQA prompts")
        self.check_not_capturing()
        self.n_calls += 1
        self.n_images += len(images)
        return [self.caption(im) for im in images]


def clear_caption_cache(env) -> int:
    """Empties VWA's per-environment caption cache so the next observation
    re-captions (needed whenever blur changes). Returns how many entries
    were dropped. The cache is text_processor.url2caption, keyed by image
    URL (browser_env/processors.py:83, 602-635; read 30 Sep)."""
    dropped = 0
    handler = getattr(env, "observation_handler", None)
    for proc in (getattr(handler, "text_processor", None), getattr(handler, "image_processor", None)):
        cache = getattr(proc, "url2caption", None)
        if isinstance(cache, dict):
            dropped += len(cache)
            cache.clear()
    return dropped
