"""
attacks/cross_modal_injection.py
-----------------------------------
CrossInject-style coordinated cross-modal IPI (Table 3.2, primary
cross-modal baseline). Wang et al. 2025, "Manipulating multimodal agents
via cross-modal prompt injection" (ACM MM 2025).

CrossInject has two coordinated halves (chapter2.tex 2.3.4):
  - Visual Latent Alignment (VLA): optimise adversarial features in the
    VISUAL EMBEDDING SPACE, not merely in pixel space against an output
    loss.
  - Textual Guidance Enhancement (TGE): generate steered textual
    commands that reinforce the visual channel.

WHAT CHANGED IN THIS VERSION: an earlier implementation had no VLA at
all. It ran the standard pixel attack with the textual payload spliced
into the teacher-forcing context and called that "coordination". That is
a real mechanism, but it is NOT Visual Latent Alignment -- it never
touched the visual embedding space. VLA is now implemented for real, and
the older behaviour is retained as an ablation (vla_weight=0.0).

HOW VLA IS IMPLEMENTED HERE:
The perturbed image's projected visual features -- the output of the
multimodal projector, i.e. the image tokens as they enter the language
model's embedding space -- are pulled toward the embedding of the
attacker's target text. Both live in the same space, which is precisely
the property this thesis is about (chapter2.tex: "image tokens and text
tokens occupy the same latent space"). The total objective is

    L = L_ce(target | image, prompt) + vla_weight * (1 - cos(v, t))

where v is the mean projected visual feature and t is the mean input
embedding of the target text. L_ce alone is the plain pixel attack; the
second term is VLA.

The projector output is captured with a NON-DETACHING forward hook, so
it stays in the autograd graph and gradients reach the perturbation.
This deliberately does NOT reuse extraction.hooks.ActivationExtractor,
which detaches and moves to CPU by design -- reusing it would silently
zero the VLA gradient. Same reasoning, and the same trap, as documented
in attacks/obfuscated_activation.py.

ARCHITECTURE PROBING: the projector attribute path differs across VLM
families and across transformers versions, so it is probed rather than
hardcoded, in the same spirit as extraction/hooks.py's layer discovery.
If no projector is found the attack FAILS LOUDLY rather than silently
running with vla_weight effectively 0 -- a silent fallback would report
"CrossInject" results that contain no VLA at all.

STILL NOT FAITHFUL, stated plainly: the paper optimises visual features
jointly with a learned textual command generator (TGE). Here the textual
half is TemplateIPIAttack, a fixed template library, not a learned
generator. Describe this as "CrossInject-style" -- VLA is now real, TGE
is not.

P2b (5 Oct), UNDER SELF-CAPTIONING: the visual half is the captioner attack
(attacks/captioner_objective.py: the agent model's own captioning request,
target = the task's target caption) plus VLA toward that caption's embedding;
the textual half is the template-IPI payload, which the runner appends to our
model's caption of the adversarial image. CrossInject minus pixel_perturbation,
paired by task, isolates VLA plus the text half (audit 12.11). Tiny-model test:
tests/test_p2b.py; real weights: the P2b smoke (vla_loss_start/end recorded).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import AttackModality, AttackResult, BaseAttack
from .template_ipi import TemplateIPIAttack

logger = logging.getLogger(__name__)

_CROSS_MODAL = getattr(AttackModality, "CROSS_MODAL", None)
if _CROSS_MODAL is None:
    raise AttributeError(
        "AttackModality has no CROSS_MODAL member. Add `CROSS_MODAL = \"cross_modal\"` "
        "to attacks/base.py's AttackModality enum."
    )

# Probed in order. Covers LLaVA-NeXT, Qwen-VL and InternVL layouts across
# transformers 4.x/5.x, where the projector moved under .model in places.
_PROJECTOR_PATHS = (
    "multi_modal_projector",
    "model.multi_modal_projector",
    "mlp1",
    "model.mlp1",
    "visual.merger",
    "model.visual.merger",
    # Gemma 4 Unified (12B/31B): no separate vision tower; image patches
    # are embedded straight into the text model's space by embed_vision
    # (output 3840 = text hidden size). Read off a meta-device build of
    # the real config (env audit S9, 30 Sep).
    "model.embed_vision",
)


def _resolve_projector(model) -> Tuple[Any, str]:
    """Returns (module, path) for the vision->LM projector, or raises."""
    for path in _PROJECTOR_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module):
            return obj, path
    raise RuntimeError(
        "Could not locate a vision->language projector on this model (tried "
        f"{_PROJECTOR_PATHS}). Visual Latent Alignment cannot run without it. Add the correct "
        "attribute path rather than proceeding -- running with vla_weight silently inert would "
        "produce results labelled CrossInject that contain no Visual Latent Alignment at all."
    )


def _projected(out):
    """The projector's output tensor, whatever container it comes in:
    a tensor (Qwen visual.merger), a tuple, or a ModelOutput -- Gemma 4's
    embed_vision is called with return_dict=True and returns a
    BaseModelOutputWithPooling whose pooler_output is the image in the text
    model's space (review P2b, B2: the hook used to see nothing on Gemma).
    Gemma's padded soft-token slots are included in the mean."""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "last_hidden_state"):
        t = getattr(out, attr, None)
        if torch.is_tensor(t):
            return t
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    return None


class CrossModalInjectionAttack(BaseAttack):
    """Coordinated visual + textual injection with real Visual Latent Alignment."""

    name = "cross_modal_injection"
    modality = _CROSS_MODAL

    def __init__(
        self,
        model,
        processor,
        epsilon: float = 16 / 255,
        alpha: float = 1 / 255,
        iters: int = 500,
        vla_weight: float = 1.0,
        templates: Optional[List[str]] = None,
        template_seed: int = 0,
        check_every: int = 50,
        max_seconds: float = 1200.0,
        opt_max_side: int = 768,
    ):
        """
        vla_weight : weight on the Visual Latent Alignment term. 0.0 ablates
            VLA, reducing the visual half to the plain captioner attack
            (pixel_perturbation), so CrossInject minus pixel isolates VLA plus
            the text half.
        epsilon, alpha, iters, check_every, max_seconds, opt_max_side : as
            pixel_perturbation (shared objective, attacks/captioner_objective.py).
        """
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.vla_weight = vla_weight
        self.check_every = check_every
        self.max_seconds = max_seconds
        self.opt_max_side = opt_max_side
        self._textual = TemplateIPIAttack(templates=templates, seed=template_seed)

    # ------------------------------------------------------------------
    def _target_text_embedding(self, target_text: str, device, dtype) -> torch.Tensor:
        """Mean input embedding of the target text, no BOS (audit 12.4) --
        the point in the shared latent space the visual features are pulled
        toward."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        ids = torch.tensor(tokenizer(target_text, add_special_tokens=False)["input_ids"],
                           device=device, dtype=torch.long)
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(ids)
        return emb.mean(dim=0).to(dtype)

    def generate(
        self,
        base_input: Dict[str, Any],
        target_behavior: str,
        image: Optional[Image.Image] = None,
        clean_text: Optional[str] = None,
        injectable_field: str = "caption",
        template_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> AttackResult:
        """P2b, under self-captioning. Visual half: the captioner attack
        toward the task's target caption, plus VLA (projected visual
        features pulled toward the target caption's embedding). Textual
        half: the template-IPI payload, appended to the description. The
        runner gives the agent <our model's caption of the adversarial
        image> + <payload> (run_vwa_adv_curated_matrix._victim_caption).

        The earlier 'coordinated prompt' (the text payload inside the
        optimisation context) is dropped: the captioner never sees the
        description it is writing, so that context does not exist in the
        episode."""
        from .captioner_objective import optimise_caption, target_caption_of
        image = image or base_input.get("image")
        if image is None:
            raise ValueError("CrossModalInjectionAttack needs a base image.")
        target = target_caption_of(base_input)

        # --- Textual half (TGE stand-in: fixed templates, not learned) ---
        text_input = dict(base_input)
        text_input["injectable_field"] = injectable_field
        if clean_text is not None:
            text_input[injectable_field] = clean_text
        text_result = self._textual.generate(text_input, target_behavior, template_idx=template_idx)

        # --- Visual half: captioner objective + VLA through a non-detaching hook ---
        device = next(self.model.parameters()).device
        projector, projector_path = _resolve_projector(self.model)
        target_embed = self._target_text_embedding(target, device, next(self.model.parameters()).dtype)
        captured: Dict[str, Any] = {}
        vla_hist: List[float] = []

        def _hook(module, inp, out):
            # Only an output WITH a gradient path is kept (Gemma's embed_vision
            # is also called on token ids; review 3, #9).
            t = _projected(out)
            if torch.is_tensor(t) and t.requires_grad:
                captured["v"] = t
                captured["n_grad"] = captured.get("n_grad", 0) + 1
            else:
                captured["n_nograd"] = captured.get("n_nograd", 0) + 1

        def _vla() -> torch.Tensor:
            v = captured.pop("v", None)
            if v is None:
                raise RuntimeError(
                    f"Projector hook at {projector_path!r} produced no output with a gradient path "
                    f"({captured.get('n_nograd', 0)} call(s) without one) -- VLA would be silently inert.")
            v_mean = v.reshape(-1, v.shape[-1]).mean(dim=0)
            loss = 1.0 - F.cosine_similarity(v_mean.unsqueeze(0).float(),
                                             target_embed.unsqueeze(0).float()).squeeze(0)
            vla_hist.append(float(loss.detach()))
            return self.vla_weight * loss

        handle = projector.register_forward_hook(_hook)
        try:
            adv_image, meta = optimise_caption(
                self.model, self.processor, image, target,
                epsilon=self.epsilon, alpha=self.alpha, iters=self.iters,
                check_every=self.check_every, max_seconds=self.max_seconds,
                opt_max_side=self.opt_max_side,
                extra_loss=_vla if self.vla_weight else None)
        finally:
            handle.remove()

        meta.update({
            "vla_weight": self.vla_weight,
            "projector_path": projector_path,
            # If vla_loss_end is not below vla_loss_start, VLA did not optimise.
            "vla_loss_start": vla_hist[0] if vla_hist else None,
            "vla_loss_end": vla_hist[-1] if vla_hist else None,
            "textual": text_result.metadata,
            "tge_faithful": False,  # fixed templates, not a learned generator
        })
        return AttackResult(
            attack_name=self.name,
            modality=self.modality,
            adversarial_image=adv_image,
            adversarial_text=text_result.adversarial_text,
            target_behavior=target_behavior,
            metadata=meta,
        )

    @staticmethod
    def to_vwa_adv_delivery(result: AttackResult, url: str):
        """Adapts an AttackResult into VWA-Adv's (adv_url2caption,
        adv_url2image) pair, keyed on the image URL."""
        cap = {url: result.adversarial_text} if result.adversarial_text else {}
        img = {url: result.adversarial_image} if result.adversarial_image else {}
        return cap, img
