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

STATUS: mechanics verified offline against a synthetic model; VLA needs a
real GPU run to confirm the projector hook fires and the alignment loss
decreases. Check vla_loss_start vs vla_loss_end in the returned metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import AttackModality, AttackResult, BaseAttack
from .pixel_perturbation import PixelPerturbationAttack, _AnyResGeometry
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
        iters: int = 200,
        vla_weight: float = 1.0,
        templates: Optional[List[str]] = None,
        template_seed: int = 0,
    ):
        """
        vla_weight : weight on the Visual Latent Alignment term. 0.0
            ablates VLA entirely, reducing this to the plain pixel attack
            with a coordinated conditioning prompt -- which is exactly
            what the previous version of this file did, so that ablation
            reproduces it and isolates VLA's contribution.
        epsilon, alpha, iters : passed to the internal pixel attack; same
            cost profile as pixel_perturbation.py.
        """
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.vla_weight = vla_weight
        self._visual = PixelPerturbationAttack(model, processor, epsilon=epsilon,
                                               alpha=alpha, iters=iters)
        self._textual = TemplateIPIAttack(templates=templates, seed=template_seed)

    # ------------------------------------------------------------------
    def _target_text_embedding(self, target_text: str, device, dtype) -> torch.Tensor:
        """Mean input embedding of the attacker's target text -- the point
        in the shared latent space the visual features are pulled toward."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        ids = torch.tensor(tokenizer.encode(target_text), device=device, dtype=torch.long)
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
        image = image or base_input.get("image")
        if image is None:
            raise ValueError("CrossModalInjectionAttack needs a base image.")

        # --- Textual half (TGE stand-in: fixed templates, not learned) ---
        text_input = dict(base_input)
        text_input["injectable_field"] = injectable_field
        if clean_text is not None:
            text_input[injectable_field] = clean_text
        text_result = self._textual.generate(text_input, target_behavior, template_idx=template_idx)
        adversarial_text = text_result.adversarial_text

        # The visual half is optimised in a context where the textual
        # payload is ALREADY present, so the two channels reinforce rather
        # than compete. Retained from the previous version; orthogonal to VLA.
        visual_prompt = base_input.get("prompt", "Describe this image.")
        coordinated_prompt = f"{visual_prompt}\n\n{adversarial_text}"

        image_token = getattr(self.processor, "image_token", None)
        prompt = coordinated_prompt
        if image_token and image_token not in prompt:
            prompt = f"{image_token}\n{prompt}"

        device = next(self.model.parameters()).device
        img = image.convert("RGB")
        clean = torch.from_numpy(np.array(img).astype("float32") / 255.0)
        clean = clean.permute(2, 0, 1).unsqueeze(0).to(device)
        delta = torch.zeros_like(clean, requires_grad=True)

        from .geometry_adapters import get_adapter
        adapter = get_adapter(self.processor)
        plan = adapter.plan(clean.shape[-2], clean.shape[-1])

        full_text = f"{prompt} {target_behavior}"
        prompt_only = self.processor(images=img, text=prompt, return_tensors="pt")
        prompt_len = prompt_only["input_ids"].shape[1]
        full_inputs = self.processor(images=img, text=full_text, return_tensors="pt")
        full_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in full_inputs.items()}
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full_inputs["use_cache"] = False

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.train()  # engages gradient checkpointing; see pixel_perturbation.py
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d,
                              torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # --- VLA setup: non-detaching hook on the projector ---
        projector, projector_path = _resolve_projector(self.model)
        target_embed = self._target_text_embedding(
            target_behavior, device, next(self.model.parameters()).dtype
        )
        captured: Dict[str, Any] = {}

        def _hook(module, inp, out):
            # NOT detached -- the VLA gradient must reach delta through this.
            # Only an output WITH a gradient path is kept: some projectors
            # (Gemma's embed_vision) are also called on vision-token ids,
            # whose output has no path to delta; keeping that call would
            # make the alignment term silently inert (review 3, #9).
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.requires_grad:
                captured["v"] = t
                captured["n_grad"] = captured.get("n_grad", 0) + 1
            else:
                captured["n_nograd"] = captured.get("n_nograd", 0) + 1

        handle = projector.register_forward_hook(_hook)

        vla_start = vla_end = ce_end = None
        proj_calls = None
        try:
            for step in range(self.iters):
                adv = (clean + delta).clamp(0, 1)
                inputs = dict(full_inputs)
                # move only tensors -- Gemma's adapter also returns
                # num_soft_tokens_per_image as a plain Python list
                _ai = adapter.build(adv.squeeze(0), plan)
                inputs.update({k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                               for k, v in _ai.items()})
                inputs["labels"] = labels

                outputs = self.model(**inputs)
                ce_loss = outputs.loss

                v = captured.get("v")
                if v is None:
                    raise RuntimeError(
                        f"Projector hook at {projector_path!r} produced no output with a gradient "
                        f"path ({captured.get('n_nograd', 0)} call(s) without one) -- VLA would be "
                        "silently inert. Fix the projector path before trusting any result."
                    )
                if step == 0:
                    proj_calls = (captured.get("n_grad", 0), captured.get("n_nograd", 0))
                v_mean = v.reshape(-1, v.shape[-1]).mean(dim=0)
                vla_loss = 1.0 - F.cosine_similarity(
                    v_mean.unsqueeze(0).float(), target_embed.unsqueeze(0).float()
                ).squeeze(0)

                loss = ce_loss + self.vla_weight * vla_loss
                loss.backward()

                if step == 0:
                    vla_start = float(vla_loss.detach())
                vla_end, ce_end = float(vla_loss.detach()), float(ce_loss.detach())

                with torch.no_grad():
                    delta.grad.sign_()
                    delta.data = delta.data - self.alpha * delta.grad
                    delta.data.clamp_(-self.epsilon, self.epsilon)
                    delta.data = torch.clamp(clean + delta, 0, 1) - clean
                    delta.grad.zero_()

                del outputs, loss, ce_loss, vla_loss
                captured.clear()
                torch.cuda.empty_cache()
        finally:
            handle.remove()

        adv_image = self._visual._tensor_to_pil((clean + delta).clamp(0, 1))

        return AttackResult(
            attack_name=self.name,
            modality=self.modality,
            adversarial_image=adv_image,
            adversarial_text=adversarial_text,
            target_behavior=target_behavior,
            metadata={
                "vla_weight": self.vla_weight,
                "projector_path": projector_path,
                # (calls with a gradient path, calls without) at step 0.
                "projector_calls_grad_nograd": proj_calls,
                # If vla_loss_end is not below vla_loss_start, VLA did not
                # actually optimise anything and the result is not CrossInject.
                "vla_loss_start": vla_start,
                "vla_loss_end": vla_end,
                "ce_loss_end": ce_end,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "iters": self.iters,
                "adapter": adapter.name,
                "target_resolution": (plan.target_h, plan.target_w),
                "textual": text_result.metadata,
                "coordinated_prompt": coordinated_prompt,
                "tge_faithful": False,  # fixed templates, not a learned generator
            },
        )

    @staticmethod
    def to_vwa_adv_delivery(result: AttackResult, url: str):
        """Adapts an AttackResult into VWA-Adv's (adv_url2caption,
        adv_url2image) pair, keyed on the image URL."""
        cap = {url: result.adversarial_text} if result.adversarial_text else {}
        img = {url: result.adversarial_image} if result.adversarial_image else {}
        return cap, img
