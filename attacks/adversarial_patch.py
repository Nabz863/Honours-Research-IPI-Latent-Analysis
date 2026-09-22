"""
attacks/adversarial_patch.py
------------------------------
Adversarial patch (Table 3.2, visual baseline).
Bailey et al. 2023, "Image hijacks: adversarial images can control
generative models at runtime".

HOW THIS DIFFERS FROM pixel_perturbation.py -- the distinction matters,
since both are gradient attacks on pixels and it would be easy to
present them as the same thing:

  pixel_perturbation : perturbation spread over the WHOLE image, tightly
      bounded (L_inf <= epsilon = 16/255). Imperceptible by construction;
      each pixel barely moves.
  adversarial_patch  : perturbation confined to a SMALL REGION, but
      UNBOUNDED within it (any pixel value in [0,1]). Clearly visible
      where it is, untouched everywhere else.

Same threat model family, opposite constraint geometry -- a budget
spread thin versus a budget concentrated. That contrast is the reason
to run both: it tests whether these models are vulnerable to subtle
global shifts, to strong local ones, or to neither.

The patch framing is also the more realistic web threat: an attacker
who can upload an image controls its content outright, but generally
cannot apply an imperceptible delta to someone ELSE'S served image.

REUSE: subclasses PixelPerturbationAttack to inherit its verified
AnyRes tiling (_build_anyres_patches, _AnyResGeometry) -- getting that
right took a real GPU run to discover and a source-verified rewrite to
fix, and duplicating it here would risk the two drifting apart.
generate() is reimplemented rather than parameterised because the
optimisation constraint genuinely differs (masked, unbounded) and
threading a flag through the parent's loop would obscure both.

STATUS: NOT yet run against real model weights. The AnyRes machinery it
inherits is verified; the masked-update loop specific to this file is
not.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import AttackModality, AttackResult
from .pixel_perturbation import PixelPerturbationAttack, _AnyResGeometry


class AdversarialPatchAttack(PixelPerturbationAttack):
    """Spatially-confined, unbounded-within-region pixel optimisation."""

    name = "adversarial_patch"
    modality = AttackModality.VISUAL

    def __init__(
        self,
        model,
        processor,
        patch_frac: float = 0.25,
        position: str = "center",
        alpha: float = 4 / 255,
        iters: int = 200,
    ):
        """
        patch_frac : patch edge length as a fraction of the image's SHORTER
            side. 0.25 -> a square covering ~6% of image area. This is the
            attack's real strength knob; the patch literature commonly
            reports ASR as a function of patch area, so it is worth
            treating as a swept parameter rather than a fixed default.
        position : "center" | "top_left" | "bottom_right" | "random".
            Centre is the strongest default (vision encoders weight
            central content heavily) and the least realistic; corner
            placements are more plausible for a real uploaded image.
        alpha : sign-gradient step size. Larger than
            pixel_perturbation's 1/255 default because there is no
            epsilon ball to stay inside -- the patch can traverse the
            full [0,1] range, so a tiny step wastes iterations.
        iters : gradient steps. Same cost profile as pixel_perturbation
            (one full forward+backward per step).
        """
        super().__init__(model, processor, epsilon=1.0, alpha=alpha, iters=iters)
        if position not in ("center", "top_left", "bottom_right", "random"):
            raise ValueError(f"position must be center/top_left/bottom_right/random, got {position!r}")
        self.patch_frac = patch_frac
        self.position = position

    def _patch_bounds(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """Returns (y0, y1, x0, x1) for the patch region."""
        side = max(1, int(min(height, width) * self.patch_frac))
        side = min(side, height, width)
        if self.position == "center":
            y0, x0 = (height - side) // 2, (width - side) // 2
        elif self.position == "top_left":
            y0, x0 = 0, 0
        elif self.position == "bottom_right":
            y0, x0 = height - side, width - side
        else:  # random
            y0 = int(torch.randint(0, max(1, height - side + 1), (1,)).item())
            x0 = int(torch.randint(0, max(1, width - side + 1), (1,)).item())
        return y0, y0 + side, x0, x0 + side

    def generate(
        self,
        base_input: Dict[str, Any],
        target_behavior: str,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> AttackResult:
        image = image or base_input.get("image")
        if image is None:
            raise ValueError("AdversarialPatchAttack needs a base image (pass image= or base_input['image']).")

        prompt = prompt or base_input.get("prompt", "Describe this image.")
        image_token = getattr(self.processor, "image_token", None)
        if image_token and image_token not in prompt:
            prompt = f"{image_token}\n{prompt}"

        device = next(self.model.parameters()).device
        image = image.convert("RGB")
        clean = torch.from_numpy(np.array(image).astype("float32") / 255.0)
        clean = clean.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]

        h, w = clean.shape[-2], clean.shape[-1]
        y0, y1, x0, x1 = self._patch_bounds(h, w)
        mask = torch.zeros_like(clean)
        mask[:, :, y0:y1, x0:x1] = 1.0

        # Initialise the patch region to mid-grey rather than to the
        # original content: an unbounded patch has no reason to start
        # near the clean image, and a neutral start avoids biasing the
        # optimisation toward whatever happened to be underneath.
        delta = torch.zeros_like(clean)
        delta[:, :, y0:y1, x0:x1] = 0.5 - clean[:, :, y0:y1, x0:x1]
        delta = delta.detach().requires_grad_(True)

        from .geometry_adapters import get_adapter
        adapter = get_adapter(self.processor)
        plan = adapter.plan(h, w)

        full_text = f"{prompt} {target_behavior}"
        prompt_only = self.processor(images=image, text=prompt, return_tensors="pt")
        prompt_len = prompt_only["input_ids"].shape[1]
        full_inputs = self.processor(images=image, text=full_text, return_tensors="pt")
        full_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in full_inputs.items()}
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full_inputs["use_cache"] = False

        for p in self.model.parameters():
            p.requires_grad_(False)
        # train() so HF's gradient checkpointing actually engages (the flag
        # alone is gated on self.training inside decoder layers); Dropout
        # forced back to eval for deterministic gradients. Same fix and
        # same reasoning as pixel_perturbation.py.
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d,
                              torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        final_loss = None
        for _ in range(self.iters):
            adv = (clean + delta).clamp(0, 1)
            inputs = dict(full_inputs)
            # move only tensors -- Gemma's adapter also returns
            # num_soft_tokens_per_image as a plain Python list
            _ai = adapter.build(adv.squeeze(0), plan)
            inputs.update({k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                           for k, v in _ai.items()})
            inputs["labels"] = labels

            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            final_loss = float(loss.detach())

            with torch.no_grad():
                grad = delta.grad
                grad.sign_()
                # Masked step: only the patch region moves. No epsilon
                # clamp -- the patch is unbounded within [0,1], which is
                # the defining difference from pixel_perturbation.
                delta.data = delta.data - self.alpha * grad * mask
                delta.data = (torch.clamp(clean + delta, 0, 1) - clean) * mask
                delta.grad.zero_()

            del outputs, loss
            torch.cuda.empty_cache()

        adv_final = self._tensor_to_pil((clean + delta).clamp(0, 1))

        return AttackResult(
            attack_name=self.name,
            modality=self.modality,
            adversarial_image=adv_final,
            target_behavior=target_behavior,
            metadata={
                "patch_frac": self.patch_frac,
                "position": self.position,
                "patch_bounds_yxyx": (y0, y1, x0, x1),
                "patch_area_frac": float(((y1 - y0) * (x1 - x0)) / (h * w)),
                "alpha": self.alpha,
                "iters": self.iters,
                "final_loss": final_loss,
                "adapter": adapter.name,
                "target_resolution": (plan.target_h, plan.target_w),
            },
        )
