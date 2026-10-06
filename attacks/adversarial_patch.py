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

REUSE: subclasses PixelPerturbationAttack for its settings; the shared
optimisation loop takes the patch mask and the unbounded step.

P2b (5 Oct): retargeted at the self-captioner, sharing pixel_perturbation's
objective (attacks/captioner_objective.py): the patch, mid-grey at the start
and unbounded within [0,1], is optimised so the agent model captions the image
as the task's target caption. Tested on a tiny real Qwen3-VL (tests/test_p2b.py:
nothing outside the patch changes); real weights: the P2b smoke.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import AttackModality, AttackResult
from .pixel_perturbation import PixelPerturbationAttack


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
        iters: int = 500,
        check_every: int = 50,
        max_seconds: float = 1200.0,
        opt_max_side: int = 768,
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
        super().__init__(model, processor, epsilon=1.0, alpha=alpha, iters=iters,
                         check_every=check_every, max_seconds=max_seconds, opt_max_side=opt_max_side)
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
        else:  # random -- seeded by the image shape, so a retry reproduces it
            g = torch.Generator().manual_seed(height * 100003 + width)
            y0 = int(torch.randint(0, max(1, height - side + 1), (1,), generator=g).item())
            x0 = int(torch.randint(0, max(1, width - side + 1), (1,), generator=g).item())
        return y0, y0 + side, x0, x0 + side

    def generate(
        self,
        base_input: Dict[str, Any],
        target_behavior: str,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> AttackResult:
        """Retargeted at the self-captioner (P2b), like pixel_perturbation:
        the patch is optimised so the agent model captions the image as the
        task's target caption. The patch region starts mid-grey and is
        unbounded within [0,1]; nothing outside it changes."""
        from .captioner_objective import optimise_caption, target_caption_of
        image = image or base_input.get("image")
        if image is None:
            raise ValueError("AdversarialPatchAttack needs a base image (pass image= or base_input['image']).")
        target = target_caption_of(base_input)
        bounds: Dict[str, Any] = {}

        def _mask(h: int, w: int) -> torch.Tensor:
            y0, y1, x0, x1 = self._patch_bounds(h, w)
            bounds.update(yxyx=(y0, y1, x0, x1), area=((y1 - y0) * (x1 - x0)) / (h * w))
            m = torch.zeros(1, 1, h, w)
            m[:, :, y0:y1, x0:x1] = 1.0
            return m

        adv, meta = optimise_caption(
            self.model, self.processor, image, target,
            epsilon=1.0, alpha=self.alpha, iters=self.iters, check_every=self.check_every,
            max_seconds=self.max_seconds, opt_max_side=self.opt_max_side,
            mask_fn=_mask, init="grey")
        meta.update({"patch_frac": self.patch_frac, "position": self.position,
                     "patch_bounds_yxyx": bounds.get("yxyx"), "patch_area_frac": bounds.get("area")})
        return AttackResult(attack_name=self.name, modality=self.modality, adversarial_image=adv,
                            adversarial_text=None, target_behavior=target_behavior, metadata=meta)
