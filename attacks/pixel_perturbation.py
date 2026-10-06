"""
attacks/pixel_perturbation.py
-------------------------------
Adversarial pixel perturbation (Table 3.2, visual baseline).
Qi et al. 2023; BIM (Kurakin et al.) family, l_inf bounded.

P2b (5 Oct): retargeted at the self-captioner -- VWA-Adv's own captioner
attack, white-box on the agent model. The objective, sizes and budget live in
attacks/captioner_objective.py; this class is a thin wrapper. The LLaVA AnyRes
geometry below is kept only because attacks/geometry_adapters.py's AnyRes
adapter reuses it.

(Earlier notes, kept for the LLaVA port:)

ANYRES TILING (faithful, not scoped-down): a real GPU run confirmed
LLaVA-NeXT's processor splits ANY image -- including a 336x336 synthetic
test square -- into a base/global patch plus one or more grid tiles
(image_grid_pinpoints tie-break selects by minimum wasted padding area,
not squareness). This module reconstructs that tiling differentiably,
matching transformers' real image_processing_llava_next.py (main
branch, fetched and read directly -- not guessed):
  - select_best_resolution: verified character-for-character against
    real source, and independently hand-traced against the actual
    observed 3-patch output before being trusted.
  - Patch order: [resized_original_image] + patches -- base/global
    FIRST, then grid tiles -- confirmed directly from source.
  - get_patch_output_size (aspect-preserving fit-within-bounds resize):
    NOT independently re-verified character-for-character -- it's a
    shared utility imported from elsewhere in transformers, not defined
    in the file that was fetched. What's implemented below is the
    standard fit-within-bounds formula its call signature requires.
  - divide_to_patches' row-major tiling order: standard convention, not
    independently re-verified from this fetch.
  - Padding: torchvision's tvF.pad with no explicit fill= in the real
    call -- implemented here as fill=0 (black), matching torchvision's
    documented default.

STATUS: mechanics verified against a synthetic fake-HF-model stand-in
for the pre-AnyRes version -- NOT yet re-run against real weights since
this rewrite. Run the smoke test via sbatch before trusting it further.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import AttackModality, AttackResult, BaseAttack


# Prefer transformers' OWN geometry functions over the reimplementations
# below. Verified importable (they are public in image_processing_utils and
# imported this way by third-party fast processors). This removes the
# "reconstructed, not line-verified" caveat on get_patch_output_size --
# when this import succeeds we are calling the exact code the processor
# calls. The local versions remain as a fallback for older/newer layouts,
# and _REAL_HF_GEOMETRY is recorded in attack metadata so any result can
# be traced to which path produced it.
_REAL_HF_GEOMETRY = False
try:
    from transformers.image_processing_utils import (
        get_patch_output_size as _hf_get_patch_output_size,
        select_best_resolution as _hf_select_best_resolution,
    )
    _REAL_HF_GEOMETRY = True
except Exception:
    _hf_get_patch_output_size = None
    _hf_select_best_resolution = None


def _select_best_resolution(
    original_size: Tuple[int, int], possible_resolutions: List[List[int]]
) -> Tuple[int, int]:
    """Faithful port of transformers.image_processing_utils.select_best_resolution.
    Verified against real source and hand-traced against observed output."""
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")
    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)
    return best_fit


def _get_patch_output_size(
    original_height: int, original_width: int, target_height: int, target_width: int
) -> Tuple[int, int]:
    """NOT independently re-verified character-for-character (see module
    docstring) -- standard fit-within-bounds computation."""
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)
    return new_height, new_width


def _get_centering_padding(
    new_height: int, new_width: int, target_height: int, target_width: int
) -> Tuple[int, int, int, int]:
    """Faithful port of _get_padding_size. Returns (left, right, top, bottom)
    for torch.nn.functional.pad's argument order."""
    paste_x, r_x = divmod(target_width - new_width, 2)
    paste_y, r_y = divmod(target_height - new_height, 2)
    return paste_x, paste_x + r_x, paste_y, paste_y + r_y


class _AnyResGeometry:
    """All AnyRes tiling geometry for ONE fixed original image size --
    computed once, reused every iteration."""

    def __init__(self, orig_height: int, orig_width: int, grid_pinpoints: List[List[int]], patch_size: int):
        self.orig_height = orig_height
        self.orig_width = orig_width
        self.patch_size = patch_size
        if _REAL_HF_GEOMETRY:
            import numpy as _np
            self.target_height, self.target_width = _hf_select_best_resolution(
                (orig_height, orig_width), grid_pinpoints
            )
            # get_patch_output_size reads only the SHAPE of its image argument,
            # so a zero-filled array of the right dimensions is sufficient and
            # avoids copying real pixel data every call.
            _dummy = _np.zeros((orig_height, orig_width, 3), dtype=_np.uint8)
            self.resize_height, self.resize_width = _hf_get_patch_output_size(
                _dummy, (self.target_height, self.target_width), None
            )
        else:
            self.target_height, self.target_width = _select_best_resolution(
                (orig_height, orig_width), grid_pinpoints
            )
            self.resize_height, self.resize_width = _get_patch_output_size(
                orig_height, orig_width, self.target_height, self.target_width
            )
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = _get_centering_padding(
            self.resize_height, self.resize_width, self.target_height, self.target_width
        )
        if self.target_height % patch_size != 0 or self.target_width % patch_size != 0:
            raise ValueError(
                f"Selected AnyRes target resolution ({self.target_height}, {self.target_width}) "
                f"is not an exact multiple of patch_size={patch_size}."
            )
        self.grid_rows = self.target_height // patch_size
        self.grid_cols = self.target_width // patch_size
        self.num_patches = 1 + self.grid_rows * self.grid_cols


class PixelPerturbationAttack(BaseAttack):
    """BIM-style l_inf pixel perturbation against the SELF-CAPTIONER (P2b).

    Retargeted (decision 30 Sep): the image is optimised so that the agent
    model, captioning it with the episode's own captioning request, writes
    the task's adversarial target caption -- VWA-Adv's own captioner
    attack, white-box on our model. See attacks/captioner_objective.py for
    what matches the paper and what does not. The earlier objective (the
    goal text, after a raw untemplated prompt, on the image alone; audit
    12.6) is gone: it optimised a context the agent never sees."""

    name = "pixel_perturbation"
    modality = AttackModality.VISUAL

    def __init__(self, model, processor, epsilon: float = 16 / 255, alpha: float = 1 / 255,
                 iters: int = 500, check_every: int = 50, max_seconds: float = 1200.0,
                 opt_max_side: int = 768):
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.check_every = check_every
        self.max_seconds = max_seconds
        self.opt_max_side = opt_max_side

    def generate(
        self,
        base_input: Dict[str, Any],
        target_behavior: str,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> AttackResult:
        from .captioner_objective import optimise_caption, target_caption_of
        image = image or base_input.get("image")
        if image is None:
            raise ValueError("PixelPerturbationAttack needs a base image (pass image= or base_input['image']).")
        target = target_caption_of(base_input)
        adv, meta = optimise_caption(
            self.model, self.processor, image, target,
            epsilon=self.epsilon, alpha=self.alpha, iters=self.iters, check_every=self.check_every,
            max_seconds=self.max_seconds, opt_max_side=self.opt_max_side)
        return AttackResult(attack_name=self.name, modality=self.modality, adversarial_image=adv,
                            adversarial_text=None, target_behavior=target_behavior, metadata=meta)

    def _build_anyres_patches(self, image_tensor: torch.Tensor, geometry: "_AnyResGeometry") -> torch.Tensor:
        img_proc = self.processor.image_processor
        mean = getattr(img_proc, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(img_proc, "image_std", [0.5, 0.5, 0.5])
        mean_t = torch.tensor(mean, device=image_tensor.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=image_tensor.device).view(1, -1, 1, 1)

        def normalize(t):
            return (t - mean_t) / std_t

        base_patch = F.interpolate(image_tensor, size=self._base_size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        base_patch = normalize(base_patch)

        resized = F.interpolate(
            image_tensor, size=(geometry.resize_height, geometry.resize_width),
            mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
        padded = F.pad(
            resized,
            (geometry.pad_left, geometry.pad_right, geometry.pad_top, geometry.pad_bottom),
            mode="constant",
            value=0.0,
        )

        ps = geometry.patch_size
        tiles = []
        for row in range(geometry.grid_rows):
            for col in range(geometry.grid_cols):
                tile = padded[:, :, row * ps:(row + 1) * ps, col * ps:(col + 1) * ps]
                tiles.append(normalize(tile))

        all_patches = torch.cat([base_patch] + tiles, dim=0)
        return all_patches.unsqueeze(0)

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        arr = (tensor.squeeze(0).detach().cpu().numpy() * 255).astype("uint8").transpose(1, 2, 0)
        return Image.fromarray(arr)
