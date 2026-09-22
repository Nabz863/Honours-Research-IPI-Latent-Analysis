"""
attacks/geometry_adapters.py
----------------------------
Architecture-agnostic differentiable image preprocessing, so pixel-space
attacks (pixel_perturbation, adversarial_patch, cross_modal_injection)
stop being LLaVA-NeXT-only.

WHY THIS EXISTS
    Those three attacks require processor.image_processor to expose
    `image_grid_pinpoints` and `crop_size` -- LLaVA-NeXT AnyRes
    attributes. Verified on the installed weights:

        llava-hf/llava-v1.6-mistral-7b-hf   pinpoints=True   crop=True
        OpenGVLab/InternVL3-8B-hf           pinpoints=False  crop=True
        Qwen/Qwen3-VL-8B-Instruct           pinpoints=False  crop=True

    so on the two non-LLaVA models they raise before doing any work.
    Neither ships its own image processor:
        Qwen3-VL -> Qwen2VLImageProcessor (models/qwen2_vl/)
        InternVL -> GotOcr2ImageProcessor (models/got_ocr2/)

HOW THIS AVOIDS RE-IMPLEMENTING THE RESAMPLER
    Both processors subclass TorchvisionBackend and work on torch
    tensors, so `ip.resize` (torchvision tvF.resize) and
    `ip.rescale_and_normalize` are ALREADY differentiable. The adapters
    call the processor's own methods instead of substituting
    F.interpolate, which removes the PIL-vs-torch bicubic mismatch
    entirely. Only reshape/permute/tiling geometry is reproduced here,
    copied from the installed `_preprocess`.

REMAINING SOURCE OF ERROR
    The real pipeline resizes a uint8 [0,255] tensor; gradients need
    float, and float resize skips uint8's integer rounding. A small
    quantisation residual is expected. verify_adapter() measures it.

Port notes -- what the source contradicted in the first draft:
  * Qwen min/max pixels come from size["shortest_edge"]/["longest_edge"],
    NOT from `min_pixels`/`max_pixels` attributes.
  * Qwen's permute is (0,2,5,3,6,1,4,7) over [B,C,gh/m,m,ps,gw/m,m,ps],
    and temporal is inserted AFTER channel via unsqueeze(6).expand(...),
    giving C*T*ps*ps -- not temporal-major.
  * InternVL keeps the LARGER grid on an aspect-ratio tie only while
    area > 0.5*tile_h*tile_w*cols*rows, and its thumbnail is resized
    from the ORIGINAL image, not from the tiled canvas.
  * InternVL's use_thumbnail is not configurable: `_preprocess` calls
    crop_image_to_patches without it, so it is always True.
  * crop_to_patches defaults False on the image processor, but
    InternVLProcessorKwargs._defaults forces it True.

THIS FILE IS NOT TRUSTWORTHY UNTIL verify_adapter() PASSES.
    A wrong permute still yields a correctly-SHAPED tensor, so the
    failure is silent and reads as "the attack doesn't work". Run the
    gate for every (adapter, model) pair before any attack uses this.

Usage
    from attacks.geometry_adapters import get_adapter, verify_adapter
    ad = get_adapter(processor)
    plan = ad.plan(h, w)
    kwargs = ad.build(adv01, plan)        # {"pixel_values": ..., ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch

try:
    from transformers.image_utils import SizeDict
except ImportError:                                    # older layouts
    from transformers.image_processing_utils import SizeDict  # type: ignore


# ---------------------------------------------------------------------------

def _size_pair(size) -> Tuple[int, int]:
    if isinstance(size, dict):
        return int(size["height"]), int(size["width"])
    return int(size.height), int(size.width)


@dataclass
class Plan:
    """Integer geometry for ONE fixed input size. No tensors, no grads."""
    kind: str
    target_h: int
    target_w: int
    extra: Dict[str, Any] = field(default_factory=dict)


class GeometryAdapter:
    name = "base"

    def __init__(self, processor):
        self.processor = processor
        self.ip = getattr(processor, "image_processor", processor)

    @classmethod
    def supports(cls, processor) -> bool:
        raise NotImplementedError

    def plan(self, orig_h: int, orig_w: int) -> Plan:
        raise NotImplementedError

    def build(self, adv01: torch.Tensor, plan: Plan) -> Dict[str, torch.Tensor]:
        """adv01: float [3,H,W] in [0,1], may require grad."""
        raise NotImplementedError

    # -- shared: call the processor's OWN ops so resampling matches --------

    def _resize(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """The real pipeline resizes a uint8 tensor, so torchvision clamps
        bicubic overshoot to [0,255] and rounds. Float resize does neither,
        which shows up as a large MAX error at high-contrast edges while the
        mean stays tiny. Replicate both, using a straight-through estimator
        for round() so gradients still reach image space."""
        y = self.ip.resize(image=x, size=SizeDict(height=h, width=w),
                           resample=self.ip.resample)
        y = y.clamp(0.0, 255.0)
        return y + (y.round() - y).detach()

    def _rescale_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.ip.rescale_and_normalize(
            x,
            getattr(self.ip, "do_rescale", True),
            getattr(self.ip, "rescale_factor", 1 / 255.0),
            getattr(self.ip, "do_normalize", True),
            self.ip.image_mean,
            self.ip.image_std,
        )


# ---------------------------------------------------------------------------
# LLaVA-NeXT / AnyRes -- wraps the already-verified port, unchanged
# ---------------------------------------------------------------------------

class AnyResAdapter(GeometryAdapter):
    name = "anyres"

    @classmethod
    def supports(cls, processor) -> bool:
        ip = getattr(processor, "image_processor", processor)
        return (getattr(ip, "image_grid_pinpoints", None) is not None
                and getattr(ip, "crop_size", None) is not None)

    def plan(self, orig_h, orig_w) -> Plan:
        from .pixel_perturbation import _AnyResGeometry
        cs = self.ip.crop_size
        patch = cs["height"] if isinstance(cs, dict) else cs.height
        proc_size = getattr(self.ip, "size", None)
        base_edge = (
            proc_size.get("shortest_edge") if isinstance(proc_size, dict)
            else getattr(proc_size, "shortest_edge", None)
        ) or patch
        # _build_anyres_patches reads self._base_size off the instance --
        # the original sets it inside generate(), so the adapter must set
        # it here or the borrowed method raises AttributeError.
        self._base_size = (base_edge, base_edge)
        g = _AnyResGeometry(orig_h, orig_w, self.ip.image_grid_pinpoints, patch)
        return Plan("anyres", g.target_height, g.target_width,
                    {"geometry": g, "base_edge": base_edge})

    def build(self, adv01, plan):
        from .pixel_perturbation import PixelPerturbationAttack
        if not hasattr(self, "_base_size"):
            self._base_size = (plan.extra["base_edge"],) * 2
        # the borrowed method expects a batched [1,C,H,W] tensor in [0,1]
        x = adv01.unsqueeze(0) if adv01.dim() == 3 else adv01
        pv = PixelPerturbationAttack._build_anyres_patches(
            self, x, plan.extra["geometry"])
        return {"pixel_values": pv}


# ---------------------------------------------------------------------------
# Qwen2/3-VL -- smart_resize + flat patch sequence
# Ported from models/qwen2_vl/image_processing_qwen2_vl.py::_preprocess
# ---------------------------------------------------------------------------

class QwenPatchAdapter(GeometryAdapter):
    name = "qwen_patch"

    @classmethod
    def supports(cls, processor) -> bool:
        ip = getattr(processor, "image_processor", processor)
        return (getattr(ip, "merge_size", None) is not None
                and getattr(ip, "temporal_patch_size", None) is not None
                and getattr(ip, "image_grid_pinpoints", None) is None)

    def plan(self, orig_h, orig_w) -> Plan:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        ip = self.ip
        patch = int(ip.patch_size)
        merge = int(ip.merge_size)
        temporal = int(ip.temporal_patch_size)

        size = ip.size
        shortest = size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
        longest = size["longest_edge"] if isinstance(size, dict) else size.longest_edge

        th, tw = smart_resize(orig_h, orig_w, factor=patch * merge,
                              min_pixels=shortest, max_pixels=longest)
        return Plan("qwen_patch", th, tw,
                    {"patch": patch, "merge": merge, "temporal": temporal,
                     "grid_h": th // patch, "grid_w": tw // patch})

    def build(self, adv01, plan):
        e = plan.extra
        patch, merge, temporal = e["patch"], e["merge"], e["temporal"]
        gh, gw = e["grid_h"], e["grid_w"]

        # the processor consumes [0,255]-scale tensors; do_rescale applies 1/255
        x = (adv01 * 255.0).unsqueeze(0)                      # [1,C,H,W]
        x = self._resize(x, plan.target_h, plan.target_w)
        patches = self._rescale_normalize(x)

        b, c = patches.shape[:2]
        patches = patches.reshape(b, c,
                                  gh // merge, merge, patch,
                                  gw // merge, merge, patch)
        # -> [B, gh/m, gw/m, m, m, C, ps, ps]   (exact order from _preprocess)
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        # temporal inserted AFTER channel, giving C * T * ps * ps
        flat = (patches.unsqueeze(6)
                .expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
                .reshape(b, gh * gw, c * temporal * patch * patch))

        pixel_values = flat.reshape(b * gh * gw, c * temporal * patch * patch)
        grid = torch.tensor([[1, gh, gw]], dtype=torch.long, device=adv01.device)
        return {"pixel_values": pixel_values, "image_grid_thw": grid}


# ---------------------------------------------------------------------------
# InternVL (GotOcr2ImageProcessor) -- dynamic tiled canvas + thumbnail
# Ported from models/got_ocr2/image_processing_got_ocr2.py
# ---------------------------------------------------------------------------

class InternVLTileAdapter(GeometryAdapter):
    name = "internvl_tile"

    @classmethod
    def supports(cls, processor) -> bool:
        ip = getattr(processor, "image_processor", processor)
        if getattr(ip, "image_grid_pinpoints", None) is not None:
            return False
        if getattr(ip, "merge_size", None) is not None:
            return False
        return hasattr(ip, "min_patches") and hasattr(ip, "max_patches")

    def plan(self, orig_h, orig_w) -> Plan:
        from transformers.models.got_ocr2.image_processing_got_ocr2 import (
            get_optimal_tiled_canvas,
        )
        ip = self.ip
        tile_h, tile_w = _size_pair(ip.size)
        min_p = int(getattr(ip, "min_patches", 1))
        max_p = int(getattr(ip, "max_patches", 12))

        # InternVLProcessorKwargs._defaults forces crop_to_patches=True,
        # overriding the image processor's own False default.
        cols, rows = get_optimal_tiled_canvas(
            (orig_h, orig_w), (tile_h, tile_w), min_p, max_p)

        return Plan("internvl_tile", rows * tile_h, cols * tile_w,
                    {"tile_h": tile_h, "tile_w": tile_w,
                     "cols": cols, "rows": rows, "crop": True})

    def build(self, adv01, plan):
        e = plan.extra
        tile_h, tile_w, cols, rows = e["tile_h"], e["tile_w"], e["cols"], e["rows"]

        x = (adv01 * 255.0).unsqueeze(0)                      # [1,C,H,W]

        if not e["crop"]:
            return {"pixel_values": self._rescale_normalize(
                self._resize(x, tile_h, tile_w))}

        resized = self._resize(x, plan.target_h, plan.target_w)

        tiles = []
        for i in range(cols * rows):                          # row-major
            col, row = i % cols, i // cols
            tiles.append(resized[..., row * tile_h:(row + 1) * tile_h,
                                 col * tile_w:(col + 1) * tile_w])

        # thumbnail comes from the ORIGINAL image, appended only when
        # there is more than one tile
        if len(tiles) != 1:
            tiles.append(self._resize(x, tile_h, tile_w))

        stacked = torch.stack(tiles, dim=0).transpose(0, 1).contiguous()
        flat = stacked.reshape(-1, *stacked.shape[2:])        # [N,C,th,tw]

        # _preprocess resizes each patch to `size` again (a no-op here --
        # tiles are already exactly size) then normalises
        return {"pixel_values": self._rescale_normalize(flat)}



# ---------------------------------------------------------------------------
# Gemma 4 Unified -- aspect-ratio-preserving resize + patch merging
# Ported from models/gemma4_unified/image_processing_gemma4_unified.py
# ---------------------------------------------------------------------------

class Gemma4UnifiedAdapter(GeometryAdapter):
    """Unlike the other three, this reimplements NOTHING.

    Gemma4UnifiedImageProcessor's whole pipeline is already torch ops on
    tensors -- get_aspect_ratio_preserving_size, convert_image_to_patches,
    patches_merge and pad_along_first_dim are module-level functions, and
    aspect_ratio_preserving_resize is a method. So this adapter calls the
    real ones directly and inherits their behaviour exactly. The argsort
    inside patches_merge produces indices, not values, so gradients still
    flow through the gather.

    Note do_normalize is False for this processor (image_mean [0,0,0],
    image_std [1,1,1]) -- only rescaling is applied, which the shared
    _rescale_normalize already reads off the processor rather than
    assuming.

    Output is a patch SEQUENCE of shape [1, max_soft_tokens, (pool*patch)^2*3]
    plus image_position_ids, not a pixel grid.
    """

    name = "gemma4_unified"

    @classmethod
    def supports(cls, processor) -> bool:
        ip = getattr(processor, "image_processor", processor)
        return (getattr(ip, "max_soft_tokens", None) is not None
                and getattr(ip, "pooling_kernel_size", None) is not None)

    def plan(self, orig_h, orig_w) -> Plan:
        from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
            get_aspect_ratio_preserving_size)
        ip = self.ip
        patch = int(ip.patch_size)
        pool = int(ip.pooling_kernel_size)
        mst = int(ip.max_soft_tokens)
        th, tw = get_aspect_ratio_preserving_size(
            height=orig_h, width=orig_w, patch_size=patch,
            max_patches=mst * pool ** 2, pooling_kernel_size=pool)
        return Plan("gemma4_unified", th, tw,
                    {"patch": patch, "pool": pool, "max_soft_tokens": mst})

    def build(self, adv01, plan):
        from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
            convert_image_to_patches, patches_merge, pad_along_first_dim)
        e = plan.extra
        patch, pool, mst = e["patch"], e["pool"], e["max_soft_tokens"]

        x = adv01 * 255.0                                   # [C,H,W]
        x = self.ip.aspect_ratio_preserving_resize(
            image=x, patch_size=patch, max_patches=mst * pool ** 2,
            pooling_kernel_size=pool, resample=self.ip.resample)
        x = self._rescale_normalize(x)

        ph, pw = x.shape[-2] // patch, x.shape[-1] // patch
        teacher = convert_image_to_patches(x, patch)
        grid = torch.meshgrid(torch.arange(pw, device=x.device),
                              torch.arange(ph, device=x.device), indexing="xy")
        pos = torch.stack(grid, dim=-1).reshape(teacher.shape[0], 2)

        n_model = teacher.shape[0] // (pool ** 2)
        merged, mpos = patches_merge(teacher.unsqueeze(0), pos.unsqueeze(0), n_model)
        merged, mpos = merged.squeeze(0), mpos.squeeze(0)
        n_soft = merged.shape[0]
        merged, mpos = pad_along_first_dim(merged, mpos, mst)

        return {"pixel_values": merged.unsqueeze(0),
                "image_position_ids": mpos.unsqueeze(0),
                "num_soft_tokens_per_image": [n_soft]}


# ---------------------------------------------------------------------------

_ADAPTERS = [AnyResAdapter, QwenPatchAdapter, InternVLTileAdapter, Gemma4UnifiedAdapter]


def get_adapter(processor) -> GeometryAdapter:
    for cls in _ADAPTERS:
        if cls.supports(processor):
            return cls(processor)
    ip = getattr(processor, "image_processor", processor)
    seen = [a for a in ("image_grid_pinpoints", "crop_size", "merge_size",
                        "patch_size", "temporal_patch_size", "min_patches",
                        "max_patches")
            if getattr(ip, a, None) is not None]
    raise ValueError(
        f"No geometry adapter matches {type(ip).__name__}. Attributes seen: "
        + (", ".join(seen) or "none"))


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------

def verify_adapter(processor, image, atol: float = 1e-3, verbose: bool = True):
    """Push one real image through the adapter AND the real processor,
    then compare. Returns (passed, report).

    atol is tight (1e-3 in normalised units) because the adapter calls
    the processor's own resize -- the only expected residual is
    uint8-vs-float quantisation. A large error means the geometry is
    wrong, not the resampler. For scale: eps=8/255 is ~0.031 raw and
    ~0.115 after dividing by a 0.27 normalisation std.
    """
    import numpy as np

    ad = get_adapter(processor)
    tok = getattr(processor, "image_token", None) or "<image>"
    ref = processor(images=image, text=str(tok), return_tensors="pt")
    if "pixel_values" not in ref:
        return False, {"error": "processor returned no pixel_values"}
    ref_pv = ref["pixel_values"].float()

    arr = (torch.from_numpy(np.array(image.convert("RGB")))
           .permute(2, 0, 1).float() / 255.0)
    plan = ad.plan(arr.shape[1], arr.shape[2])
    got = ad.build(arr, plan)
    got_pv = got["pixel_values"].float()

    rep = {
        "adapter": ad.name,
        "processor": type(getattr(processor, "image_processor", processor)).__name__,
        "plan_hw": (plan.target_h, plan.target_w),
        "ref_shape": tuple(ref_pv.shape),
        "got_shape": tuple(got_pv.shape),
    }

    r = ref_pv
    if r.dim() == got_pv.dim() + 1 and r.shape[0] == 1:
        r = r.squeeze(0)
    if tuple(r.shape) != tuple(got_pv.shape):
        rep["verdict"] = "SHAPE MISMATCH -- geometry wrong, not resampling"
        if verbose:
            print(f"[verify] {ad.name}: FAIL  ref={rep['ref_shape']} got={rep['got_shape']}")
        return False, rep

    d = (r - got_pv).abs()
    rep["max_abs_err"] = float(d.max())
    rep["mean_abs_err"] = float(d.mean())
    ok = rep["max_abs_err"] <= atol

    if "image_grid_thw" in ref and "image_grid_thw" in got:
        m = bool(torch.equal(ref["image_grid_thw"].cpu().long(),
                             got["image_grid_thw"].cpu().long()))
        rep["image_grid_thw_match"] = m
        ok = ok and m

    rep["verdict"] = "PASS" if ok else "FAIL"
    if verbose:
        print(f"[verify] {ad.name} on {rep['processor']}: {rep['verdict']}  "
              f"shape={rep['got_shape']}  max_err={rep['max_abs_err']:.2e}  "
              f"mean_err={rep['mean_abs_err']:.2e}")
    return ok, rep


def check_gradients(processor, image, verbose: bool = True) -> bool:
    """Confirm gradients actually reach image space. A build() that
    silently detaches would make every pixel attack a no-op."""
    import numpy as np
    ad = get_adapter(processor)
    arr = (torch.from_numpy(np.array(image.convert("RGB")))
           .permute(2, 0, 1).float() / 255.0).requires_grad_(True)
    plan = ad.plan(arr.shape[1], arr.shape[2])
    ad.build(arr, plan)["pixel_values"].sum().backward()
    g = arr.grad
    ok = bool(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0)
    if verbose:
        print(f"[grad] {ad.name}: {'PASS' if ok else 'FAIL'}  "
              f"grad_absmax={0.0 if g is None else float(g.abs().max()):.3e}")
    return ok


if __name__ == "__main__":
    import argparse, sys
    from PIL import Image
    from transformers import AutoProcessor

    ap = argparse.ArgumentParser(description="Gate for the geometry adapters.")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--atol", type=float, default=1e-3)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model_id, trust_remote_code=True)
    img = Image.open(a.image)
    ok, rep = verify_adapter(proc, img, atol=a.atol)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    gok = check_gradients(proc, img)
    sys.exit(0 if (ok and gok) else 1)
