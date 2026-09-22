"""
experiments/adapter_logit_gate.py
---------------------------------
The gate that actually decides whether a geometry adapter is usable.

WHY PIXEL EQUALITY IS THE WRONG TEST
    Gradient-based image attacks cannot backprop through the real
    preprocessing path (PIL -> uint8 -> resize -> normalise), so they
    optimise through a differentiable surrogate and then save a real PNG
    that goes through the REAL processor at evaluation time. Some gap
    between surrogate and real pipeline is inherent to the method.

    So the question is not "are the pixel_values bit-identical" -- they
    never will be. It is "does the model behave the same on both". If
    the model's next-token distribution is materially unchanged, the
    surrogate carries useful gradient signal. If it is not, no pixel
    tolerance rescues it.

WHAT THIS MEASURES
    1. top-1 agreement       -- does the model predict the same token
    2. top-5 overlap         -- is the ranking stable, not just the peak
    3. KL(real || adapter)   -- distributional distance, in nats
    4. max abs logit delta   -- worst-case single-logit disagreement
    5. rank of the real top-1 under the adapter -- catches a near-miss
       that top-1 agreement alone would report as a hard failure

    Reported alongside a CONTROL: the same comparison against the image
    perturbed by a random +/- eps sign pattern (the same budget an attack
    is allowed). If adapter-vs-real disagreement is much smaller than
    real-vs-perturbed, the surrogate error is small relative to the
    signal the attack is trying to create, which is the thing that
    matters. Without that control a KL number is uninterpretable.

Usage
    python3 experiments/adapter_logit_gate.py --model qwen3vl \\
        --image /path/to/victim_image.png
    python3 experiments/adapter_logit_gate.py --model llava --n_images 5 \\
        --image_glob '/datasets/.../agent_adv/reddit_*/victim_image.png'
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks.geometry_adapters import get_adapter  # noqa: E402
from extraction.model_loader import (  # noqa: E402
    load_llavanext_7b, load_internvl_8b, load_qwen3vl_8b, load_gemma4_12b,
)

LOADERS = {
    "llava": load_llavanext_7b,
    "internvl": load_internvl_8b,
    "qwen3vl": load_qwen3vl_8b,
    "gemma4": load_gemma4_12b,
}


def _last_logits(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float().cpu()


def _prompt(processor, text="Describe this image."):
    """Build a prompt with exactly one image placeholder, the way the
    processor itself expects -- a hardcoded '<image>' raises on InternVL."""
    try:
        return processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": text}]}],
            add_generation_prompt=True, tokenize=False)
    except Exception:
        tok = getattr(processor, "image_token", None) or "<image>"
        return f"{tok}\n{text}"


def compare(model, processor, image, eps: float = 8 / 255):
    from PIL import Image

    device = next(model.parameters()).device
    text = _prompt(processor)

    ref_inputs = processor(images=image, text=text, return_tensors="pt")
    ref_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in ref_inputs.items()}
    lo_real = _last_logits(model, ref_inputs)

    # --- adapter path ----------------------------------------------------
    ad = get_adapter(processor)
    arr = (torch.from_numpy(np.array(image.convert("RGB")))
           .permute(2, 0, 1).float() / 255.0)
    plan = ad.plan(arr.shape[1], arr.shape[2])
    kw = ad.build(arr.to(device), plan)
    ad_inputs = dict(ref_inputs)
    # Gemma's build() returns num_soft_tokens_per_image as a plain
    # Python list alongside its tensors, so .to(device) cannot be
    # applied blindly.
    ad_inputs.update({k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in kw.items()})
    lo_adapt = _last_logits(model, ad_inputs)

    # --- control: real pipeline on an eps-perturbed image ----------------
    g = torch.Generator().manual_seed(0)
    sign = (torch.randint(0, 2, arr.shape, generator=g).float() * 2 - 1)
    pert = (arr + eps * sign).clamp(0, 1)
    pert_img = Image.fromarray(
        (pert.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
    ctl_inputs = processor(images=pert_img, text=text, return_tensors="pt")
    ctl_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in ctl_inputs.items()}
    lo_pert = _last_logits(model, ctl_inputs)

    def stats(a, b):
        """a = reference, b = comparison."""
        pa, pb = a.log_softmax(-1), b.log_softmax(-1)
        kl = float(F.kl_div(pb, pa, log_target=True, reduction="sum"))
        t5a = set(a.topk(5).indices.tolist())
        t5b = set(b.topk(5).indices.tolist())
        rank = int((b > b[a.argmax()]).sum())      # rank of a's top-1 under b
        return {
            "top1_same": bool(a.argmax() == b.argmax()),
            "top5_overlap": len(t5a & t5b),
            "kl_nats": kl,
            "max_logit_delta": float((a - b).abs().max()),
            "rank_of_real_top1": rank,
        }

    return {
        "adapter": ad.name,
        "plan_hw": [plan.target_h, plan.target_w],
        "adapter_vs_real": stats(lo_real, lo_adapt),
        "eps_perturbed_vs_real": stats(lo_real, lo_pert),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LOADERS))
    ap.add_argument("--image")
    ap.add_argument("--image_glob")
    ap.add_argument("--n_images", type=int, default=1)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--out")
    a = ap.parse_args()

    from PIL import Image

    paths = ([a.image] if a.image
             else sorted(globmod.glob(a.image_glob))[:a.n_images])
    if not paths:
        sys.exit("no images matched")

    model, processor = LOADERS[a.model](hpc=True)
    model.eval()

    results = []
    for p in paths:
        r = compare(model, processor, Image.open(p), eps=a.eps)
        r["image"] = Path(p).parent.name
        results.append(r)
        av, cv = r["adapter_vs_real"], r["eps_perturbed_vs_real"]
        print(f"\n{r['image']}  [{r['adapter']}]  plan={r['plan_hw']}")
        print(f"  adapter vs real : top1={av['top1_same']}  "
              f"top5={av['top5_overlap']}/5  KL={av['kl_nats']:.4f}  "
              f"dlogit={av['max_logit_delta']:.3f}  rank={av['rank_of_real_top1']}")
        print(f"  eps-pert vs real: top1={cv['top1_same']}  "
              f"top5={cv['top5_overlap']}/5  KL={cv['kl_nats']:.4f}  "
              f"dlogit={cv['max_logit_delta']:.3f}  rank={cv['rank_of_real_top1']}")

    n = len(results)
    t1 = sum(r["adapter_vs_real"]["top1_same"] for r in results)
    kl_a = float(np.mean([r["adapter_vs_real"]["kl_nats"] for r in results]))
    kl_c = float(np.mean([r["eps_perturbed_vs_real"]["kl_nats"] for r in results]))
    ratio = kl_a / kl_c if kl_c > 0 else float("inf")

    print("\n" + "=" * 66)
    print(f"top-1 agreement (adapter vs real) : {t1}/{n}")
    print(f"mean KL  adapter vs real          : {kl_a:.4f} nats")
    print(f"mean KL  eps-perturbed vs real    : {kl_c:.4f} nats   (control)")
    print(f"ratio adapter/control             : {ratio:.3f}")
    if ratio < 0.25 and t1 == n:
        v = "USABLE -- surrogate error well below the attack's own footprint"
    elif ratio < 1.0:
        v = "MARGINAL -- surrogate error same order as the attack signal"
    else:
        v = "NOT USABLE -- surrogate perturbs the model more than the attack does"
    print(f"verdict: {v}")
    print("=" * 66)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"model": a.model, "eps": a.eps, "results": results,
                   "mean_kl_adapter": kl_a, "mean_kl_control": kl_c,
                   "ratio": ratio, "verdict": v},
                  open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
