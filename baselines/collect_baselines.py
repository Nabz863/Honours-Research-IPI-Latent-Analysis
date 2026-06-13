"""
baselines/collect_baselines.py
-------------------------------
Phase 1, Part 2: Collect clean-condition latent space baselines.

For each (model, task_input) pair, run a forward pass under clean conditions
and save the activation tensors. These become the reference geometry that
all attack and defence phases are measured against.

Outputs (per model)
-------------------
  baselines/<model_name>/
      activations/
          sample_0000.pt   # {layer_idx: tensor [hidden_dim]}
          sample_0001.pt
          ...
      metadata.json        # config, layer indices, aggregation mode, etc.

Usage
-----
  python baselines/collect_baselines.py \
      --model llava7b \
      --n_samples 100 \
      --output_dir baselines/llava7b \
      --hpc

  # Dry-run on CPU with tiny synthetic inputs (for unit testing):
  python baselines/collect_baselines.py --model llava7b --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.hooks import ActivationExtractor, _resolve_target_layers, _get_layer_list
from extraction.model_loader import (
    load_llavanext_7b,
    load_llavanext_13b,
    load_qwen2vl_7b,
    load_internvl_8b,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic input generators (for dry-run / unit tests)
# ---------------------------------------------------------------------------

def make_synthetic_text_input(processor, prompt: str = "Describe the image.") -> dict:
    """Minimal text-only input for architecture validation."""
    # Use a plain prompt without image token for text-only baseline
    return processor(text=prompt, return_tensors="pt")


def make_synthetic_multimodal_input(processor, prompt: str = "What is in this image?") -> dict:
    """
    Create a minimal multimodal input with a synthetic blank image.
    Used for dry-run testing — not a meaningful semantic input.
    """
    # 224x224 blank white image
    image = Image.new("RGB", (224, 224), color=(255, 255, 255))

    # LlavaNextProcessor expects a specific conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        # LLaVA-style processors
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt")
    except (AttributeError, TypeError):
        # Qwen2-VL / generic fallback
        inputs = processor(images=image, text=prompt, return_tensors="pt")

    return inputs


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

def collect_baselines(
    model: torch.nn.Module,
    processor,
    output_dir: Path,
    n_samples: int,
    target_layers: Optional[List[int]],
    token_aggregation: str = "last",
    modality: str = "multimodal",
    device: str = "cuda",
    dry_run: bool = False,
) -> None:
    """
    Run n_samples clean-condition forward passes and save activations.

    Parameters
    ----------
    modality : str
        'text', 'image', or 'multimodal' — recorded in metadata, used
        to label samples for downstream modality-separated analysis.
    dry_run : bool
        If True, generate synthetic inputs rather than loading real data.
        Useful for validating the pipeline before committing HPC time.
    """
    acts_dir = output_dir / "activations"
    acts_dir.mkdir(parents=True, exist_ok=True)

    # Resolve and record target layers
    resolved_layers = _resolve_target_layers(
        model, target_layers if target_layers else None
    )
    n_layers_total = len(_get_layer_list(model))

    logger.info(
        "Collecting %d clean-condition samples | modality=%s | layers=%s",
        n_samples, modality, resolved_layers,
    )

    extractor = ActivationExtractor(
        model,
        target_layers=resolved_layers,
        token_aggregation=token_aggregation,
    )

    timings = []
    for i in range(n_samples):
        t0 = time.perf_counter()

        # ---- Build input ------------------------------------------------
        if dry_run or modality == "text":
            inputs = make_synthetic_text_input(processor)
        else:
            inputs = make_synthetic_multimodal_input(processor)

        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # ---- Forward pass with hooks ------------------------------------
        with extractor:
            with torch.no_grad():
                try:
                    _ = model(**inputs)
                except Exception as e:
                    logger.warning("Forward pass %d failed: %s", i, e)
                    extractor.clear()
                    continue

            acts = extractor.get_activations()

        # ---- Save -------------------------------------------------------
        save_path = acts_dir / f"sample_{i:04d}.pt"
        torch.save(acts, save_path)
        extractor.clear()

        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "  [%d/%d] saved %s (%.2fs, avg %.2fs)",
                i + 1, n_samples, save_path.name,
                elapsed, sum(timings) / len(timings),
            )

    # ---- Write metadata -------------------------------------------------
    meta = {
        "model_id": getattr(model.config, "_name_or_path", "unknown"),
        "n_samples": n_samples,
        "n_saved": len(list(acts_dir.glob("sample_*.pt"))),
        "target_layers": resolved_layers,
        "n_layers_total": n_layers_total,
        "token_aggregation": token_aggregation,
        "modality": modality,
        "dtype": str(next(model.parameters()).dtype),
        "device": device,
        "dry_run": dry_run,
        "avg_forward_pass_s": sum(timings) / len(timings) if timings else None,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Baseline collection complete. Metadata: %s", output_dir / "metadata.json")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "llava7b": load_llavanext_7b,
    "llava13b": load_llavanext_13b,
    "qwen2vl7b": load_qwen2vl_7b,
    "internvl8b": load_internvl_8b,
}


def main():
    parser = argparse.ArgumentParser(
        description="Collect clean-condition VLM activation baselines (Phase 1)."
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="llava7b",
        help="Which VLM to load.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of clean-condition samples to collect.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory. Defaults to baselines/<model>.",
    )
    parser.add_argument(
        "--target_layers", type=int, nargs="+", default=None,
        help="Explicit layer indices to hook. Defaults to late-middle heuristic.",
    )
    parser.add_argument(
        "--token_aggregation", choices=["last", "mean", "full"], default="last",
        help="How to aggregate across sequence tokens.",
    )
    parser.add_argument(
        "--modality", choices=["text", "image", "multimodal"], default="multimodal",
        help="Input modality for this baseline collection run.",
    )
    parser.add_argument(
        "--hpc", action="store_true",
        help="Load model for HPC (fp16, device_map=auto). Off = CPU, fp32.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Use synthetic inputs. Validates pipeline without real data.",
    )
    args = parser.parse_args()

    # Determine device
    device = "cuda" if args.hpc and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("Running on CPU — activations will be in fp32. Not suitable for large models.")

    # Load model
    logger.info("Loading model: %s (hpc=%s)", args.model, args.hpc)
    loader_fn = MODEL_REGISTRY[args.model]
    model, processor = loader_fn(hpc=args.hpc)

    # Output directory
    out = Path(args.output_dir) if args.output_dir else Path("baselines") / args.model
    out.mkdir(parents=True, exist_ok=True)

    collect_baselines(
        model=model,
        processor=processor,
        output_dir=out,
        n_samples=args.n_samples,
        target_layers=args.target_layers,
        token_aggregation=args.token_aggregation,
        modality=args.modality,
        device=device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
