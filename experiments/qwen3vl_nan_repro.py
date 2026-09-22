"""
experiments/qwen3vl_nan_repro.py
------------------------------------
Fast, minimal repro attempt for the real-run NaN/inf-in-probability-tensor
crash hit during AgentDojo's clean baseline pass (job 25580, task #1,
torch.multinomial inside _sample()). Deliberately NOT replaying a full
AgentDojo episode's context (tool definitions, environment state, multi-
turn history) -- this is a small synthetic prompt + tiny image + short
max_new_tokens, sized to have a real chance of fitting on bigbatch's 24GB
cards despite Qwen3-VL's known tightness there (model_loader.py's own
notes: "only ~2.5GB headroom short during generation" for real AgentDojo-
scale contexts). Targets bigbatch specifically for fast queue turnaround
instead of waiting behind biggpu's matrix jobs for what should be a
two-minute check.

Uses do_sample=True explicitly (not the model's inherited default) so
this genuinely exercises the same torch.multinomial code path that
crashed for real, regardless of what CUDA_LAUNCH_BLOCKING reveals about
where the NaN actually originates.

Three possible outcomes, all informative:
  - Crashes with the same NaN/inf assertion -> confirms this is a
    model/attention/dtype-level bug, not specific to that one AgentDojo
    task's particular context -- matches the pattern found in multiple
    open GitHub issues across the whole Qwen-VL family.
  - Succeeds cleanly -> suggests something about the real task's longer/
    more complex context matters, not a bare model+sampling issue --
    points investigation toward context length or specific input content
    instead.
  - OOMs even at this tiny scale -> a new, worse finding: Qwen3-VL may
    not fit AT ALL on bigbatch's 24GB, even for trivial generation, not
    just tight on headroom for real episodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from extraction.model_loader import load_qwen3vl_8b


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device visible.", file=sys.stderr)
        sys.exit(1)

    print("Loading Qwen3-VL-8B-Instruct (hpc=True)...")
    model, processor = load_qwen3vl_8b(hpc=True)
    print(f"Model loaded on {next(model.parameters()).device}, dtype={next(model.parameters()).dtype}.")

    image = Image.new("RGB", (336, 336), color=(128, 128, 128))
    prompt = "Describe this image in one short sentence."

    try:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"apply_chat_template failed ({type(e).__name__}: {e}) -- falling back to a plain string prompt.")
        text = prompt

    inputs = processor(images=image, text=text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(f"Input shapes: { {k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')} }")
    print("Running generate() with do_sample=True (matching the real crash's code path) ...")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=True)

    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)
    print("=== SUCCESS -- no NaN/inf crash on this minimal input ===")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
