"""
tests/test_pixel_perturbation.py
-----------------------------------
Mechanics-only smoke test for PixelPerturbationAttack: verifies gradients
flow, the loop completes, and a valid image is produced -- using a fake
HF-convention model (pixel_values/input_ids/labels -> object with .loss),
NOT a real VLM. This does NOT verify correctness against real model
weights or real HF processor behavior -- run the real thing via sbatch
before trusting this for actual attacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks.pixel_perturbation import PixelPerturbationAttack


class _FakeOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class _FakeHFVLM(nn.Module):
    """Toy model matching the HF calling convention: forward(pixel_values, input_ids, labels) -> .loss"""

    def __init__(self, vocab_size: int = 50, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.vision_proj = nn.Linear(3, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, pixel_values=None, input_ids=None, labels=None, **kwargs):
        img_feat = pixel_values.mean(dim=[2, 3])  # [1, 3]
        img_embed = self.vision_proj(img_feat)  # [1, dim]
        tok_embed = self.embed(input_ids)  # [1, seq, dim]
        combined = tok_embed + img_embed.unsqueeze(1)
        logits = self.lm_head(combined)  # [1, seq, vocab]
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return _FakeOutput(loss, logits)


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        # Deterministic fake tokenization: one token per word, capped at vocab size
        ids = [min(hash(w) % 40, 39) + 1 for w in text.split()]

        class _Out:
            pass

        out = _Out()
        out.input_ids = torch.tensor([ids])
        return out


class _FakeImageProcessor:
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    size = {"height": 4, "width": 4}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeImageProcessor()

    def __call__(self, images=None, text=None, return_tensors="pt"):
        # Toy "pixel_values": just the image resized to a fixed tiny tensor
        import numpy as np

        arr = np.array(images.convert("RGB").resize((4, 4))).astype("float32") / 255.0
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        input_ids = self.tokenizer(text).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def test_pixel_perturbation_mechanics():
    model = _FakeHFVLM()
    processor = _FakeProcessor()
    attack = PixelPerturbationAttack(model, processor, epsilon=16 / 255, alpha=1 / 255, iters=5)

    base_image = Image.new("RGB", (16, 16), color=(120, 120, 120))
    base_input = {"image": base_image, "prompt": "What is this?"}

    result = attack.generate(base_input, target_behavior="a cat")

    assert result.attack_name == "pixel_perturbation"
    assert result.adversarial_image is not None
    assert result.adversarial_image.size == base_image.size
    assert result.metadata["iters"] == 5

    # Confirm the perturbation actually changed the image (attack did something)
    import numpy as np

    orig_arr = np.array(base_image).astype("int16")
    adv_arr = np.array(result.adversarial_image).astype("int16")
    assert not np.array_equal(orig_arr, adv_arr), "Adversarial image identical to clean -- no perturbation applied"

    # Confirm perturbation stayed within the epsilon ball (in pixel units, ~16/255 * 255 = 16)
    max_diff = np.abs(orig_arr - adv_arr).max()
    assert max_diff <= 17, f"Perturbation exceeded epsilon ball: max diff {max_diff}"


if __name__ == "__main__":
    tests = [test_pixel_perturbation_mechanics]
  = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)
