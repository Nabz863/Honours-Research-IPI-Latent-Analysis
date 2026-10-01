"""
attacks/shipped_transfer.py
---------------------------
The VWA-Adv paper's OWN adversarial images, used as transfer attacks.

agent-attack ships, per task directory, the images its attacks produced:
  bim_caption_attack_image.png   the captioner attack (PGD on LLaVA-1.5-7B,
                                 eps 16/256, toward the task's adversarial
                                 caption)
  clip_attack_attack_image.png   the CLIP-ensemble attack aimed at
                                 GPT-4V (other files target GPT-4o, Claude 3
                                 Opus and Gemini 1.5 Pro)
With self-captioning, OUR model captions these images in the episode, so
each is a transfer attack from the paper's source model to ours -- the
paper-comparable condition, at no optimisation cost. The paper's stored
attack CAPTIONS (LLaVA-1.5's outputs) are recorded in metadata for
reference but are not injected: in this configuration the caption comes
from the agent's own captioner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PIL import Image

from .base import AttackModality, AttackResult, BaseAttack


class ShippedImageAttack(BaseAttack):
    modality = AttackModality.VISUAL

    def __init__(self, name: str, prefix: str):
        self.name = name
        self.prefix = prefix

    def generate(self, base_input: Dict[str, Any], target_behavior: str, **kwargs) -> AttackResult:
        adv_dir = base_input.get("adv_dir")
        if adv_dir is None:
            raise ValueError(f"{self.name} needs base_input['adv_dir'] (the agent-attack task directory)")
        adv_dir = Path(adv_dir)
        img_path = adv_dir / f"{self.prefix}_attack_image.png"
        if not img_path.exists():
            # "not shipped" is a per-task data condition, recognised by the
            # runner as benign so a run of missing files cannot trip its
            # consecutive-failure abort (review 3, #8).
            raise FileNotFoundError(f"transfer image not shipped for this task: {img_path}")
        img = Image.open(img_path).convert("RGB")
        cap_path = adv_dir / f"{self.prefix}_attack_caption.txt"
        paper_caption = cap_path.read_text().strip() if cap_path.exists() else None
        return AttackResult(
            attack_name=self.name,
            modality=self.modality,
            adversarial_image=img,
            adversarial_text=None,          # the self-captioner supplies the caption
            target_behavior=target_behavior,
            metadata={"source_file": img_path.name, "image_size": img.size,
                      "paper_caption_llava15": paper_caption,
                      "transfer_from": "llava-v1.5-7b" if self.prefix == "bim_caption" else "clip_ensemble"},
        )
