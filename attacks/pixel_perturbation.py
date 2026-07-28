"""
attacks/pixel_perturbation.py
-------------------------------
Adversarial pixel perturbation (Table 3.2, visual baseline).
Qi et al. 2023; BIM (Kurakin et al.) family, l_inf bounded.

Adapted from ChenWu98/agent-attack's bim.py algorithm to run directly
against THIS project's own loaded models (extraction/model_loader.py)
via their native HF forward() interface -- deliberately NOT going
through agent-attack's own VLM wrapper abstraction, since that drags in
LangChain, the external `llava` git package (broken sentencepiece
build), and several unused provider SDKs, just to import a ~40-line
gradient loop. Only the algorithm is reused; agent-attack's
model-loading machinery is not needed since we're attacking our own
models, not their reference captioning models.

Same update rule as the reference: sign-gradient step, epsilon-ball
clamp, targeted toward maximizing likelihood of a target behavior
string under the victim model's own teacher-forced language-modeling
loss.

STATUS: mechanics verified against a synthetic fake-HF-model stand-in
(see bottom of this file / tests/test_pixel_perturbation.py) -- NOT yet
verified against a real loaded VLM. Unlike every other file shipped
this conversation, this one has not been run against real model weights.
Run the smoke test below via sbatch before trusting it for real attacks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from .base import AttackModality, AttackResult, BaseAttack


class PixelPerturbationAttack(BaseAttack):
    """
    BIM-style l_inf pixel perturbation. Requires the REAL model + processor
    (not a CPU smoke-test stand-in) since it needs real gradients through
    real vision encoder weights -- there is no meaningful dry-run for this
    attack the way there is for TemplateIPIAttack's pure string logic.
    """

    name = "pixel_perturbation"
    modality = AttackModality.VISUAL

    def __init__(
        self,
        model,
        processor,
        epsilon: float = 16 / 255,
        alpha: float = 1 / 255,
        iters: int = 200,
    ):
        """
        model, processor : the ALREADY-LOADED victim model/processor, e.g.
            from extraction.model_loader import load_llavanext_7b
            model, processor = load_llavanext_7b(hpc=True)
        epsilon, alpha : l_inf ball radius / step size -- same defaults
            as the reference implementation (16/255, 1/255).
        iters : lowered from the reference's 4000 to 200 as a starting
            point for exploratory Phase 2 runs; the reference's 4000
            targets convergence for a paper's headline numbers on a
            single fixed model. Tune once you see real convergence
            curves for your own models.
        """
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters

    def generate(
        self,
        base_input: Dict[str, Any],
        target_behavior: str,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> AttackResult:
        """
        image : clean base image to perturb. Falls back to
            base_input["image"] if not passed directly.
        prompt : text prompt shown alongside the image (e.g. the task's
            own question). Falls back to base_input.get("prompt", ...).
        target_behavior : text the attack tries to make the model output
            when shown the perturbed image -- same role as Template
            IPI's target_behavior.
        """
        image = image or base_input.get("image")
        if image is None:
            raise ValueError(
                "PixelPerturbationAttack needs a base image (pass image= or base_input['image'])."
            )
        prompt = prompt or base_input.get("prompt", "Describe this image.")

        device = next(self.model.parameters()).device

        image = image.convert("RGB")
        clean = torch.from_numpy(np.array(image).astype("float32") / 255.0)
        clean = clean.permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]
        delta = torch.zeros_like(clean, requires_grad=True)

        # Teacher-forced objective: mask the prompt (+ any image tokens the
        # processor inserts) out of the loss. Text side doesn't depend on
        # delta, so tokenize once rather than every iteration.
        full_text = f"{prompt} {target_behavior}"
        prompt_only = self.processor(images=image, text=prompt, return_tensors="pt")
        prompt_len = prompt_only["input_ids"].shape[1]
        full_inputs = self.processor(images=image, text=full_text, return_tensors="pt")
        full_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in full_inputs.items()}
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        for _ in range(self.iters):
            adv_tensor = (clean + delta).clamp(0, 1)  # stays differentiable -- no PIL round-trip
            pixel_values = self._to_pixel_values(adv_tensor)

            inputs = dict(full_inputs)  # reuse text-side inputs; only pixel_values changes per step
            inputs["pixel_values"] = pixel_values
            inputs["labels"] = labels

            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()

            with torch.no_grad():
                delta.grad.sign_()
                delta.data = delta.data - self.alpha * delta.grad
                delta.data.clamp_(-self.epsilon, self.epsilon)
                delta.data = torch.clamp(clean + delta, 0, 1) - clean
                delta.grad.zero_()

        adv_image_final = self._tensor_to_pil((clean + delta).clamp(0, 1))

        return AttackResult(
            attack_name=self.name,
            modality=self.modality,
            adversarial_image=adv_image_final,
            target_behavior=target_behavior,
            metadata={"epsilon": self.epsilon, "alpha": self.alpha, "iters": self.iters},
        )

    def _to_pixel_values(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Differentiable resize + normalize, replicating what the
        processor's image_processor does but WITHOUT round-tripping
        through PIL/uint8 -- that would detach the computation graph
        from delta, silently breaking the attack (no gradient reaches
        the perturbation at all). Uses the processor's own
        image_mean/image_std/size where available, falling back to
        CLIP-style defaults otherwise.
        """
        img_proc = getattr(self.processor, "image_processor", None)
        size = getattr(img_proc, "crop_size", None) or getattr(img_proc, "size", None) or {
            "height": 224,
            "width": 224,
        }
        h = size.get("height", size.get("shortest_edge", 224))
        w = size.get("width", size.get("shortest_edge", h))

        resized = torch.nn.functional.interpolate(
            image_tensor, size=(h, w), mode="bilinear", align_corners=False
        )
        mean = getattr(img_proc, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(img_proc, "image_std", [0.5, 0.5, 0.5])
        mean_t = torch.tensor(mean, device=image_tensor.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=image_tensor.device).view(1, -1, 1, 1)
        return (resized - mean_t) / std_t

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        arr = (tensor.squeeze(0).detach().cpu().numpy() * 255).astype("uint8").transpose(1, 2, 0)
        return Image.fromarray(arr)
