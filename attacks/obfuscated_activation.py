"""
attacks/obfuscated_activation.py
-----------------------------------
Obfuscated activation attack (Table 3.2, adaptive baseline). Jain et al.
2023 baseline-defenses framing; directly operationalizes this project's
own RQ4 ("do latent signatures remain detectable when an adaptive
adversary optimises to mimic clean-condition geometry?") as an attack,
not only an evaluation question.

Extends GCGAttack's discrete suffix optimization (see gcg.py for the
base mechanics) with a SECOND loss term: alongside the usual
target-behavior cross-entropy, penalize the adversarial suffix's induced
activation at a chosen layer for deviating from a supplied clean-
condition reference activation (e.g. a real Phase 1 baseline sample).
The adversarial objective is explicit: succeed behaviourally (push the
model toward target_behavior) while ALSO leaving a clean-looking
geometric signature at the targeted layer -- testing whether the
latent-space detection this thesis proposes can be defeated by an
adversary that specifically knows to target it.

IMPORTANT: this does NOT reuse extraction.hooks.ActivationExtractor for
EITHER the gradient step or candidate evaluation -- both use their own
lightweight, single-layer local hooks instead. For the gradient step,
ActivationExtractor's hook deliberately .detach()es and moves
activations to CPU (correct for its actual job -- post-hoc readout for
saving/analysis, see hooks.py's own _make_hook), which would silently
break backprop through the geometry term if reused there. For candidate
evaluation, the reason is performance, discovered on real hardware, not
correctness: a real GPU run (job 23421) measured this attack running
~6.6x slower per task than GCGAttack at IDENTICAL num_steps/
candidates_per_step settings (353s vs 53s), traced to an earlier version
paying ActivationExtractor's full register()/remove()/clear() cost --
hooks on all of extractor.target_layers, not just the one this attack
needs, plus a synchronous gc.collect() inside clear() -- on every one of
hundreds of candidates per task. The `extractor` constructor argument is
still required and still meaningful: it validates token_aggregation and
supplies a sensible default target_layer, keeping this attack's geometry
comparison anchored to the same layer convention the rest of the
project's analysis uses -- it's just no longer used for the expensive
per-candidate hook registration itself.

STATUS: mechanics verified against a fake HF-convention text model (see
tests/test_obfuscated_activation.py) -- gradients flow through BOTH loss
terms, geometry_weight=0.0 is confirmed to produce IDENTICAL gradients to
plain GCGAttack (a true off-switch, not just a small effect), and a large
geometry_weight is confirmed to measurably reduce the final cosine
distance to the clean reference versus geometry_weight=0, at some cost to
the behavioral loss -- i.e. the actual tradeoff this attack exists to
study is confirmed to be real, not just plumbed through. NOT yet run
against real model weights or a real Phase 1 baseline activation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import AttackResult
from .gcg import GCGAttack


class ObfuscatedActivationAttack(GCGAttack):
    name = "obfuscated_activation"
    # Modality stays TEXTUAL (Table 3.2 lists this under Adaptive as a
    # category of evaluation, but the mechanism itself operates purely on
    # text tokens, same as GCGAttack) -- inherited from GCGAttack.

    def __init__(
        self,
        model,
        processor,
        extractor,
        num_steps: int = 30,
        suffix_length: int = 20,
        topk: int = 64,
        candidates_per_step: int = 32,
        init_suffix_token: str = "x",
        geometry_weight: float = 1.0,
        target_layer: Optional[int] = None,
    ):
        """
        extractor : an already-constructed extraction.hooks.
            ActivationExtractor for `model`, with token_aggregation="last"
            (required -- asserted below -- to match this attack's own
            local hook convention of reading the last-token activation).
            Reused for candidate evaluation only; see module docstring
            for why not for the gradient step.
        geometry_weight : mixing weight between the two loss terms:
            total = behavioral_ce_loss + geometry_weight * (1 - cosine_sim).
            0.0 reduces to plain GCG (confirmed identical in tests, not
            just "small"). Larger values push harder toward clean-looking
            geometry at the cost of behavioral success -- this tradeoff
            is the actual scientific object of interest for RQ4, not
            simply a knob to tune away.
        target_layer : which layer index (from extractor.target_layers)
            the geometry term is computed against. Defaults to the first
            target layer if not given.
        """
        super().__init__(
            model, processor,
            num_steps=num_steps, suffix_length=suffix_length, topk=topk,
            candidates_per_step=candidates_per_step, init_suffix_token=init_suffix_token,
        )
        if extractor.token_aggregation != "last":
            raise ValueError(
                f"ObfuscatedActivationAttack requires token_aggregation='last' "
                f"(got {extractor.token_aggregation!r}) -- its own local hook "
                "reads the last-token activation to match."
            )
        self.extractor = extractor
        self.geometry_weight = geometry_weight
        self.target_layer = target_layer if target_layer is not None else extractor.target_layers[0]
        if self.target_layer not in extractor.target_layers:
            raise ValueError(
                f"target_layer={self.target_layer} is not among the extractor's own "
                f"target_layers={extractor.target_layers}."
            )

    def generate(
        self,
        base_input: Dict[str, Any],
        target_behavior: str,
        clean_reference_activation: torch.Tensor,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> AttackResult:
        """
        clean_reference_activation : torch.Tensor, shape [hidden_dim] --
            the clean-condition activation this attack optimizes the
            suffix to resemble at self.target_layer. In real runs this
            should be a genuine Phase 1 baseline sample (e.g. loaded from
            baselines/<model>_clean/activations/sample_XXXX.pt), not a
            synthetic placeholder -- the entire point is mimicking REAL
            clean-condition geometry, not an arbitrary target.
        """
        tokenizer = self._tokenizer()
        embed_layer = self._embedding_matrix()
        embed_weight = embed_layer.weight
        device = embed_weight.device
        vocab_size = embed_weight.shape[0]

        clean_reference_activation = clean_reference_activation.to(device=device, dtype=embed_weight.dtype)

        prompt = prompt or base_input.get("prompt", "Follow the instructions in this content.")
        injected_field = base_input.get("injectable_field", "page_text")
        clean_text = base_input.get(injected_field, "")

        prefix_text = f"{prompt}\n\n{clean_text}\n\n"
        prefix_ids = torch.tensor(tokenizer.encode(prefix_text), device=device, dtype=torch.long)
        target_ids = torch.tensor(tokenizer.encode(target_behavior), device=device, dtype=torch.long)

        seed_ids = tokenizer.encode(self.init_suffix_token)
        seed_id = seed_ids[-1] if seed_ids else 0

        # Same fix as gcg.py: the candidate mask alone doesn't protect an
        # untouched seed position -- validate it explicitly, same
        # rationale as gcg.py's docstring (real crash was exactly this).
        safe_mask = self._safe_token_mask(tokenizer, vocab_size, device)
        if not bool(safe_mask[seed_id]):
            fallback_id = int(torch.nonzero(safe_mask, as_tuple=False)[0].item())
            import warnings

            warnings.warn(
                f"init_suffix_token={self.init_suffix_token!r} encodes to token id {seed_id}, "
                f"which is NOT in the safe-character set for this tokenizer. Falling back to "
                f"token id {fallback_id} as the seed instead."
            )
            seed_id = fallback_id

        suffix_ids = torch.full((self.suffix_length,), seed_id, device=device, dtype=torch.long)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        combined_losses = ce_losses = geom_losses = None
        best = 0
        for _ in range(self.num_steps):
            grad = self._combined_gradient(embed_layer, prefix_ids, suffix_ids, target_ids, clean_reference_activation)
            # Same fix as gcg.py: restrict candidates to safe_chars-only
            # tokens, inherited from GCGAttack. This class doesn't call
            # super().generate() (needs its own combined-loss loop), so
            # the mask has to be applied here explicitly too, not just in
            # the parent class -- real GPU run crashed here for exactly
            # this reason before the fix (see gcg.py's docstring).
            # safe_mask computed once above, reused every step.
            masked_neg_grad = (-grad).masked_fill(~safe_mask.unsqueeze(0), float("-inf"))
            topk_ids = masked_neg_grad.topk(min(self.topk, vocab_size), dim=-1).indices

            n_cand = self.candidates_per_step
            candidates = suffix_ids.unsqueeze(0).repeat(n_cand, 1)
            positions = torch.randint(0, self.suffix_length, (n_cand,), device=device)
            choices = torch.randint(0, topk_ids.shape[1], (n_cand,), device=device)
            for i in range(n_cand):
                candidates[i, positions[i]] = topk_ids[positions[i], choices[i]]

            combined_losses, ce_losses, geom_losses = self._evaluate_candidates_combined(
                embed_layer, prefix_ids, candidates, target_ids, clean_reference_activation
            )
            best = int(combined_losses.argmin())
            suffix_ids = candidates[best]

        adv_suffix_text = tokenizer.decode(suffix_ids.tolist())
        adversarial_text = f"{clean_text} {adv_suffix_text}"

        return AttackResult(
            attack_name=self.name,
            modality=self.modality,
            adversarial_text=adversarial_text,
            target_behavior=target_behavior,
            metadata={
                "num_steps": self.num_steps,
                "suffix_length": self.suffix_length,
                "topk": self.topk,
                "geometry_weight": self.geometry_weight,
                "target_layer": self.target_layer,
                "final_combined_loss": float(combined_losses[best]) if combined_losses is not None else None,
                "final_behavioral_ce_loss": float(ce_losses[best]) if ce_losses is not None else None,
                "final_geometry_distance": float(geom_losses[best]) if geom_losses is not None else None,
            },
        )

    def _local_activation_capture(self):
        """
        Registers a NON-detaching forward hook on the model's own layer
        module at self.target_layer -- same layer-list resolution
        extraction.hooks.py uses internally, but deliberately not going
        through ActivationExtractor itself (see module docstring).
        Returns (handle, holder); holder['activation'] is populated
        (still attached to the live autograd graph) after the next
        forward pass. Caller must call handle.remove().
        """
        from extraction.hooks import _get_layer_list

        layer_list = _get_layer_list(self.model)
        layer = layer_list[self.target_layer]
        holder: Dict[str, torch.Tensor] = {}

        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            holder["activation"] = hidden[0, -1]  # last token, batch 0 -- same convention as
            # ActivationExtractor's 'last' aggregation mode, but deliberately NOT detached.

        handle = layer.register_forward_hook(hook)
        return handle, holder

    def _combined_gradient(
        self, embed_layer, prefix_ids, suffix_ids, target_ids, clean_reference_activation
    ) -> torch.Tensor:
        embed_weight = embed_layer.weight
        vocab_size = embed_weight.shape[0]

        one_hot = torch.zeros(len(suffix_ids), vocab_size, device=suffix_ids.device, dtype=embed_weight.dtype)
        one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        suffix_embeds = one_hot @ embed_weight

        prefix_embeds = embed_layer(prefix_ids)
        target_embeds = embed_layer(target_ids)
        inputs_embeds = torch.cat([prefix_embeds, suffix_embeds, target_embeds], dim=0).unsqueeze(0)

        labels = torch.cat([
            torch.full((len(prefix_ids) + len(suffix_ids),), -100, device=suffix_ids.device, dtype=torch.long),
            target_ids,
        ]).unsqueeze(0)

        handle, holder = self._local_activation_capture()
        try:
            outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
        finally:
            handle.remove()

        ce_loss = outputs.loss
        activation = holder.get("activation")
        if activation is None:
            raise RuntimeError(
                f"target_layer={self.target_layer}'s hook never fired during the forward pass -- "
                "check that this layer index is actually reachable for this model/input."
            )
        geometry_loss = 1.0 - F.cosine_similarity(
            activation.unsqueeze(0), clean_reference_activation.unsqueeze(0)
        ).squeeze(0)

        combined = ce_loss + self.geometry_weight * geometry_loss
        combined.backward()

        grad = one_hot.grad.clone()
        one_hot.grad = None
        return grad

    def _evaluate_candidates_combined(
        self, embed_layer, prefix_ids, candidates, target_ids, clean_reference_activation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Real GPU run (job 23421) measured this method making
        obfuscated_activation ~6.6x slower per task than GCGAttack at
        IDENTICAL num_steps/candidates_per_step (353s vs 53s), despite
        the same core optimization loop. Traced to the previous
        implementation wrapping every one of the candidates_per_step *
        num_steps candidate evaluations (960 at default settings) in
        self.extractor's full register()/remove()/clear() cycle --
        registering hooks on all of extractor.target_layers (12 for
        LLaVA-NeXT's default heuristic) when only ONE layer
        (self.target_layer) is actually needed here, plus a synchronous
        gc.collect() inside clear() on every single call. Registering one
        lightweight hook ONCE for the whole candidate batch (same pattern
        already used for the gradient step's _local_activation_capture)
        reads the exact same layer via the same _get_layer_list()
        resolution ActivationExtractor uses internally -- this is a
        performance fix, not a change in what gets measured.
        """
        from extraction.hooks import _get_layer_list

        device = candidates.device
        n = candidates.shape[0]
        combined_losses = torch.empty(n, device=device)
        ce_losses = torch.empty(n, device=device)
        geom_losses = torch.empty(n, device=device)

        prefix_embeds = embed_layer(prefix_ids)
        target_embeds = embed_layer(target_ids)
        prefix_len = len(prefix_ids)
        suffix_len = candidates.shape[1]

        layer_list = _get_layer_list(self.model)
        layer = layer_list[self.target_layer]
        holder: Dict[str, torch.Tensor] = {}

        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            holder["activation"] = hidden[0, -1].detach()

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                for i in range(n):
                    suffix_embeds = embed_layer(candidates[i])
                    inputs_embeds = torch.cat([prefix_embeds, suffix_embeds, target_embeds], dim=0).unsqueeze(0)
                    labels = torch.cat([
                        torch.full((prefix_len + suffix_len,), -100, device=device, dtype=torch.long),
                        target_ids,
                    ]).unsqueeze(0)

                    outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
                    activation = holder["activation"].to(device=device, dtype=clean_reference_activation.dtype)

                    ce_losses[i] = outputs.loss
                    geom_losses[i] = 1.0 - F.cosine_similarity(
                        activation.unsqueeze(0), clean_reference_activation.unsqueeze(0)
                    ).squeeze(0)
                    combined_losses[i] = ce_losses[i] + self.geometry_weight * geom_losses[i]
        finally:
            handle.remove()

        return combined_losses, ce_losses, geom_losses
