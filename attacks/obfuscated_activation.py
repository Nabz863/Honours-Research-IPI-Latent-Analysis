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
    uses_task_context = True
    # Modality stays TEXTUAL (Table 3.2 lists this under Adaptive as a
    # category of evaluation, but the mechanism itself operates purely on
    # text tokens, same as GCGAttack) -- inherited from GCGAttack.

    def __init__(
        self,
        model,
        processor,
        extractor,
        num_steps: int = 50,
        suffix_length: int = 20,
        topk: int = 128,
        candidates_per_step: int = 64,
        init_suffix_token: str = "x",
        num_restarts: int = 3,
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
            num_restarts=num_restarts,
        )
        if extractor.token_aggregation != "last":
            raise ValueError(
                f"ObfuscatedActivationAttack requires token_aggregation='last' "
                f"(got {extractor.token_aggregation!r}) -- its own local hook "
                "reads the last-token activation to match."
            )
        self.extractor = extractor
        self.geometry_weight = geometry_weight
        if target_layer is not None:
            self.target_layer = target_layer
        else:
            # Default to the DEEPEST captured layer -- the one the analysis
            # reads (audit 12.5c). The old default target_layers[0] optimised
            # the geometry term at the shallowest captured layer, a different
            # layer from the one RQ1/RQ4 report. Final runs should pass
            # target_layer explicitly; this is only the fallback.
            self.target_layer = extractor.target_layers[-1]
            import warnings
            warnings.warn(
                "ObfuscatedActivationAttack: target_layer not given; defaulting to "
                f"deepest captured layer {self.target_layer}. Pass it explicitly for "
                "final runs so the geometry term matches the analysis layer."
            )
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
        clean_transcript: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AttackResult:
        """
        clean_reference_activation : torch.Tensor, shape [hidden_dim] --
            the clean-condition activation this attack optimizes the
            suffix to resemble at self.target_layer. In real runs this
            should be a genuine Phase 1 baseline sample (e.g. loaded from
            baselines/<model>_clean/activations/sample_XXXX.pt), not a
            synthetic placeholder -- the entire point is mimicking REAL
            clean-condition geometry, not an arbitrary target. It MUST be
            captured at self.target_layer and at the decision position -- a
            reference from a different layer silently compares across layers
            (audit 12.5b). The caller is responsible for this; the tensor
            carries no layer tag to check.
        """
        tokenizer = self._tokenizer()
        embed_layer = self._embedding_matrix()
        embed_weight = embed_layer.weight
        device = embed_weight.device
        vocab_size = embed_weight.shape[0]

        clean_reference_activation = clean_reference_activation.to(device=device, dtype=embed_weight.dtype)

        prefix_ids, tail_ids, clean_text, used_real_context = self._context_segments(
            base_input, target_behavior, tokenizer, clean_transcript=clean_transcript
        )
        prefix_ids = prefix_ids.to(device=device, dtype=torch.long)
        tail_ids = tail_ids.to(device=device, dtype=torch.long)
        target_ids = torch.tensor(self._encode_without_specials(tokenizer, target_behavior), device=device, dtype=torch.long)

        seed_ids = tokenizer.encode(self.init_suffix_token)
        seed_id = seed_ids[-1] if seed_ids else 0

        # Same fix as gcg.py: the candidate mask alone doesn't protect an
        # untouched seed position -- validate it explicitly, same
        # rationale as gcg.py's docstring (real crash was exactly this).
        safe_mask = self._safe_token_mask(tokenizer, vocab_size, device)
        self._active_tokenizer = tokenizer
        if not bool(safe_mask[seed_id]):
            fallback_id = int(torch.nonzero(safe_mask, as_tuple=False)[0].item())
            import warnings

            warnings.warn(
                f"init_suffix_token={self.init_suffix_token!r} encodes to token id {seed_id}, "
                f"which is NOT in the safe-character set for this tokenizer. Falling back to "
                f"token id {fallback_id} as the seed instead."
            )
            seed_id = fallback_id
        seed_id = self._stable_seed_id(tokenizer, safe_mask, seed_id)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        overall_suffix = None
        overall_combined = float("inf")
        overall_ce = None
        overall_geom = None
        overall_restart = -1
        restart_losses = []

        for restart_idx in range(self.num_restarts):
            suffix_ids = torch.full((self.suffix_length,), seed_id, device=device, dtype=torch.long)
            best_suffix = suffix_ids.clone()
            best_combined = float("inf")
            best_ce = None
            best_geom = None

            for _ in range(self.num_steps):
                decision_position = len(prefix_ids) + len(suffix_ids) + len(tail_ids) - 1
                grad = self._combined_gradient(
                    embed_layer, prefix_ids, suffix_ids, target_ids, clean_reference_activation,
                    tail_ids=tail_ids, decision_position=decision_position,
                )
                masked_neg_grad = (-grad).masked_fill(~safe_mask.unsqueeze(0), float("-inf"))
                safe_count = int(safe_mask.sum().item())
                if safe_count == 0:
                    raise RuntimeError("No safe tokens remain for obfuscated-activation candidate selection")
                topk_ids = masked_neg_grad.topk(min(self.topk, safe_count), dim=-1).indices

                n_cand = self.candidates_per_step
                candidates = suffix_ids.unsqueeze(0).repeat(n_cand, 1)
                positions = torch.randint(0, self.suffix_length, (n_cand,), device=device)
                choices = torch.randint(0, topk_ids.shape[1], (n_cand,), device=device)
                for i in range(n_cand):
                    candidates[i, positions[i]] = topk_ids[positions[i], choices[i]]
                candidates[0] = suffix_ids

                combined_losses, ce_losses, geom_losses = self._evaluate_candidates_combined(
                    embed_layer, prefix_ids, candidates, target_ids, clean_reference_activation,
                    tail_ids=tail_ids, decision_position=decision_position,
                )
                valid_candidates = self._roundtrip_candidate_mask(candidates)
                if valid_candidates is not None:
                    combined_losses = combined_losses.masked_fill(~valid_candidates, float("inf"))
                best = int(combined_losses.argmin())
                step_combined = float(combined_losses[best])
                suffix_ids = candidates[best].clone()
                if step_combined < best_combined:
                    best_combined = step_combined
                    best_suffix = suffix_ids.clone()
                    best_ce = float(ce_losses[best])
                    best_geom = float(geom_losses[best])

            restart_losses.append(best_combined)
            if best_combined < overall_combined:
                overall_combined = best_combined
                overall_suffix = best_suffix.clone()
                overall_ce = best_ce
                overall_geom = best_geom
                overall_restart = restart_idx

        suffix_ids = overall_suffix
        adv_suffix_text = tokenizer.decode(suffix_ids.tolist())
        roundtrip_verified = self._roundtrip_safe_ids(tokenizer, suffix_ids.tolist())
        if not roundtrip_verified:
            raise RuntimeError("Selected obfuscated-activation suffix does not survive decode→encode round trip")
        adversarial_text = f"{target_behavior} {adv_suffix_text}"

        return AttackResult(
            attack_name=self.name,
            modality=self.modality,
            adversarial_text=adversarial_text,
            target_behavior=target_behavior,
            metadata={
                "num_steps": self.num_steps,
                "suffix_length": self.suffix_length,
                "topk": self.topk,
                "candidates_per_step": self.candidates_per_step,
                "num_restarts": self.num_restarts,
                "best_restart_idx": overall_restart,
                "restart_losses": restart_losses,
                "geometry_weight": self.geometry_weight,
                "target_layer": self.target_layer,
                "used_real_templated_context": used_real_context,
                "target_has_special_tokens": False,
                "suffix_roundtrip_verified": roundtrip_verified,
                "final_combined_loss": overall_combined if overall_ce is not None else None,
                "final_behavioral_ce_loss": overall_ce,
                "final_geometry_distance": overall_geom,
            },
        )

    def _local_activation_capture(self, position: int):
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
            holder["activation"] = hidden[0, position]  # decision-position token, batch 0 -- same convention as
            # ActivationExtractor's 'last' aggregation mode, but deliberately NOT detached.

        handle = layer.register_forward_hook(hook)
        return handle, holder

    def _combined_gradient(
        self, embed_layer, prefix_ids, suffix_ids, target_ids, clean_reference_activation,
        tail_ids=None, decision_position: Optional[int] = None,
    ) -> torch.Tensor:
        embed_weight = embed_layer.weight
        vocab_size = embed_weight.shape[0]

        one_hot = torch.zeros(len(suffix_ids), vocab_size, device=suffix_ids.device, dtype=embed_weight.dtype)
        one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        embed_scale = getattr(embed_layer, "embed_scale", 1.0)  # B5: see gcg.py
        suffix_embeds = (one_hot @ embed_weight) * embed_scale

        prefix_embeds = embed_layer(prefix_ids)
        target_embeds = embed_layer(target_ids)
        tail_ids = tail_ids if tail_ids is not None else prefix_ids.new_empty((0,))
        tail_embeds = embed_layer(tail_ids)
        inputs_embeds = torch.cat([prefix_embeds, suffix_embeds, tail_embeds, target_embeds], dim=0).unsqueeze(0)

        labels = torch.cat([
            torch.full((len(prefix_ids) + len(suffix_ids) + len(tail_ids),), -100, device=suffix_ids.device, dtype=torch.long),
            target_ids,
        ]).unsqueeze(0)

        if decision_position is None:
            decision_position = len(prefix_ids) + len(suffix_ids) + len(tail_ids) - 1
        handle, holder = self._local_activation_capture(decision_position)
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
        self, embed_layer, prefix_ids, candidates, target_ids, clean_reference_activation,
        tail_ids=None, decision_position: Optional[int] = None,
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
        tail_ids = tail_ids if tail_ids is not None else prefix_ids.new_empty((0,))
        tail_embeds = embed_layer(tail_ids)
        prefix_len = len(prefix_ids)
        suffix_len = candidates.shape[1]
        tail_len = len(tail_ids)
        if decision_position is None:
            decision_position = prefix_len + suffix_len + tail_len - 1

        layer_list = _get_layer_list(self.model)
        layer = layer_list[self.target_layer]
        holder: Dict[str, torch.Tensor] = {}

        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            holder["activation"] = hidden[0, decision_position].detach()

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                for i in range(n):
                    suffix_embeds = embed_layer(candidates[i])
                    inputs_embeds = torch.cat([prefix_embeds, suffix_embeds, tail_embeds, target_embeds], dim=0).unsqueeze(0)
                    labels = torch.cat([
                        torch.full((prefix_len + suffix_len + tail_len,), -100, device=device, dtype=torch.long),
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
