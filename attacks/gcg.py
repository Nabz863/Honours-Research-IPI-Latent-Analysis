"""
attacks/gcg.py
----------------
Greedy Coordinate Gradient adversarial suffix (Table 3.2, "strong
textual baseline"). Zou et al. 2023.

Unlike pixel_perturbation.py's continuous PGD-style optimization over
pixel values, GCG optimizes over a DISCRETE sequence of token ids -- the
gradient only tells you which single-token substitutions look locally
promising (via a one-hot relaxation), not a step to take directly. Each
iteration: compute the loss gradient w.r.t. a one-hot encoding of the
current suffix, take the top-k most loss-reducing replacement tokens per
suffix position, sample a batch of single-position substitution
candidates from those, evaluate each candidate's REAL (non-approximated)
loss with an actual forward pass, and keep the best one.

This is a TEXT-ONLY attack (Table 3.2 lists it as Textual/Gradient-based),
operating purely on the model's language-model input embeddings -- no
image/pixel_values needed, unlike pixel_perturbation.py. That also makes
it directly usable against AgentDojo's text-only suites, where
pixel_perturbation has no image to attack at all.

Same framing as template_ipi.py: the optimized suffix gets appended to
whatever text field the benchmark exposes as injectable (an AgentDojo
injection vector's default text, or a VWA-Adv page's text content) --
this module produces the suffix, the caller (attack matrix driver)
decides where it gets spliced in.

HYPERPARAMETERS, revised from this file's original smoke-test-scale
defaults (num_steps=30, topk=64, candidates_per_step=32): real AgentDojo
matrix data now exists at that config, and found exactly ONE confirmed
ASR=True episode across the whole run -- consistent with a search this
shallow, not proof the attack mechanism is broken. Reference
implementation uses ~500 steps; num_steps=50 with num_restarts=3 below
targets roughly 10x the old config's total compute (3 * 50 * 65 vs
30 * 33), not the reference's ~130x, chosen to be affordable against a
real end-of-September deadline. These are DELIBERATE, evidence-informed
defaults now, not placeholders -- if real timing data from a completed
run shows this doesn't fit the remaining budget, that's a real
constraint worth revisiting explicitly, not silently working around.

MULTI-RESTART: GCG's per-step candidate sampling is stochastic (random
position + random draw from the topk pool each step), so a single
trajectory can land in a bad local optimum purely by chance. Running
self.num_restarts independent trajectories from the same known-safe
starting point and keeping whichever converges to the lowest final loss
is standard practice in the GCG literature specifically because it
measurably improves success rate at similar total compute versus one
longer single trajectory -- not just "try harder," a real, established
technique for this exact discrete-optimization failure mode. Restarts
diverge from each other via the stochastic candidate sampling alone
(torch's global RNG state advances across calls, never reset between
restarts) -- they do NOT currently use a randomized starting suffix,
only the single validated safe seed_id repeated suffix_length times;
randomizing the start too is a possible further extension, not done
here since it would reopen the untouched-position safety question
(see init_suffix_token's docstring below) for every restart, not just
the one seed value already validated.

STATUS: mechanics verified against a fake HF-convention text model (see
tests/test_gcg.py) for the pre-restart version -- gradients flow through
the one-hot relaxation, the topk/candidate loop completes, the optimized
suffix's loss is confirmed lower than the untouched initial suffix's.
The restart wrapper itself has NOT yet been re-verified against real
model weights -- it's a straightforward loop around already-verified
logic, but "straightforward" isn't the same as "checked," per this
project's own established standard. Run a real smoketest before
trusting this for a full matrix run.
"""

from __future__ import annotations

import string
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import AttackModality, AttackResult, BaseAttack


class GCGAttack(BaseAttack):
    name = "gcg"
    modality = AttackModality.TEXTUAL
    uses_task_context = True

    def __init__(
        self,
        model,
        processor,
        num_steps: int = 50,
        suffix_length: int = 20,
        topk: int = 128,
        candidates_per_step: int = 64,
        init_suffix_token: str = "x",
        safe_chars: Optional[str] = None,
        num_restarts: int = 3,
    ):
        """
        model, processor : the ALREADY-LOADED victim model/processor, e.g.
            from extraction.model_loader import load_llavanext_7b
            model, processor = load_llavanext_7b(hpc=True)
        num_steps : optimization iterations PER RESTART. See module
            docstring for the revised default's reasoning and cost math.
        suffix_length : number of adversarial tokens appended.
        topk : candidate replacement tokens considered per suffix
            position, selected by steepest (most loss-reducing) one-hot
            gradient direction. Does not affect per-step compute cost
            (that's candidates_per_step) -- only search quality/diversity.
        candidates_per_step : how many single-position substitutions are
            actually evaluated with a real forward pass each step (out of
            the topk * suffix_length possible ones) -- this, together
            with num_steps and num_restarts, is what actually drives
            total compute cost.
        num_restarts : independent optimization trajectories run from the
            same safe starting point; the lowest-final-loss trajectory's
            suffix is what's actually returned. See module docstring.
        init_suffix_token : token repeated suffix_length times to seed the
            initial (pre-optimization) suffix, shared by every restart.
            Deliberately a plain letter ("x"), not "!" or similar -- see
            safe_chars below for why: a real GPU run against AgentDojo
            crashed with a YAML parse error because the untouched seed
            character ("!" in that run) is a YAML tag indicator, and
            AgentDojo's own environment-injection mechanism does a raw
            yaml.safe_load() on the spliced-in payload with no escaping.
            Some suffix positions plausibly never get touched by
            optimization (num_steps can be << suffix_length in
            expectation for a short run), so the seed character itself
            needs to be safe, not just the optimized substitutions.
        safe_chars : the set of characters candidate replacement tokens
            are allowed to decode to (see _safe_token_mask). Defaults to
            ASCII letters, digits, space, period, and comma -- restrictive
            on purpose. Real GCG follow-up work commonly constrains the
            search this way; it isn't only about AgentDojo's specific YAML
            quirk, raw unconstrained GCG output is known more generally to
            contain control characters and other structurally awkward
            content that can break whatever pipeline the payload lands in
            downstream. Pass a wider set if a specific benchmark's
            injection path is confirmed to safely escape special
            characters and the larger search space is worth it.
        """
        self.model = model
        self.processor = processor
        self.num_steps = num_steps
        self.suffix_length = suffix_length
        self.topk = topk
        self.candidates_per_step = candidates_per_step
        self.init_suffix_token = init_suffix_token
        self.num_restarts = num_restarts
        self.safe_chars = set(safe_chars) if safe_chars is not None else set(string.ascii_letters + string.digits + " .,")
        self._safe_token_mask_cache: Dict[int, torch.Tensor] = {}  # vocab_size -> mask, computed lazily

    def _tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def _safe_token_mask(self, tokenizer, vocab_size: int, device) -> torch.Tensor:
        """
        Boolean mask over the vocabulary: True for token ids whose
        decoded text consists ENTIRELY of characters in self.safe_chars.
        Computed once per (attack instance, vocab_size) and cached --
        this is an O(vocab_size) decode loop, a one-time cost worth
        paying once rather than per optimization step (and shared across
        every restart, not recomputed per restart -- it doesn't depend
        on anything restart-specific).
        """
        if vocab_size in self._safe_token_mask_cache:
            return self._safe_token_mask_cache[vocab_size].to(device)

        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in range(vocab_size):
            try:
                text = tokenizer.decode([token_id])
            except Exception:
                continue
            if text and all(c in self.safe_chars for c in text):
                try:
                    roundtrip = tokenizer.encode(text, add_special_tokens=False)
                except TypeError:
                    # Test/fallback tokenizers may not implement HF's kwarg;
                    # they cannot provide a meaningful round-trip guarantee.
                    mask[token_id] = True
                else:
                    if list(roundtrip) == [token_id]:
                        mask[token_id] = True

        if not mask.any():
            raise RuntimeError(
                "No vocabulary tokens decode entirely within safe_chars -- this tokenizer's "
                "encoding scheme (e.g. byte-pair pieces that rarely align to clean whole "
                "characters) may be incompatible with per-token safety filtering as implemented "
                "here. Widen safe_chars or investigate this tokenizer's token->text mapping."
            )

        self._safe_token_mask_cache[vocab_size] = mask
        return mask.to(device)

    def _embedding_matrix(self):
        """
        The model's input token embedding table. get_input_embeddings()
        is the standard HF accessor -- present on every CausalLM, and VLM
        wrappers delegate it to their language-model submodule -- so this
        works across LLaVA-NeXT/Qwen2-VL/InternVL without per-architecture
        branching, same principle as extraction/hooks.py's layer-list
        auto-discovery.
        """
        return self.model.get_input_embeddings()

    @staticmethod
    def _encode_without_specials(tokenizer, text: str) -> List[int]:
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            # Small fake tokenizers used by unit tests may not accept the kwarg.
            return list(tokenizer.encode(text))

    @staticmethod
    def _replace_first_text(value, needle: str, replacement: str):
        """Recursively replace the first occurrence in transcript message content."""
        if isinstance(value, str):
            if needle and needle in value:
                return value.replace(needle, replacement, 1), True
            return value, False
        if isinstance(value, list):
            out = []
            found = False
            for item in value:
                if found:
                    out.append(item)
                else:
                    item, found = GCGAttack._replace_first_text(item, needle, replacement)
                    out.append(item)
            return out, found
        if isinstance(value, dict):
            out = {}
            found = False
            for key, item in value.items():
                if found:
                    out[key] = item
                else:
                    item, found = GCGAttack._replace_first_text(item, needle, replacement)
                    out[key] = item
            return out, found
        return value, False

    def _supports_roundtrip_check(self, tokenizer) -> bool:
        try:
            tokenizer.encode("", add_special_tokens=False)
            return True
        except TypeError:
            return False

    def _roundtrip_safe_ids(self, tokenizer, token_ids) -> bool:
        if not self._supports_roundtrip_check(tokenizer):
            return True  # a non-HF test tokenizer cannot express this contract
        text = tokenizer.decode(list(token_ids))
        return self._encode_without_specials(tokenizer, text) == list(token_ids)

    def _stable_seed_id(self, tokenizer, safe_mask, preferred_id: int) -> int:
        """Find a safe token whose repeated decoded suffix preserves token IDs."""
        if not self._supports_roundtrip_check(tokenizer):
            return preferred_id
        candidates = [preferred_id] + [int(x) for x in torch.nonzero(safe_mask, as_tuple=False).flatten().tolist()
                                       if int(x) != preferred_id]
        for token_id in candidates[:512]:
            if self._roundtrip_safe_ids(tokenizer, [token_id] * self.suffix_length):
                return token_id
        raise RuntimeError(
            "Could not find a round-trip-stable repeated seed among the first 512 safe tokens; "
            "refusing to optimize token IDs that will change when the payload is decoded and delivered."
        )

    def _roundtrip_candidate_mask(self, candidates) -> Optional[torch.Tensor]:
        tokenizer = getattr(self, "_active_tokenizer", None)
        if tokenizer is None or not self._supports_roundtrip_check(tokenizer):
            return None
        valid = [self._roundtrip_safe_ids(tokenizer, row) for row in candidates.detach().cpu().tolist()]
        return torch.tensor(valid, dtype=torch.bool, device=candidates.device)

    def _context_segments(self, base_input, target_behavior, tokenizer, clean_transcript=None):
        """Return token ids on either side of the optimized suffix insertion point.

        For AgentDojo, replay the actual clean episode's latest transcript turn
        with the malicious instruction inserted where the injectable vector was
        observed, then use the victim processor's chat template. The suffix is
        optimized at that exact insertion point. For callers without a transcript
        (e.g. legacy text-only VWA invocations), use the supplied page text as the
        available context rather than inventing a system prompt or tool schema.
        """
        field = base_input.get("injectable_field", "page_text")
        clean_text = str(base_input.get(field, "") or "")
        marker = "ZXQGCSUFFIXMARKER7F3A"
        if clean_transcript and clean_text:
            messages_table = clean_transcript.get("messages", [])
            steps = clean_transcript.get("steps", [])
            # Prefer the latest transcript step containing this exact injection vector.
            for step in reversed(steps):
                ids = step.get("input") or step.get("raw_input") or []
                try:
                    conversation = [messages_table[int(i)] for i in ids]
                except (IndexError, TypeError, ValueError):
                    continue
                replaced, found = self._replace_first_text(
                    conversation, clean_text, f"{target_behavior} {marker}"
                )
                if not found:
                    continue
                rendered = self.processor.apply_chat_template(
                    replaced, add_generation_prompt=True, tokenize=False
                )
                if marker not in rendered:
                    continue
                left, right = rendered.split(marker, 1)
                return (torch.tensor(self._encode_without_specials(tokenizer, left), dtype=torch.long),
                        torch.tensor(self._encode_without_specials(tokenizer, right), dtype=torch.long),
                        clean_text, True)
            raise RuntimeError(
                "clean_transcript was supplied, but its recorded messages do not contain "
                "the injectable vector text; refusing to optimize GCG in a fabricated context."
            )

        # Fallback for callers that do not provide a clean AgentDojo transcript.
        left = f"{clean_text}\n\n{target_behavior} "
        right = "\n\n"
        return (torch.tensor(self._encode_without_specials(tokenizer, left), dtype=torch.long),
                torch.tensor(self._encode_without_specials(tokenizer, right), dtype=torch.long),
                clean_text, False)

    def generate(
        self,
        base_input: Dict[str, Any],
        target_behavior: str,
        prompt: Optional[str] = None,
        clean_transcript: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AttackResult:
        """
        prompt : task-facing text shown alongside the injectable content
            (e.g. the benign task instruction) -- included in the
            optimization context since that's what the victim model
            actually sees surrounding the injected field in practice.
            Falls back to base_input.get("prompt", ...).
        target_behavior : the adversary's goal text -- same role as
            TemplateIPIAttack's target_behavior. The suffix is optimized
            to make the model's continuation more likely to match this
            string under teacher forcing, exactly as pixel_perturbation.py
            optimizes pixels toward the same kind of teacher-forced
            objective, just over discrete tokens instead of continuous
            pixel values.
        """
        tokenizer = self._tokenizer()
        embed_layer = self._embedding_matrix()
        embed_weight = embed_layer.weight  # [vocab_size, hidden_dim]
        device = embed_weight.device
        vocab_size = embed_weight.shape[0]

        prefix_ids, tail_ids, clean_text, used_real_context = self._context_segments(
            base_input, target_behavior, tokenizer, clean_transcript=clean_transcript
        )
        prefix_ids = prefix_ids.to(device=device, dtype=torch.long)
        tail_ids = tail_ids.to(device=device, dtype=torch.long)
        target_ids = torch.tensor(self._encode_without_specials(tokenizer, target_behavior), device=device, dtype=torch.long)

        suffix_token_ids = tokenizer.encode(self.init_suffix_token)
        # Some tokenizers prepend a special token (e.g. BOS) even for a
        # single word -- take the last id as the real "content" token.
        seed_id = suffix_token_ids[-1] if suffix_token_ids else 0

        # The candidate mask below only protects tokens chosen DURING
        # optimization -- it does nothing for the untouched seed itself,
        # which is exactly what caused a real crash previously: some
        # suffix positions never get touched (num_steps can be <<
        # suffix_length in expectation), so if the seed token isn't safe,
        # neither is the final suffix, regardless of how well the mask
        # works elsewhere. init_suffix_token defaulting to a safe-looking
        # string ("x") is not a guarantee for every tokenizer -- check
        # explicitly rather than assume. Validated ONCE here, shared by
        # every restart (all restarts share the same seed_id).
        safe_mask = self._safe_token_mask(tokenizer, vocab_size, device)
        self._active_tokenizer = tokenizer
        if not bool(safe_mask[seed_id]):
            fallback_id = int(torch.nonzero(safe_mask, as_tuple=False)[0].item())
            import warnings

            warnings.warn(
                f"init_suffix_token={self.init_suffix_token!r} encodes to token id {seed_id}, "
                f"which is NOT in the safe-character set for this tokenizer. Falling back to "
                f"token id {fallback_id} as the seed instead -- pass a different init_suffix_token "
                "if this fallback is inappropriate for this model/tokenizer."
            )
            seed_id = fallback_id
        seed_id = self._stable_seed_id(tokenizer, safe_mask, seed_id)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        best_suffix_ids: Optional[torch.Tensor] = None
        best_loss = float("inf")
        best_restart_idx = -1
        restart_losses: List[float] = []

        for restart_idx in range(self.num_restarts):
            suffix_ids, final_loss = self._run_single_restart(
                embed_layer, prefix_ids, tail_ids, target_ids, safe_mask, seed_id, device
            )
            restart_losses.append(final_loss)
            if final_loss < best_loss:
                best_loss = final_loss
                best_suffix_ids = suffix_ids
                best_restart_idx = restart_idx

        adv_suffix_text = tokenizer.decode(best_suffix_ids.tolist())
        roundtrip_verified = self._roundtrip_safe_ids(tokenizer, best_suffix_ids.tolist())
        if not roundtrip_verified:
            raise RuntimeError("Selected GCG suffix does not survive decode→encode round trip")
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
                "best_restart_idx": best_restart_idx,
                "final_loss": best_loss,
                "restart_losses": restart_losses,
                "used_real_templated_context": used_real_context,
                "target_has_special_tokens": False,
                "suffix_roundtrip_verified": roundtrip_verified,
            },
        )

    def _run_single_restart(
        self,
        embed_layer,
        prefix_ids: torch.Tensor,
        tail_ids: torch.Tensor,
        target_ids: torch.Tensor,
        safe_mask: torch.Tensor,
        seed_id: int,
        device,
    ) -> Tuple[torch.Tensor, float]:
        """
        Runs ONE full GCG optimization trajectory (self.num_steps
        iterations) from the shared, already-validated-safe seed_id
        starting point. Returns (best_suffix_ids, final_loss) for THIS
        trajectory only -- generate() calls this self.num_restarts times
        and keeps whichever call returns the lowest final_loss.
        """
        suffix_ids = torch.full((self.suffix_length,), seed_id, device=device, dtype=torch.long)
        vocab_size = safe_mask.shape[0]

        losses = None
        best = 0
        best_suffix = suffix_ids.clone()
        best_loss = float(self._evaluate_candidates(
            embed_layer, prefix_ids, suffix_ids.unsqueeze(0), target_ids, tail_ids=tail_ids
        )[0])
        for _ in range(self.num_steps):
            grad = self._suffix_gradient(embed_layer, prefix_ids, suffix_ids, target_ids, tail_ids=tail_ids)

            # Steepest-descent candidates: topk tokens whose one-hot
            # direction MOST DECREASES loss at each position (smallest/
            # most-negative gradient), not the topk highest-gradient ones.
            # Restrict candidates to safe_chars-only tokens -- masked-out
            # tokens get -inf so they can never win topk, regardless of
            # how promising their raw gradient looked.
            masked_neg_grad = (-grad).masked_fill(~safe_mask.unsqueeze(0), float("-inf"))
            safe_count = int(safe_mask.sum().item())
            if safe_count == 0:
                raise RuntimeError("No safe tokens remain for GCG candidate selection")
            topk_ids = masked_neg_grad.topk(min(self.topk, safe_count), dim=-1).indices  # [suffix_length, <= topk]

            n_cand = self.candidates_per_step
            candidates = suffix_ids.unsqueeze(0).repeat(n_cand, 1)
            positions = torch.randint(0, self.suffix_length, (n_cand,), device=device)
            choices = torch.randint(0, topk_ids.shape[1], (n_cand,), device=device)
            for i in range(n_cand):
                candidates[i, positions[i]] = topk_ids[positions[i], choices[i]]
            candidates[0] = suffix_ids

            losses = self._evaluate_candidates(embed_layer, prefix_ids, candidates, target_ids, tail_ids=tail_ids)
            valid_candidates = self._roundtrip_candidate_mask(candidates)
            if valid_candidates is not None:
                losses = losses.masked_fill(~valid_candidates, float("inf"))
            best = int(losses.argmin())
            step_loss = float(losses[best])
            suffix_ids = candidates[best].clone()
            if step_loss < best_loss:
                best_loss = step_loss
                best_suffix = suffix_ids.clone()

        return best_suffix, best_loss

    def _suffix_gradient(self, embed_layer, prefix_ids, suffix_ids, target_ids, tail_ids=None) -> torch.Tensor:
        """
        Returns d(loss)/d(one_hot(suffix)) -- shape [suffix_length, vocab_size].

        Standard GCG relaxation: represent the suffix as one-hot vectors,
        multiply by the embedding matrix to get embeddings that stay
        differentiable w.r.t. the one-hot tensor, run an ordinary forward
        pass through inputs_embeds, backprop. The suffix is the ONLY thing
        requiring grad here -- prefix/target embeddings are looked up
        normally (plain integer index), since we never edit them.
        """
        embed_weight = embed_layer.weight
        vocab_size = embed_weight.shape[0]

        one_hot = torch.zeros(len(suffix_ids), vocab_size, device=suffix_ids.device, dtype=embed_weight.dtype)
        one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        # B5: Gemma 3's word-embedding module scales its forward output by
        # embed_scale (= hidden_size**0.5). embed_layer(prefix_ids) below goes
        # through that forward and IS scaled; one_hot @ weight hits the raw
        # weight and is NOT, leaving the suffix sqrt(hidden) smaller than the
        # tokens beside it and shrinking its gradient by the same factor.
        # _evaluate_candidates uses embed_layer(...), so it is already correct;
        # match it here. embed_scale is 1.0 (a no-op) on models without one.
        embed_scale = getattr(embed_layer, "embed_scale", 1.0)
        suffix_embeds = (one_hot @ embed_weight) * embed_scale  # [suffix_length, hidden_dim]

        prefix_embeds = embed_layer(prefix_ids)
        target_embeds = embed_layer(target_ids)
        tail_ids = tail_ids if tail_ids is not None else prefix_ids.new_empty((0,))
        tail_embeds = embed_layer(tail_ids)
        inputs_embeds = torch.cat([prefix_embeds, suffix_embeds, tail_embeds, target_embeds], dim=0).unsqueeze(0)

        labels = torch.cat([
            torch.full((len(prefix_ids) + len(suffix_ids) + len(tail_ids),), -100, device=suffix_ids.device, dtype=torch.long),
            target_ids,
        ]).unsqueeze(0)

        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
        outputs.loss.backward()

        grad = one_hot.grad.clone()
        one_hot.grad = None
        return grad

    def _evaluate_candidates(self, embed_layer, prefix_ids, candidates, target_ids, tail_ids=None) -> torch.Tensor:
        """
        Real (non-approximated) loss for each candidate suffix -- this is
        what actually decides which single-token substitution gets kept.
        The gradient step only narrows down which substitutions are worth
        evaluating in the first place; it's a first-order approximation,
        not the real objective.
        """
        device = candidates.device
        losses = torch.empty(candidates.shape[0], device=device)
        prefix_embeds = embed_layer(prefix_ids)
        target_embeds = embed_layer(target_ids)
        tail_ids = tail_ids if tail_ids is not None else prefix_ids.new_empty((0,))
        tail_embeds = embed_layer(tail_ids)
        prefix_len = len(prefix_ids)
        suffix_len = candidates.shape[1]
        tail_len = len(tail_ids)

        with torch.no_grad():
            for i in range(candidates.shape[0]):
                suffix_embeds = embed_layer(candidates[i])
                inputs_embeds = torch.cat([prefix_embeds, suffix_embeds, tail_embeds, target_embeds], dim=0).unsqueeze(0)
                labels = torch.cat([
                    torch.full((prefix_len + suffix_len + tail_len,), -100, device=device, dtype=torch.long),
                    target_ids,
                ]).unsqueeze(0)
                outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
                losses[i] = outputs.loss
        return losses
