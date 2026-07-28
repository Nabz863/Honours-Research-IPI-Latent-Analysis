# Scoping Decisions

## Decision 1: Reddit-first VWA-Adv environment setup

**Date:** Phase 2, environment setup stage

**Decision:** Stand up only the Reddit (Social Forum) Docker container first,
rather than all four VWA sites (Reddit, Shopping, Classifieds, Wikipedia)
simultaneously.

**Rationale:**
- All attacks in the Phase 2 attack matrix (Table 3.2 of the proposal) are
  content-injected (template IPI, GCG suffixes, typographic attacks,
  adversarial pixel perturbation, adversarial patches, steganographic
  embedding, CrossInject, COMET) — none depend on site-specific
  functionality. The same attack generation code is reusable across any
  site; only the delivery URL changes.
- The proposal's research questions (RQ1-RQ5) claim generalisation across
  *model architectures* (LLaVA-NeXT, Qwen2-VL, InternVL), not across
  *websites*. VWA-Adv's multi-site design serves behavioural task
  diversity (Wu et al. 2025), which is orthogonal to the mechanistic
  latent-geometry questions this project investigates.
- Reddit is the smallest download and simplest setup (2 commands, no
  Magento reindexing), making it the fastest path to a working,
  validated end-to-end pipeline (attack injection -> agent processes
  page -> hooks capture activations -> defence applied -> geometry
  compared).

**Plan:** Validate the full pipeline against Reddit first. Add Shopping,
Classifieds, and Wikipedia afterward — this is expected to be a
low-risk repetition of the same Docker setup pattern, not new
engineering. If time constraints (per Section 4.2.1 contingency
priorities) prevent adding all sites, this will be stated explicitly
as a scope limitation in the writeup, not left implicit.

---

## Decision 2: Prompting strategy — use each environment's native/recommended approach

**Date:** Phase 2, experimental design stage

**Decision:** Rather than forcing a single prompting strategy (e.g. SoM)
across all models and environments, use the native/recommended prompting
strategy for each (environment, model) pairing — e.g. SoM for VWA-Adv
with LLaVA-NeXT, Qwen2-VL's native grounding for Qwen2-VL tasks.

**Rationale:**
- Each model is evaluated under the conditions it was designed and
  optimised for, which is the fairest basis for cross-architecture
  comparison.
- The critical experimental control is that prompting strategy is held
  constant *within* a given (environment, task, model) triplet across
  the clean condition, attack condition, and defence condition. This
  ensures that geometric deltas (activation delta, cosine similarity
  to clean manifold, etc.) are attributable to the attack/defence
  manipulation, not to prompting variation.
- Cross-architecture comparisons ask whether attacks produce similar
  *patterns of deviation from each model's own clean baseline* — not
  whether absolute activation values are comparable across architectures.
  This framing is valid regardless of prompting strategy differences
  between models.

**Implication for writeup:** Prompting strategy is a controlled variable
within each architecture condition, and a documented varying factor
across architectures. This should be stated explicitly in the methodology
chapter.

---

## Decision 3: InternVL2-8B deferred — transformers 5.x compatibility issue

**Date:** Phase 1, baseline collection stage

**Issue:** InternVL2-8B's custom `InternVLChatModel` class (loaded via
`trust_remote_code=True`) does not implement `all_tied_weights_keys`,
which transformers 5.x requires during model loading. This causes an
`AttributeError` regardless of `device_map` setting.

**Decision:** InternVL2-8B baseline collection deferred. LLaVA-NeXT 7B
and Qwen2-VL 7B baselines are complete (100 samples each).

**Resolution path:** Either (a) create a separate venv with
transformers==4.40.x for InternVL-only jobs, (b) switch to InternVL3
which has native transformers support, or (c) use an alternative
cross-architecture model (e.g. PaliGemma 2). To be resolved before
Phase 3.

---

## Experimental Design Reference

**Independent variables:**
- Attack modality: clean (control) / textual / visual / cross-modal
- Attack type: template IPI, GCG suffix, typographic, adversarial pixel,
  adversarial patch, steganographic, CrossInject, COMET, obfuscated
  activation, MUZZLE (see Table 3.2)
- Defence applied: none / spotlighting / Gaussian blur / StruQ / LAT /
  circuit breaking / TaskTracker / ICON / AgentSentry / SALLIE / ECSO
  (see Table 3.3)
- Model architecture: LLaVA-NeXT 7B / Qwen2-VL 7B / InternVL 8B

**Dependent variables:**
- Geometric: activation delta ||Δh||, cosine similarity to clean manifold,
  attention entropy, deflection angle, linear probe accuracy, persistent
  homology (Betti numbers)
- Behavioural: ASR (Attack Success Rate), UA (Utility Under Attack)

**Control condition:**
Clean-condition inference — same model, same task, same site, same
prompting strategy, no adversarial content. Phase 1 baseline activations
serve as the control group. All geometric metrics are defined as deviation
from this baseline.

**Unit of observation:**
One agent episode (a single VWA-Adv task run) producing one set of
activation tensors per targeted layer. Sample size per condition is
governed by the combinatorial reduction strategy in Section 3.4.2 of
the proposal: only the highest-ASR attacks (by activation delta) are
carried forward to Phase 3, managing experiment count within compute
budget.

**Statistical note:**
A formal statistical test plan (test choice, sample size justification,
correction for multiple comparisons across layers/architectures) is
deferred to the thesis methodology chapter once compute constraints and
per-condition variance are empirically established in early Phase 2 runs.
