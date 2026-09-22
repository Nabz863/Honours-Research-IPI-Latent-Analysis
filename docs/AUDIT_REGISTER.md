# Code Audit Register

**Purpose.** One row per script audited: what was found, what was changed,
and whether any collected data has to be regenerated as a result.

This is the working record for the correctness audit that gates reporting
results. It is separate from the two fault documents — those cover faults
found *during* experimentation; this covers faults found by *reading the
code afterwards*, which is a different and more systematic pass.

**Standard applied to each file.** Not "does it run" but:

1. Does it do what its docstring and the write-up claim?
2. What is ported from published work, and what differs?
3. What is verified, and what is asserted?
4. Could it produce plausible-looking wrong numbers?

Question 4 is the one that matters. Every fault found during
experimentation produced a complete run and reasonable-looking output.

---

## Status

| # | File | Audited | Findings | Re-run needed |
|---|------|---------|----------|---------------|
| 1 | `extraction/hooks.py` | yes | 3 | no |
| 2 | `analysis/geometry.py` | yes | 3 | no (verified by timestamp) |
| 3 | `analysis/stats.py` | yes | 3 | no |
| 4 | `analysis/rq2_geometry.py` | yes | 4 | **partial — see 4.1** |
| — | remaining ~26 files | no | — | — |

---

## 1. `extraction/hooks.py`

Forward-hook infrastructure. Every activation in the project passes
through it. Original code, nothing ported.

**1.1 False capability claim in the docstring.** The header listed
"modality-aware" capture — recording whether an activation came from a
text-only, image-only or mixed forward pass — as a design goal. No such
code exists anywhere in the file. A Phase 1 requirement that was never
implemented and never removed from the documentation.
*Fixed: claim deleted.* No result affected, but the file is headed for
public release and the claim would have been read as a feature.

**1.2 Auto-discovery could silently hook the vision tower.**
`_discover_layer_list` returns the longest homogeneous `nn.ModuleList`.
A VLM's vision encoder is also a homogeneous stack. On any model whose
vision tower is deeper than its language decoder this selects the wrong
stack, producing activations at the vision hidden size with no error
raised. It never happened — Gemma passed through this path and its
48-layer decoder is deeper than its vision stack — but nothing prevented
it.
*Fixed: guard added cross-checking the discovered length against
`config.text_config.num_hidden_layers`, raising on mismatch.*

**1.3 Layer-band constants undocumented.** Capture uses the top 50–90%
of the stack sampled at 8 points. Three constants, no citation, never
tuned against project data.
*Fixed: caveat added to the docstring stating they were chosen a priori.*
Separately verified empirically: the attack ordering holds at **14 of 14**
captured layers, so conclusions are invariant to the choice within the
band. That check is the defence, not the constants.

**Re-run: none.**

---

## 2. `analysis/geometry.py`

Per-layer displacement, cosine, probe, PCA. Produces every RQ1 magnitude.
Original code.

**2.1 Ignored `defense_name` — silent overwrite.** Episodes were keyed
on `(results_dir, suite, task_id)` with conditions keyed on
`attack_name`. A defended-attacked row carries the *same* `attack_name`
as an undefended one, so in any directory containing both, whichever
loaded last silently replaced the other. This is the same fault that was
found and fixed in `stats.py` earlier; it was never propagated here.
*Fixed: undefended rows only by default, `--include_defended` to
override.*

**2.2 Default `turn_strategy` produced zeros.** Default was `first`, and
the docstring recommended it as primary. On AgentDojo, turn 0 is
pre-injection — clean and attacked prompts are byte-identical — so the
default yields exactly 0.000 displacement. Every actual result uses
`mean`.
*Fixed: default changed to `mean`.*

**2.3 Probe accuracy not independently interpretable.** 4096 features
against ~200 samples. The number is only meaningful alongside the
permutation null in `stats.py`.
*Not a code change — a reporting constraint. Accuracy is never cited
without its permutation p-value.*

**Re-run: none.** All three live geometry summaries were checked against
the modification times of the defended tensors in their source
directories; each predates the first defended tensor, so none was
contaminated. Regenerated anyway into `*_clean` directories for a
consistent set, and to add InternVL and LLaVA at v4.

---

## 3. `analysis/stats.py`

Reachability, Wilcoxon, permutation probe, Fisher exact. Produces every
p-value reported. Original code.

**3.1 Correct where it counts.** The defence filter is present. Wilcoxon
over a t-test is justified from the measured skew. Wilson intervals are
right for proportions near 0 and 1. Holm includes the monotonicity
enforcement that naive implementations omit. The permutation null
shuffles labels *within group*, preserving the grouped-CV structure
under H0 — this is the correct null and is what makes the probe
accuracies interpretable.

**3.2 Probe pairs the same task across classes.** For each episode the
clean vector is class 0 and the attacked vector class 1, sharing a group
id. `GroupKFold` keeps them in the same fold, so there is no train/test
leakage — but the two vectors differ only by the injection, and a linear
model at 4096 dimensions can exploit that pairing. The permutation null
controls for it, since the null sees the identical structure.
*Accepted with a reporting constraint: the p-value is interpretable, the
raw accuracy overstates separability.*

**3.3 No guard on `--layer`.** Default 31, valid for the three 32-layer
models. On Gemma's 48-layer stack layer 31 exists but sits mid-stack, so
a run without `--layer 42` produces a valid-looking number measured at
the wrong depth.
*Fixed: warning when the requested layer is not the deepest captured,
naming the available band.*

**3.4 Fisher exact is undefined on VWA-Adv.** Reachability is 100% there,
so the contingency table has two empty cells and every test returns
`p=1, OR=None`. Structurally undefined, not a null result.
*Not a code change — those p-values are not cited.*

**Re-run: none.**

---

## 4. `analysis/rq2_geometry.py`

Restoration-vs-suppression decomposition. Produces the "displacement
retained" figures and the whole of RQ2's answer. Original code, written
most recently.

**4.1 The significance test compared the wrong quantities.** The
hypothesis is that off-axis movement is no larger than the defence's own
*off-axis* footprint. The Wilcoxon compared `orth` — the off-axis
component of the defended displacement — against `footprint`, the
**full** norm of the defence's own displacement including its on-axis
part. The correct quantity, `footprint_orth`, was computed and printed
but never tested.
*Fixed: the test now uses `footprint_orth` on both sides.*
**Re-run required** for every RQ2 output, to regenerate the p-values.
The medians are unaffected. Empirically the correction is expected to be
small, because the measured `cos(footprint, attack axis)` is 0.01–0.08 —
the footprint is nearly orthogonal to the attack axis already — but the
published test must be the one the hypothesis describes.

**4.2 Medians reported without dispersion.** `along_frac` medians are
the headline RQ2 figures. A median alone cannot distinguish "most tasks
retain ~57% of the displacement" from "half fully restored, half
untouched" — materially different claims.
*Fixed: quartiles and the fraction of tasks below 0.2 / above 0.8 added.*
**Re-run required** to populate them.

**4.3 `n_tasks` mislabelled.** Printed as "tasks with all four states";
counts tasks with *any* state.
*Fixed: relabelled.*

**4.4 Verdict thresholds arbitrary.** The RESTORATION / SUPPRESSION /
PARTIAL labels turn on cut-offs of 1.3 and 0.4, undocumented and
untuned.
*Fixed: comment stating they are descriptive labels chosen a priori, and
that the write-up should cite the measured quantities rather than the
labels.*

**An earlier self-correction worth recording.** A hypothesis was raised
that `orth/footprint ≈ 1.00` might be an artefact: if the defence's own
displacement shared a direction with the attack axis, projecting that
axis out would strip part of the footprint too, deflating the ratio by
construction. Diagnostics were added to test it, and it was **disproved**
— `cos(footprint, attack) = 0.07–0.08`, so the footprint is essentially
orthogonal to the attack axis and the ratio is real. The diagnostics
were retained.

**Re-run: yes.** All RQ2 outputs — four AgentDojo defences plus five
VWA-Adv attacks — for the corrected test and the dispersion figures.
CPU only, minutes per run.

---

## Cross-cutting observations

**The same fault class recurs across files.** A fault found and fixed in
one analysis script was not propagated to another doing the same join
(2.1 versus 3.1). Worth a deliberate pass asking, for each fixed fault,
"where else does this pattern appear".

**Every fault found so far is silent.** None raised an exception. All
four files ran to completion and produced plausible output both before
and after correction. This is the argument for auditing by reading
rather than by testing.

**No collected data has been invalidated by the audit.** One fix (4.1)
requires re-deriving p-values from existing activations; nothing requires
re-running a model.

---

## Remaining

Roughly 26 files unaudited. Priority order, highest risk first:

1. `analysis/strict_asr.py` and `analysis/ua_review_sheet.py` — they
   author evaluators, so a fault changes what "success" means
2. `environments/agentdojo_llm.py` and `environments/vwa_adv_llm.py` —
   every episode passes through them
3. `attacks/*.py` — nine files; the pixel-space three carry the
   preprocessing adapters
4. `defenses/*.py` — seven files
5. `experiments/run_*.py` — the matrix runners
6. SLURM scripts — lower risk, but they encode the cluster-specific
   knowledge anyone reusing this repository would need
