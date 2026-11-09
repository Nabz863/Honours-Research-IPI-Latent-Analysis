# Indirect prompt injection in multimodal agents: benchmark and latent-space analysis

Code for the Wits Honours research report *Benchmarking Indirect Prompt
Injection in Multimodal Agentic Systems: Shared Injection Directions in the
Latent Space* (N. Mahomed, supervised by R. Klein, 2026).

Two open-weight vision-language model (VLM) agents, Gemma-4-12B-it and
Qwen3-VL-8B-Instruct, are run on **AgentDojo** (v1.1.2) and **VWA-Adv**
(Reddit) under ten textual, visual and cross-modal attacks and seven defences.
Every task is run clean, attacked, defended-clean and defended-attacked, and the
hidden state of the last prompt token is saved at every agent decision. The
analysis scripts turn the run directories into every table and figure in the
paper.

---

## Repository layout

| Path | Contents |
|---|---|
| `experiments/run_attack_matrix.py` | Runs the full matrix (clean, attacks, defences) on AgentDojo or VWA-Adv; resumable |
| `experiments/run_with_dual_judge.py` | Launcher for VWA-Adv runs; records the LLM-judge verdicts for the two judged tasks |
| `attacks/` | Template IPI, *important instructions*, typographic, steganographic, pixel (BIM), patch, transfer sets, CrossInject, COMET |
| `defenses/` | Spotlighting (delimiting, datamarking, encoding), Attention Tracker, ICON, ECSO, Gaussian blur |
| `environments/` | AgentDojo and VWA-Adv adapters, self-captioner, judge wrapper |
| `extraction/` | Model loaders and the activation hooks (`hooks.py`) |
| `analysis/` | Every analysis behind the paper (see [Reproducing the paper](#reproducing-the-paper)) |
| `strict_configs/` | VWA-Adv tasks whose evaluator passes any failed episode, re-specified to score the attacker's own target |
| `clean_configs/` | Re-authored benign evaluators so utility is scorable on more VWA-Adv tasks |
| `tests/` | CPU-only unit and fixture tests (`pytest`) |
| `docs/AUDIT_REGISTER.md` | Code audit: every fault found, its fix, and whether results were re-run |
| `SCOPING_DECISIONS.md` | Design decisions and their reasons |
| `hpc_*.sh`, `serve_postmill*.slurm` | Development-era cluster jobs, kept for the record; not needed to reproduce the paper |
| `archive/` | Code that was built and then dropped (not used in the paper) |

Run directories (`final_*`, `repro_*`), activations (`*.pt`) and
`analysis_out/` are not in git; they are large and archived separately. Other
tracked files not listed above (`baselines/`, `icon_trace_prober.py`, `patch_*.py`,
smoke-test outputs) are development artefacts. The GCG and obfuscated-activation
attacks in `attacks/` are implemented but not part of the paper.

---

## Setup

**Hardware used for the paper.** One NVIDIA RTX PRO 6000 (Blackwell, 96 GB) per
run job, 96 GB host RAM. All analysis is CPU-only (8 cores, 64 GB RAM is
enough). Runs are resumable: if a job hits its time limit, rerun the same
command; completed episodes are skipped and errored ones retried.

**Python environment.** Python 3.14; `requirements.txt` pins every package to
the version used for the paper (CUDA 13 wheels for PyTorch), including AgentDojo
at the exact commit used.
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium      # browser for VWA-Adv
```

**Model weights** (Hugging Face): `google/gemma-4-12B-it`,
`Qwen/Qwen3-VL-8B-Instruct`, and `meta-llama/Llama-3.1-8B-Instruct` (judge used
by the VWA-Adv launcher). Set `HF_HOME` to the cache that holds them. Models run
in bfloat16 without quantisation.

**AgentDojo.** Installed by `requirements.txt`; the benchmark version is pinned
with
```bash
export AGENTDOJO_BENCHMARK_VERSION=v1.1.2
```

**VWA-Adv.**
1. Clone VisualWebArena and VWA-Adv's `agent-attack` repository and point to
   them:
   ```bash
   export VWA_ADV_REPO=/path/to/visualwebarena
   export AGENT_ATTACK_REPO=/path/to/agent-attack
   ```
2. Serve the VisualWebArena Reddit site (Postmill) on port 8080. The paper's
   runs used the VisualWebArena Postmill image as an Apptainer container
   (`postmill.sif`), started before each run:
   ```bash
   apptainer instance start --fakeroot --no-home --writable-tmpfs postmill.sif postmill_instance
   apptainer exec --fakeroot --no-home instance://postmill_instance rm -f /run/postgresql/.init
   apptainer exec --fakeroot --no-home instance://postmill_instance \
       chown -R postgres:postgres /usr/local/pgsql/data /run/postgresql
   apptainer exec --fakeroot --no-home instance://postmill_instance \
       sh -c "sed -i 's/listen 80;/listen 8080;/; s/listen \[::\]:80/listen [::]:8080/' /etc/nginx/conf.d/*.conf"
   apptainer run --fakeroot --no-home instance://postmill_instance > postmill.log 2>&1 &
   # wait until `curl -s http://localhost:8080/` answers, then start the run
   ```
3. Set the VisualWebArena site variables. Only Reddit is used, but
   VisualWebArena reads every site's URL, so the paper's runs pointed all of
   them at the Postmill instance:
   ```bash
   export REDDIT=http://localhost:8080
   export SHOPPING=$REDDIT WIKIPEDIA=$REDDIT HOMEPAGE=$REDDIT CLASSIFIEDS=$REDDIT
   export HF_HUB_OFFLINE=1          # weights are read from the local cache
   ```
   (The paper's environment also set `CLASSIFIEDS_RESET_TOKEN`, which
   VisualWebArena reads; Classifieds itself is not used.) Set `HF_HOME` and
   `ICON_ORACLE_ATTACKED` (below) after these, as the paper's jobs did.
4. Point the harness at the corrected evaluators:
   ```bash
   export STRICT_ASR_CONFIG_DIR=$PWD/strict_configs
   export CLEAN_UA_CONFIG_DIR=$PWD/clean_configs
   ```
5. Two tasks are scored by VWA-Adv's LLM judge with its pinned model
   (gpt-4o-2024-05-13), so an OpenAI key is needed:
   `export OPENAI_API_KEY=...` (never pass it on the command line).
   OpenAI withdrew gpt-4o-2024-05-13 on 2026-10-23, after the paper's runs. The
   verdicts the paper used are stored in each run directory's
   `judge_verdicts.jsonl`; re-running those two tasks now needs a different
   judge model, which can change their scores.

**ICON.** ICON's prober did not train in our setting, so its rectifier is
applied to every attacked episode (perfect detection) and to every clean one
(a detector that always fires). Set this for every run:
```bash
export ICON_ORACLE_ATTACKED=1
```

**Tests** (CPU only):
```bash
python3 -m pytest -q
```

---

## Reproducing the paper

Analysis layer: **42** for Gemma-4 (48 layers), **31** for Qwen3-VL (36 layers),
the same relative depth. The runs save every decoder layer; the layer sweeps in
the paper cover 24–42 (Gemma-4) and 18–31 (Qwen3-VL).

Run every job from a clean commit: each result row records the code version,
and the run scripts used for the paper refused to start with uncommitted
changes in `attacks/ defenses/ environments/ experiments/ extraction/`.

### 1. Runs (GPU)

**AgentDojo**, once per model (`MODEL=gemma4` or `MODEL=qwen3vl`), about 25–40 h
each:
```bash
python experiments/run_attack_matrix.py --agentdojo --model $MODEL --hpc \
    --attack_names template_ipi important_instructions \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding icon attention_tracker \
    --dedup_defended_clean \
    --output_dir final_agentdojo_$MODEL
```

**Attention Tracker at τ = 0.3** (after the run above). Only episodes whose
outcome could differ from the τ = 0.5 run are re-run; the rest are carried over:
```bash
mkdir -p analysis_out
python3 analysis/at_threshold_band.py select final_agentdojo_$MODEL --lo 0.30 --hi 0.50 \
    > final_agentdojo_$MODEL/at_t30_samples.txt
if grep -q . final_agentdojo_$MODEL/at_t30_samples.txt; then   # skip if nothing is in the band
  python experiments/run_attack_matrix.py --agentdojo --model $MODEL --hpc \
      --attack_names template_ipi important_instructions \
      --defense_names attention_tracker_t30 --dedup_defended_clean \
      --only_samples final_agentdojo_$MODEL/at_t30_samples.txt \
      --output_dir final_agentdojo_$MODEL
fi
python3 analysis/at_threshold_band.py merge final_agentdojo_$MODEL --lo 0.30 --hi 0.50 \
    > analysis_out/at_t30_rows_$MODEL.json
```

**VWA-Adv**, once per model, with Postmill running:
```bash
MODEL_ID=google/gemma-4-12B-it;     OUT=final_vwa_gemma4      # or:
# MODEL_ID=Qwen/Qwen3-VL-8B-Instruct; OUT=final_vwa_qwen3vl
mkdir -p $OUT
python experiments/run_with_dual_judge.py \
    --dj_records $OUT/judge_verdicts.jsonl --dj_require_openai \
    --model_id $MODEL_ID \
    --attack_names template_ipi typographic steganographic bim_caption_transfer clip_attack_transfer \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding gaussian_blur_1.5 gaussian_blur_3.0 ecso icon \
    --site reddit --max_steps 15 \
    --output_dir $OUT
```
The pixel, patch, CrossInject and COMET attacks are then added to the same
directory with the same launcher and defences (completed episodes are skipped):
```bash
python experiments/run_with_dual_judge.py \
    --dj_records $OUT/judge_verdicts.jsonl --dj_require_openai \
    --model_id $MODEL_ID \
    --attack_names comet cross_modal_injection pixel_perturbation adversarial_patch \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding gaussian_blur_1.5 gaussian_blur_3.0 ecso icon \
    --site reddit --max_steps 15 \
    --output_dir $OUT
```

Always use the launcher for VWA-Adv runs, including resumes: it is part of the
harness hash recorded with every row. Attention Tracker is not run on VWA-Adv
(every long page observation scores below its threshold, clean or attacked).

**Run-to-run check** (optional, about 1 h per model). Re-runs a fixed sample of
50 episodes and compares scores, prompts, completions and activations with the
original:
```bash
mkdir -p analysis_out
[ -s analysis_out/repro_samples_$MODEL.txt ] || \
  python3 analysis/repro_check.py select final_agentdojo_$MODEL > analysis_out/repro_samples_$MODEL.txt
python experiments/run_attack_matrix.py --agentdojo --model $MODEL --hpc \
    --attack_names template_ipi important_instructions \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding icon attention_tracker \
    --dedup_defended_clean --only_samples analysis_out/repro_samples_$MODEL.txt \
    --output_dir repro_agentdojo_$MODEL
python3 analysis/repro_check.py compare final_agentdojo_$MODEL repro_agentdojo_$MODEL \
    --samples analysis_out/repro_samples_$MODEL.txt --out analysis_out/repro_agentdojo_$MODEL
```

### 2. Analysis (CPU)

All scripts are read-only on the run directories. Outputs go to
`analysis_out/` (figures to `figures/`) under the names below, which later
steps and `make_figures.py` expect. Run in this order: `directions.py` needs
the homology outputs.

```bash
mkdir -p analysis_out
AD="final_agentdojo_gemma4 final_agentdojo_qwen3vl"

# Behaviour: ASR / utility table, exposure, Attention Tracker thresholds, ICON flips
python3 analysis/agentdojo_behaviour.py $AD --out analysis_out/agentdojo_behaviour_gemma4_qwen3vl \
    --at_rows final_agentdojo_gemma4=analysis_out/at_t30_rows_gemma4.json \
              final_agentdojo_qwen3vl=analysis_out/at_t30_rows_qwen3vl.json
# Compliance bounds (bootstrap over user tasks); --common: cross-model comparison on identical pairs
python3 analysis/compliance_bootstrap.py $AD --B 2000 --out analysis_out/compliance_bootstrap.json
python3 analysis/compliance_bootstrap.py $AD --B 2000 --common --out analysis_out/compliance_bootstrap_common.json
# VWA-Adv behaviour, prevented / enabled attacks
python3 analysis/vwa_behaviour.py final_vwa_gemma4 final_vwa_qwen3vl --out analysis_out/vwa_behaviour_gemma4_qwen3vl

# Per run directory: label, directory, analysis layer, layer sweep, extra flag for VWA-Adv
while read label dir layer band extra; do
  # reach, displacement tests, probe with permutation null
  python3 analysis/stats.py --results_dir $dir --layer $layer --n_perm 500 $extra \
      --out analysis_out/stats_$label.json
  # displacement by outcome and layer
  python3 analysis/geometry.py --results_dir $dir --turn_strategy mean --skip_probe $extra \
      --out_dir analysis_out/geometry_v2_$label
  # shape of the clean and attacked clouds
  python3 analysis/persistent_homology.py --results_dir $dir --layer $layer --turn_strategy arrival \
      $extra --pca_dim 50 --max_points 150 --n_perm 2000 --out analysis_out/homology_$label.json
  # defended states: restoration vs suppression, with the layer sweep
  python3 analysis/rq2_geometry.py --results_dir $dir --attack all --layer $layer \
      --sweep_layers $band $extra --out analysis_out/rq2_$label.json
done <<'RUNS'
agentdojo_gemma4  final_agentdojo_gemma4  42 24-42
agentdojo_qwen3vl final_agentdojo_qwen3vl 31 18-31
vwa_gemma4        final_vwa_gemma4        42 24-42 --accept_step0
vwa_qwen3vl       final_vwa_qwen3vl       31 18-31 --accept_step0
RUNS

# Shared injection direction: consistency, transfer across attacks and to VWA-Adv,
# one-dot-product monitor, defence footprints, Betti-curve figure data
python3 analysis/directions.py --results_dir final_agentdojo_gemma4 --layer 42 --topology \
    --homology_json analysis_out/homology_agentdojo_gemma4.json \
    --transfer_dir final_vwa_gemma4 --transfer_accept_step0 --out analysis_out/directions_agentdojo_gemma4.json
python3 analysis/directions.py --results_dir final_agentdojo_qwen3vl --layer 31 --topology \
    --homology_json analysis_out/homology_agentdojo_qwen3vl.json \
    --transfer_dir final_vwa_qwen3vl --transfer_accept_step0 --out analysis_out/directions_agentdojo_qwen3vl.json
python3 analysis/directions.py --results_dir final_vwa_gemma4  --layer 42 --accept_step0 --out analysis_out/directions_vwa_gemma4.json
python3 analysis/directions.py --results_dir final_vwa_qwen3vl --layer 31 --accept_step0 --out analysis_out/directions_vwa_qwen3vl.json

# What the direction encodes (logit lens); *_SNAPSHOT = each model's Hugging Face snapshot directory
python3 analysis/logit_lens.py --results_dir final_agentdojo_gemma4  --layer 42 \
    --model_dir "$GEMMA_SNAPSHOT" --out analysis_out/logit_lens_agentdojo_gemma4.json
python3 analysis/logit_lens.py --results_dir final_agentdojo_qwen3vl --layer 31 \
    --model_dir "$QWEN_SNAPSHOT"  --out analysis_out/logit_lens_agentdojo_qwen3vl.json

# Every data figure in the paper (PDF)
python3 analysis/make_figures.py --analysis_out analysis_out --out figures
```

`analysis/report_ws2.py` prints a compact text summary of the stats, homology
and defended-state outputs. The homology and direction steps need `ripser`,
`persim`, `scipy` and `scikit-learn`; the figures need `matplotlib`; the logit
lens needs `safetensors`.

---

## Notes on reproducibility

- **Determinism.** Decoding is greedy, so a run is deterministic on given
  hardware and code. A replication of 50 Gemma-4 AgentDojo episodes on a
  different GPU (Quadro RTX 8000, which emulates bfloat16) reproduced the
  attack outcome in 42 of the 45 episodes that fit in its memory, but not the
  activations bit for bit. Use the same GPU type to reproduce activations
  exactly; the runner records the GPU in the run settings and refuses to resume a
  directory on a different GPU type.
- **Static payloads.** Payloads are generated once against the undefended agent
  (cached in each run directory) and reused unchanged under every defence.
- **Benchmark corrections.** VWA-Adv scoring changes (re-specified evaluators,
  excluded tasks, scorable utility subset) are documented in
  `docs/AUDIT_REGISTER.md` and the paper's appendix.

## Citation

```
N. Mahomed and R. Klein, "Benchmarking Indirect Prompt Injection in Multimodal
Agentic Systems: Shared Injection Directions in the Latent Space," Honours
research report, University of the Witwatersrand, 2026.
```
