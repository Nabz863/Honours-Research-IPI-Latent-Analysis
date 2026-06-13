# Phase 1 — Hook Engineering & Baseline Collection

## What's here

```
ipi_project/
├── extraction/
│   ├── hooks.py          # ActivationExtractor — core hook infrastructure
│   └── model_loader.py   # VLM loaders for LLaVA-NeXT, Qwen2-VL, InternVL
├── baselines/
│   └── collect_baselines.py  # Clean-condition baseline collection script
├── tests/
│   └── test_hooks.py     # 13 unit tests (no GPU required)
└── hpc_baseline.sh       # SLURM job script for Wits HPC
```

## Validate locally first (no GPU needed)

```bash
# Run all unit tests
python tests/test_hooks.py

# Dry-run the full pipeline with synthetic inputs (no model weights downloaded)
python baselines/collect_baselines.py --model llava7b --dry_run --n_samples 5
```

## On HPC

```bash
# 1. Install dependencies
pip install torch transformers accelerate sentencepiece pillow

# 2. Submit baseline collection job
sbatch hpc_baseline.sh

# 3. Check output
ls baselines/llava7b_clean/activations/   # sample_0000.pt ... sample_0099.pt
cat baselines/llava7b_clean/metadata.json
```

## How activations are stored

Each `sample_NNNN.pt` is a Python dict saved with `torch.save`:

```python
{
    16: tensor([...]),   # shape [hidden_dim] — last-token activation at layer 16
    20: tensor([...]),
    24: tensor([...]),
    28: tensor([...]),
}
```

Layer indices follow the **late-middle heuristic** (50–90% of model depth)
unless overridden with `--target_layers`.

## Key design decisions

| Decision | Rationale |
|---|---|
| Context manager for hook lifecycle | Guarantees removal on exception — prevents memory leaks in long-context agentic tasks |
| Activations stored as detached CPU tensors | Avoids holding GPU memory between inference calls |
| Late-middle layer heuristic | Semantic convergence typically in deeper layers; avoids task-specific final layers |
| `token_aggregation='last'` default | Standard for causal LMs; captures the model's "decision state" |
| Batch size 1 assumed | Simplifies extraction; agentic tasks are inherently sequential |

## Phase 2 readiness

Once baselines are collected, Phase 2 loads the same extractor against
attacked inputs and computes:
- **Activation delta** `‖Δh‖` — Euclidean displacement from clean baseline
- **Cosine similarity to clean manifold**
- **Attention entropy** and **deflection angle** (requires attention weights)

These are the geometric signatures that characterise each attack modality.
