"""
experiments/calibrate_icon.py
-----------------------------
Trains ICON's Latent Space Trace Prober, then tests whether it
generalises to attacks it was not trained on.

WHY CALIBRATION IS MANDATORY
    LatentSpaceTraceProber is a randomly-initialised 1D-CNN + MLP. An
    untrained one classifies at chance, so ICON would flag roughly half
    of all episodes at random -- rectifying clean episodes and missing
    attacked ones. defenses/icon_runtime.load_prober() refuses to run
    without a trained checkpoint for exactly this reason.

WHAT IT TRAINS ON
    Real episodes from this project's own AgentDojo suites, recorded
    through the same code path inference uses (LocalVLMPipelineElement
    with icon_record_only=True). Training on features produced by a
    different code path than inference is a classic silent failure, so
    both go through record_features().

    label 0 = clean episode
    label 1 = attacked episode with --train_attack (default template_ipi)

    LIMITATION (review P2a-A4): features are taken from each episode's LAST
    turn, where an attacked episode has read the injected tool result; the
    runners consult the prober on EVERY turn, including turn 0, which it was
    not trained on. End-to-end ICON numbers inherit that mismatch. The
    final runs use ORACLE mode (ICON_ORACLE_ATTACKED=1), which does not use
    the prober at all.

THE OOD TEST, AND WHY IT MATTERS FOR THE WRITE-UP
    ICON's headline claim is out-of-distribution transfer: the paper
    trains on TrojanTools and evaluates on AgentDojo at 97-98%. A prober
    trained on template_ipi and evaluated on template_ipi is
    IN-distribution and will flatter itself -- it is not the paper's
    claim and should not be reported as if it were.

    So this script holds out entire ATTACK FAMILIES. It trains on clean
    vs --train_attack, then reports detection rate on each of
    --eval_attacks separately (default gcg, obfuscated_activation),
    which the prober has never seen. That number is the one comparable
    to the paper.

    obfuscated_activation is the most informative held-out family: it is
    explicitly optimised to mimic clean-condition geometry, so it is the
    adversarial worst case for a latent-space detector.

TASK SELECTION
    Tasks are sampled evenly across suites, and the SAME task ids are
    used for the clean and attacked passes, so a positive and its
    control differ only by the injection -- not by task difficulty or
    suite composition.

Usage
    python3 experiments/calibrate_icon.py --model qwen3vl --n_tasks 60 \\
        --out defenses/icon_prober_qwen3vl.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from defenses.icon import ICONDefense, LatentSpaceTraceProber, train_prober  # noqa: E402
from environments.agentdojo_tasks import load_tasks, run_agentdojo_episode  # noqa: E402
from extraction.hooks import ActivationExtractor, _resolve_target_layers  # noqa: E402
from extraction.model_loader import (  # noqa: E402
    load_llavanext_7b, load_internvl_8b, load_qwen3vl_8b, load_gemma4_12b)
from defenses.icon_runtime import features_from_layer_stats, save_prober  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOADERS = {"llava": load_llavanext_7b,
           "internvl": load_internvl_8b,
           "qwen3vl": load_qwen3vl_8b,
           "gemma4": load_gemma4_12b}


def build_payload(attack_name: str, model, processor, task, context):
    """Reuses the matrix's own attack registry and payload construction so
    calibration payloads are byte-identical to the ones the attack matrix
    generated. A prober trained on differently-constructed payloads would
    be calibrated for an attack that was never actually run.

    Mirrors run_attack_matrix.py's placement (audit entry 10, P1.1): the
    payload goes into EVERY vector the user task reads
    (injection_candidates), appended to each vector's own default, or
    replacing it for an attack that replaces vector content. The old rule
    -- the suite's first vector for every task -- put the payload where the
    task reads it in 54 of 629 pairs.
    """
    from experiments.run_attack_matrix import _build_attack_registry
    reg = _build_attack_registry(model, processor)
    if attack_name not in reg:
        raise SystemExit(f"unknown attack {attack_name!r}; have {sorted(reg)}")
    atk = reg[attack_name]

    from environments.agentdojo_tasks import injection_candidates
    suite = context["suites"][task.suite_name]
    injection_defaults = suite.get_injection_vector_defaults()
    if not injection_defaults:
        return {}
    vectors = injection_candidates(suite, suite.get_user_task_by_id(task.user_task_id))
    clean_vector_text = injection_defaults[vectors[0]]
    res = atk.generate({"page_text": clean_vector_text, "injectable_field": "page_text"},
                       target_behavior=task.target_behavior)
    adv = res.adversarial_text or clean_vector_text
    if not (res.metadata or {}).get("replaces_vector_content") and adv.startswith(clean_vector_text):
        tail = adv[len(clean_vector_text):]
        return {v: injection_defaults[v] + tail for v in vectors}
    return {v: adv for v in vectors}


def collect(model, processor, extractor, tasks, context, icon,
            attack_name=None, limit=None):
    """Runs episodes in record-only mode and returns, per task, the LAST
    turn's per-layer statistics ({"stats": [L, 3, H], "fis": [L]}); the
    prober's layers are chosen afterwards, once (review B9)."""
    feats, used = [], []
    for t in tasks[:limit]:
        try:
            inj = ({} if attack_name is None
                   else build_payload(attack_name, model, processor, t, context))
            out = run_agentdojo_episode(model, processor, extractor, t, context,
                                        injections=inj, icon=icon,
                                        icon_record_only=True)
            f = out.get("turn_icon_features") or []
            if f:
                # LAST turn, not turn 0. In AgentDojo the injection lives
                # in a TOOL RESULT, which does not exist until the agent
                # has called a tool -- so at turn 0 the clean and attacked
                # prompts are byte-identical and so are their attention
                # entropies. Measured directly: clean and attacked feature
                # tensors had mean=0.087067 std=0.070260 and a per-element
                # difference of exactly 0.0, which is why the prober sat at
                # loss=ln(2). This is the same pre-injection artefact that
                # forced analysis/geometry.py off turn_strategy="first".
                feats.append(f[-1])
                used.append(t.task_id)
        except Exception as e:
            logger.warning("  %s failed (%s: %s)", t.task_id,
                           type(e).__name__, str(e)[:120])
    return feats, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl", choices=list(LOADERS))
    ap.add_argument("--n_tasks", type=int, default=60)
    ap.add_argument("--train_attack", default="template_ipi")
    ap.add_argument("--eval_attacks", nargs="*", default=["gcg"],
                    help="held-out attack families. obfuscated_activation is "
                         "NOT in _build_attack_registry (it needs a "
                         "clean_reference_activation the matrix supplies "
                         "separately), so it cannot be built here.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    out_path = a.out or f"defenses/icon_prober_{a.model}.pt"

    model, processor = LOADERS[a.model](hpc=True)
    model.eval()
    extractor = ActivationExtractor(
        model, target_layers=_resolve_target_layers(model, None),
        token_aggregation="last")
    icon = ICONDefense()

    tasks, context = load_tasks(pair_with_injections=True)
    # even coverage across suites, same ids for clean and attacked
    by_suite = {}
    for t in tasks:
        by_suite.setdefault(t.suite_name, []).append(t)
    per = max(1, a.n_tasks // max(1, len(by_suite)))
    chosen = [t for ts in by_suite.values() for t in ts[:per]][:a.n_tasks]
    logger.info("calibrating on %d tasks across %d suites",
                len(chosen), len(by_suite))

    logger.info("--- clean pass (label 0) ---")
    clean_f, clean_ids = collect(model, processor, extractor, chosen, context, icon)
    logger.info("--- %s pass (label 1) ---", a.train_attack)
    atk_f, atk_ids = collect(model, processor, extractor, chosen, context, icon,
                             attack_name=a.train_attack)

    # keep only tasks that produced BOTH, so every positive has its control
    common = [t for t in clean_ids if t in set(atk_ids)]
    cf = [clean_f[clean_ids.index(t)] for t in common]
    af = [atk_f[atk_ids.index(t)] for t in common]
    logger.info("paired training samples: %d clean + %d attacked", len(cf), len(af))
    if len(common) < 10:
        sys.exit("too few paired samples to train a prober")

    # ONE layer set for every sample (review B9), chosen on the TRAINING
    # pairs: the top-K layers by how much an injection raises mean FIS
    # (attacked - clean), i.e. the most attack-sensitive layers, which is
    # what the selection is for (review P2a-A5; clean-only FIS picks the
    # most concentrated layers, e.g. attention sinks). Per-sample ranking
    # put different layers into the same feature positions.
    fis_gap = (torch.stack([f["fis"] for f in af]).mean(0)
               - torch.stack([f["fis"] for f in cf]).mean(0))
    layers = torch.topk(fis_gap, min(icon.top_k_layers, fis_gap.numel())).indices.tolist()
    logger.info("prober layers (top-%d by attacked-clean mean FIS): %s", len(layers), layers)
    stats_c, stats_a = cf, af
    cf = [features_from_layer_stats(f["stats"], layers) for f in stats_c]
    af = [features_from_layer_stats(f["stats"], layers) for f in stats_a]

    # Save the raw features BEFORE training. A prober that ends at
    # loss=ln(2)=0.6931 with accuracy exactly 0.5 has emitted identical
    # logits for both classes -- which means either the features do not
    # separate, or the gradient is dead. Neither is diagnosable without
    # the features themselves, and re-running 30 minutes of GPU to get
    # them back is wasteful.
    import numpy as _np
    feat_path = Path(f"analysis_out/icon_features_{a.model}.pt")
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"clean": stats_c, "attacked": stats_a, "layers": layers,
                "task_ids": common}, feat_path)
    logger.info("saved raw features -> %s", feat_path)

    C = torch.stack(cf).float(); A = torch.stack(af).float()
    logger.info("feature stats  clean: mean=%.6f std=%.6f  attacked: mean=%.6f std=%.6f",
                float(C.mean()), float(C.std()), float(A.mean()), float(A.std()))
    logger.info("per-class mean abs difference: %.6e",
                float((C.mean(0) - A.mean(0)).abs().mean()))

    prober = LatentSpaceTraceProber()
    stats = train_prober(prober, cf + af, [0] * len(cf) + [1] * len(af),
                         epochs=a.epochs)
    logger.info("train: %s", stats)
    if abs(stats["final_loss"] - 0.6931) < 1e-3:
        logger.error("prober did not train: loss is ln(2) and accuracy is "
                     "chance. Do NOT report any detection number from this "
                     "run. Inspect %s before rerunning.", feat_path)

    report = {"model": a.model, "train_attack": a.train_attack, "layers": layers,
              "n_paired": len(common), "train": stats, "ood": {}}

    # --- held-out attack families -------------------------------------
    prober.eval()
    for atk in a.eval_attacks:
        logger.info("--- OOD eval: %s ---", atk)
        f, ids = collect(model, processor, extractor, chosen, context, icon,
                         attack_name=atk)
        if not f:
            report["ood"][atk] = {"n": 0, "note": "no features collected"}
            continue
        f = [features_from_layer_stats(x["stats"], layers) for x in f]
        with torch.no_grad():
            pred = torch.stack([prober(z.float()).argmax(-1).squeeze()
                                for z in f])
        rate = float((pred == 1).float().mean())
        report["ood"][atk] = {"n": len(f), "detection_rate": rate}
        logger.info("  %s detected in %d/%d (%.1f%%)", atk,
                    int((pred == 1).sum()), len(f), 100 * rate)

    # false positives on the clean set the prober was trained to reject
    with torch.no_grad():
        cp = torch.stack([prober(z.float()).argmax(-1).squeeze() for z in cf])
    report["clean_false_positive_rate"] = float((cp == 1).float().mean())

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_prober(prober, layers, out_path,
                meta={"model": a.model, "train_attack": a.train_attack, "n_paired": len(common)})
    logger.info("wrote prober -> %s", out_path)

    print("\n" + "=" * 62)
    print(f"train accuracy (in-distribution, {a.train_attack}): "
          f"{stats['train_accuracy']:.3f}")
    print(f"clean false-positive rate                    : "
          f"{report['clean_false_positive_rate']:.3f}")
    for atk, r in report["ood"].items():
        if r.get("n"):
            print(f"held-out detection, {atk:22s}: {r['detection_rate']:.3f}  (n={r['n']})")
    print("\nThe held-out rates are the ones comparable to the paper's OOD")
    print("claim. Train accuracy is in-distribution and will look better")
    print("than it deserves -- do not report it as detection performance.")
    print("=" * 62)

    rp = a.report or f"analysis_out/icon_calibration_{a.model}.json"
    Path(rp).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(rp, "w"), indent=2)
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
