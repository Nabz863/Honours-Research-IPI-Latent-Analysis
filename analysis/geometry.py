"""
analysis/geometry.py -- latent geometry analysis for IPI attack matrices.

Environment-agnostic: consumes any results directory containing
  attack_results.json  (list of behavioural rows)
  <attack>_<suite>_<task_id>.pt  (list[turn] -> dict{layer_idx: Tensor[d]})

Works unchanged on AgentDojo matrices and on VWA-Adv output, provided the
VWA-Adv driver writes the same two artefacts.

Metrics (Table 3.4 of the proposal):
  - activation delta ||dh||        per layer, per attack condition   (RQ1, RQ2)
  - normalised delta ||dh||/||h||  per layer                          (RQ1)
  - cosine similarity to clean     per layer                          (RQ2)
  - linear probe accuracy          per layer, leakage-controlled      (RQ1, RQ4)
  - PCA projection coords          per layer, for manifold plots      (RQ1, RQ2)

Usage:
  python3 analysis/geometry.py --results_dir agentdojo_matrix_qwen3vl \
      --out_dir analysis_out/qwen3vl --turn_strategy first

  # compare several models at once
  python3 analysis/geometry.py --results_dir agentdojo_matrix_qwen3vl \
      --results_dir agentdojo_matrix_internvl --out_dir analysis_out/combined
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("torch is required (activation tensors are .pt files)")

CLEAN_NAME = "clean"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_rows(results_dir):
    path = os.path.join(results_dir, "attack_results.json")
    with open(path) as f:
        rows = json.load(f)
    for r in rows:
        r["_results_dir"] = results_dir
    return rows


def tensor_path(row):
    """sample_id is exactly the filename stem, e.g.
      sample_id = clean_workspace_user_task_0_injection_task_0
      file      = clean_workspace_user_task_0_injection_task_0.pt
    Note task_id ALREADY carries the suite prefix, so attack_name + task_id
    is the correct fallback -- adding suite_name separately duplicates it."""
    stem = row.get("sample_id")
    if not stem:
        stem = "{}_{}".format(row.get("attack_name"), row.get("task_id"))
    return os.path.join(row["_results_dir"], stem + ".pt")


def load_episode(path, turn_strategy="first"):
    """Return {layer_idx: np.float32[d]} or None.

    turn_strategy:
      first -- turn 0 only. The first forward pass. Clean and attacked
               episodes share the same user task and differ only by the
               injected content, so this isolates representational change.
      mean  -- mean over all turns. Includes trajectory divergence: after
               turn 0 the agent's own tool calls differ between conditions,
               so a delta here confounds representation change with
               behavioural divergence. Report it as secondary, not primary.
      last  -- final turn only. Maximally confounded; diagnostic use only.
    """
    try:
        turns = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only
        turns = torch.load(path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(turns, list) or len(turns) == 0:
        return None

    if turn_strategy == "first":
        picked = [turns[0]]
    elif turn_strategy == "last":
        picked = [turns[-1]]
    elif turn_strategy == "mean":
        picked = turns
    else:
        raise ValueError("unknown turn_strategy: %s" % turn_strategy)

    acc = defaultdict(list)
    for t in picked:
        if not isinstance(t, dict):
            return None
        for layer, vec in t.items():
            # bfloat16 -> float32; numpy has no native bfloat16
            acc[int(layer)].append(vec.detach().float().cpu().numpy())

    return {L: np.mean(np.stack(v), axis=0) for L, v in acc.items()}


def episode_key(row):
    """Identifies the same underlying episode across conditions."""
    return (row["_results_dir"], row.get("suite_name"), row.get("task_id"))


def user_task_group(task_id):
    """Group id for leakage control: every episode sharing a user task --
    clean, attacked, and all injection_task variants -- must land in the
    same fold. task_id looks like '<suite>_user_task_N_injection_task_M',
    so search (not match) for the '<suite>_user_task_N' prefix; anchoring
    at the string start fails because the suite comes first, which would
    silently degrade grouping to one-group-per-episode."""
    m = re.search(r"^(.*user_task_\d+)", str(task_id))
    return m.group(1) if m else str(task_id)


def build_dataset(results_dirs, turn_strategy, verbose=True, include_defended=False):
    rows = []
    for d in results_dirs:
        rows.extend(load_rows(d))

    by_key = defaultdict(dict)
    missing, loaded, errored = 0, 0, 0

    for r in rows:
        # Defended rows carry the SAME attack_name as undefended ones, so
        # keying on attack alone silently overwrites one with the other in
        # any directory holding both. Undefended rows only, unless the
        # caller asks otherwise -- the same fault was found and fixed in
        # analysis/stats.py.
        if not include_defended and (r.get("defense_name") not in (None, "none")):
            continue
        attack = r.get("attack_name")
        p = tensor_path(r)
        if not os.path.exists(p):
            missing += 1
            continue
        acts = load_episode(p, turn_strategy)
        if acts is None:
            errored += 1
            continue
        by_key[episode_key(r)][attack] = {"acts": acts, "row": r}
        loaded += 1

    if verbose:
        total = len(rows)
        print("[join] rows=%d  tensors_loaded=%d  missing_file=%d  unreadable=%d"
              % (total, loaded, missing, errored))
        if total and loaded / total < 0.9:
            print("[join] WARNING: join rate %.1f%% -- every downstream number "
                  "is computed on this subset only." % (100.0 * loaded / total))

    # keep only episodes that have a clean reference to compare against
    paired = {k: v for k, v in by_key.items() if CLEAN_NAME in v}
    if verbose:
        print("[join] episodes with clean reference: %d / %d"
              % (len(paired), len(by_key)))
    return paired


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def per_condition_geometry(paired):
    """activation delta, normalised delta, cosine-to-clean, per layer."""
    out = defaultdict(lambda: defaultdict(list))  # attack -> layer -> [vals]

    for _, conds in paired.items():
        clean = conds[CLEAN_NAME]["acts"]
        for attack, entry in conds.items():
            if attack == CLEAN_NAME:
                continue
            for layer, h_adv in entry["acts"].items():
                h_cln = clean.get(layer)
                if h_cln is None or h_cln.shape != h_adv.shape:
                    continue
                d = h_adv - h_cln
                nd = float(np.linalg.norm(d))
                base = float(np.linalg.norm(h_cln))
                out[attack][layer].append(
                    (nd, nd / base if base else float("nan"), cosine(h_adv, h_cln))
                )

    summary = {}
    for attack, layers in out.items():
        summary[attack] = {}
        for layer, vals in sorted(layers.items()):
            arr = np.array(vals, dtype=np.float64)
            summary[attack][layer] = {
                "n": int(arr.shape[0]),
                "delta_mean": float(np.nanmean(arr[:, 0])),
                "delta_std": float(np.nanstd(arr[:, 0])),
                "delta_norm_mean": float(np.nanmean(arr[:, 1])),
                "cosine_mean": float(np.nanmean(arr[:, 2])),
                "cosine_std": float(np.nanstd(arr[:, 2])),
            }
    return summary


def probe_separability(paired, seed=0):
    """Per-layer linear probe: clean vs each attack condition.

    GroupKFold on user_task so that the clean and attacked variants of the
    same task never land on opposite sides of a split -- they are highly
    correlated by construction and an ungrouped split inflates accuracy.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        print("[probe] scikit-learn not installed -- skipping probe")
        return {}

    attacks = set()
    for conds in paired.values():
        attacks.update(k for k in conds if k != CLEAN_NAME)

    layers = sorted({L for c in paired.values() for L in c[CLEAN_NAME]["acts"]})
    results = {}

    for attack in sorted(attacks):
        results[attack] = {}
        for layer in layers:
            X, y, g = [], [], []
            for _, conds in paired.items():
                if attack not in conds:
                    continue
                grp = user_task_group(conds[CLEAN_NAME]["row"].get("task_id"))
                for label, cond in ((0, CLEAN_NAME), (1, attack)):
                    v = conds[cond]["acts"].get(layer)
                    if v is None:
                        continue
                    X.append(v); y.append(label); g.append(grp)

            X, y, g = np.array(X), np.array(y), np.array(g)
            n_groups = len(set(g.tolist()))
            if len(y) < 20 or len(set(y.tolist())) < 2 or n_groups < 3:
                continue

            n_splits = min(3, n_groups)
            gkf = GroupKFold(n_splits=n_splits)
            scores = []
            for tr, te in gkf.split(X, y, groups=g):
                if len(set(y[tr].tolist())) < 2:
                    continue
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(solver="liblinear", dual=True, C=1.0,
                                       max_iter=200, random_state=seed),
                )
                clf.fit(X[tr], y[tr])
                scores.append(float(clf.score(X[te], y[te])))
            if scores:
                results[attack][layer] = {
                    "acc_mean": float(np.mean(scores)),
                    "acc_std": float(np.std(scores)),
                    "n": int(len(y)),
                    "folds": len(scores),
                    "majority_baseline": float(max(np.mean(y), 1 - np.mean(y))),
                }
    return results


def pca_coords(paired, layer, n_components=2):
    """2D projection of clean + attacked states at one layer, for the
    manifold plots that answer 'restoration vs suppression' visually."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None

    X, labels = [], []
    for _, conds in paired.items():
        for cond, entry in conds.items():
            v = entry["acts"].get(layer)
            if v is not None:
                X.append(v); labels.append(cond)
    if len(X) < n_components + 1:
        return None

    X = np.array(X)
    p = PCA(n_components=n_components, random_state=0)
    Z = p.fit_transform(X)
    return {
        "layer": layer,
        "labels": labels,
        "coords": Z.tolist(),
        "explained_variance_ratio": p.explained_variance_ratio_.tolist(),
    }


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", action="append", required=True,
                    help="repeatable; one per model/environment matrix")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--turn_strategy", default="mean",
                    choices=["first", "mean", "last"])
    ap.add_argument("--pca_layer", type=int, default=None,
                    help="default: deepest available layer")
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument("--include_defended", action="store_true",
                    help="defended rows share attack_name with undefended "
                         "ones and will overwrite them; use rq2_geometry.py "
                         "for defence analysis instead")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paired = build_dataset(args.results_dir, args.turn_strategy,
                           include_defended=args.include_defended)
    if not paired:
        sys.exit("no episodes with a clean reference -- nothing to compare")

    geom = per_condition_geometry(paired)
    probe = {} if args.skip_probe else probe_separability(paired)

    layers = sorted({L for c in paired.values() for L in c[CLEAN_NAME]["acts"]})
    pca_layer = args.pca_layer if args.pca_layer is not None else layers[-1]
    pca = pca_coords(paired, pca_layer)

    payload = {
        "config": vars(args),
        "n_episodes_paired": len(paired),
        "layers": layers,
        "geometry": geom,
        "probe": probe,
        "pca": pca,
    }
    out = os.path.join(args.out_dir, "geometry_summary.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    # flat CSV for tables/plots
    csv = os.path.join(args.out_dir, "geometry_by_layer.csv")
    with open(csv, "w") as f:
        f.write("attack,layer,n,delta_mean,delta_std,delta_norm_mean,"
                "cosine_mean,cosine_std,probe_acc,probe_std\n")
        for attack, layers_d in sorted(geom.items()):
            for layer, m in sorted(layers_d.items()):
                pa = probe.get(attack, {}).get(layer, {})
                f.write("%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s\n" % (
                    attack, layer, m["n"], m["delta_mean"], m["delta_std"],
                    m["delta_norm_mean"], m["cosine_mean"], m["cosine_std"],
                    ("%.4f" % pa["acc_mean"]) if pa else "",
                    ("%.4f" % pa["acc_std"]) if pa else ""))

    print("\n=== summary (turn_strategy=%s, %d paired episodes) ==="
          % (args.turn_strategy, len(paired)))
    for attack in sorted(geom):
        deep = max(geom[attack])
        m = geom[attack][deep]
        pa = probe.get(attack, {}).get(deep, {})
        line = ("  %-24s layer %2d  n=%4d  ||dh||=%8.3f  norm=%.3f  cos=%.4f"
                % (attack, deep, m["n"], m["delta_mean"],
                   m["delta_norm_mean"], m["cosine_mean"]))
        if pa:
            line += "  probe=%.3f+-%.3f (base %.2f)" % (
                pa["acc_mean"], pa["acc_std"], pa["majority_baseline"])
        print(line)
    print("\nwrote %s\n      %s" % (out, csv))


if __name__ == "__main__":
    main()
