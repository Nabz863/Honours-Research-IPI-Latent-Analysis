"""
analysis/rq2_geometry.py
------------------------
RQ2: when a defence is applied, does it RESTORE the clean-condition
representation, or produce a novel SUPPRESSION state?

THE GEOMETRIC FORMULATION
    Four states exist per task, all saved as activations:
        c  = clean                  (clean_<task>)
        a  = attacked               (template_ipi_<task>)
        f  = defended clean         (defended_<defense>_<task>)
        d  = defended attacked      (template_ipi_defended_<defense>_<task>)

    The attack moves the representation from c to a. Write the attack
    axis as u = (a - c)/||a - c||. Decompose the defended-attacked
    displacement (d - c) into a component ALONG that axis and a
    component ORTHOGONAL to it:

        along = <(d - c), u>              signed; ||a-c|| means "no
                                          movement back", 0 means "fully
                                          back to the clean position on
                                          this axis"
        orth  = ||(d - c) - along * u||   displacement in directions the
                                          attack never used

    RESTORATION looks like: along -> 0 AND orth no larger than the
    defence's own footprint ||f - c||. The defended state sits where a
    clean input under that defence would sit.

    SUPPRESSION looks like: orth substantially exceeding ||f - c||. The
    defence has pushed the representation somewhere neither clean nor
    attacked -- a third region that happens to yield correct behaviour.

    ||f - c|| is the essential control. Every defence perturbs the
    representation even with nothing to defend against (spotlighting
    rewrites the prompt), so without measuring that footprint you cannot
    tell "moved somewhere new" from "moved the way this defence always
    moves things".

REACHABILITY GATE
    83.5% of AgentDojo episodes have ||a - c|| == 0 exactly: the agent
    never retrieved the tool output carrying the injection, so the
    attacked forward pass is byte-identical to the clean one. For those
    the attack axis u is undefined and there is nothing to restore.
    They are excluded, and the count is reported -- restricting to the
    episodes where an injection actually entered the representation is
    the population RQ2 is about.

Usage
    python3 analysis/rq2_geometry.py \\
        --results_dir agentdojo_matrix_v4_qwen3vl \\
        --layer 31 --out analysis_out/rq2_qwen3vl.json
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
    sys.exit("torch required")

try:
    from scipy import stats as sps
except ImportError:
    sps = None
    print("[warn] scipy missing -- no p-values\n")

ZERO_TOL = 1e-6


def holm(entries, key_in, key_out):
    have = [e for e in entries if e.get(key_in) is not None]
    m = len(have)
    prev = 0.0
    for rank, e in enumerate(sorted(have, key=lambda x: x[key_in])):
        e[key_out] = float(min(1.0, e[key_in] * (m - rank)))
        e[key_out] = max(e[key_out], prev)
        prev = e[key_out]


def load(results_dir, layer, attack="template_ipi"):
    """{task_id: {state_key: vector}} where state_key is one of
    'clean', 'attacked', ('def_clean', defense), ('def_att', defense)."""
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    cache = {}

    def vec(sample_id):
        if sample_id in cache:
            return cache[sample_id]
        p = os.path.join(results_dir, sample_id + ".pt")
        v = None
        if os.path.exists(p):
            try:
                turns = torch.load(p, map_location="cpu", weights_only=True)
                vs = [t[layer].float().numpy() for t in turns
                      if isinstance(t, dict) and layer in t]
                v = np.mean(vs, axis=0) if vs else None
            except Exception:
                v = None
        cache[sample_id] = v
        return v

    states = defaultdict(dict)
    for r in rows:
        if r.get("error"):
            continue
        cond, dn = r.get("condition"), r.get("defense_name")
        t = r["task_id"]
        v = vec(r["sample_id"])
        if v is None:
            continue
        if cond == "clean":
            states[t]["clean"] = v
        elif cond == "attacked" and r["attack_name"] == attack:
            states[t]["attacked"] = v
        elif cond == "defended_clean":
            states[t][("def_clean", dn)] = v
        elif cond == "defended_attacked" and r["attack_name"] == attack:
            states[t][("def_att", dn)] = v
    return states


def decompose(c, a, d, f):
    """Returns the RQ2 quantities for one (task, defence)."""
    atk = a - c
    n_atk = float(np.linalg.norm(atk))
    if n_atk <= ZERO_TOL:
        return None
    u = atk / n_atk
    dd = d - c
    along = float(np.dot(dd, u))
    orth = float(np.linalg.norm(dd - along * u))
    foot = float(np.linalg.norm(f - c))
    return {
        "attack_disp": n_atk,
        "defended_disp": float(np.linalg.norm(dd)),
        "along": along,
        "orth": orth,
        "footprint": foot,
        # 1.0 = defence left it where the attack put it (on this axis)
        # 0.0 = defence returned it to the clean position on this axis
        "along_frac": along / n_atk,
        # >1 means the defence moved it further off-axis than it moves a
        # clean input, i.e. a direction neither clean nor attack explains
        "orth_over_footprint": (orth / foot) if foot > ZERO_TOL else None,
        "dist_to_attacked": float(np.linalg.norm(d - a)),
        # Does the defence's OWN displacement share a direction with the
        # attack axis? If it does, projecting that axis out of (d - c)
        # also strips part of the footprint, so orth/footprint is
        # deflated by construction rather than by restoration. Comparing
        # orth against footprint_orth (both measured after removing the
        # attack axis) is the like-for-like version.
        "footprint_cos_attack": (float(np.dot(f - c, u) / foot)
                                 if foot > ZERO_TOL else None),
        "footprint_orth": (float(np.linalg.norm((f - c) - np.dot(f - c, u) * u))
                           if foot > ZERO_TOL else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--attack", default="template_ipi",
                    help="which attacked condition to analyse. AgentDojo "
                         "carried only template_ipi into the defence phase; "
                         "VWA-Adv carried five, so each needs its own run.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    states = load(a.results_dir, a.layer, a.attack)
    defenses = sorted({k[1] for s in states.values() for k in s
                       if isinstance(k, tuple) and k[0] == "def_att"})

    n_tasks = len(states)
    per = defaultdict(list)
    n_zero = 0
    for t, s in states.items():
        c, av = s.get("clean"), s.get("attacked")
        if c is None or av is None:
            continue
        if np.linalg.norm(av - c) <= ZERO_TOL:
            n_zero += 1
            continue
        for dn in defenses:
            d, f = s.get(("def_att", dn)), s.get(("def_clean", dn))
            if d is None or f is None:
                continue
            m = decompose(c, av, d, f)
            if m:
                per[dn].append(m)

    print(f"\n=== RQ2: restoration vs suppression  ({a.results_dir}, "
          f"attack={a.attack}, layer {a.layer}) ===")
    print(f"tasks with at least one recorded state: {n_tasks}")
    print(f"excluded, attack displacement exactly zero (injection never "
          f"entered context): {n_zero}")

    rows = []
    for dn in defenses:
        m = per[dn]
        if not m:
            continue
        g = lambda k: np.array([x[k] for x in m if x[k] is not None], dtype=float)
        along_f, orth, foot = g("along_frac"), g("orth"), g("footprint")
        e = {"defense": dn, "n": len(m),
             "along_frac_median": float(np.median(along_f)),
             # A median alone cannot distinguish "most tasks retain ~57%"
             # from "half fully restored, half untouched". Report the
             # spread alongside it.
             "along_frac_q1": float(np.percentile(along_f, 25)),
             "along_frac_q3": float(np.percentile(along_f, 75)),
             "along_frac_frac_below_0_2": float(np.mean(along_f < 0.2)),
             "along_frac_frac_above_0_8": float(np.mean(along_f > 0.8)),
             "orth_median": float(np.median(orth)),
             "footprint_median": float(np.median(foot)),
             "orth_over_footprint_median": float(np.median(g("orth_over_footprint"))),
             "footprint_cos_attack_median": float(np.median(g("footprint_cos_attack"))),
             "orth_over_footprint_orth_median": float(np.median(np.array(
                 [x["orth"] / x["footprint_orth"] for x in m
                  if x.get("footprint_orth")], dtype=float))),
             "attack_disp_median": float(np.median(g("attack_disp"))),
             "defended_disp_median": float(np.median(g("defended_disp")))}
        # is the off-axis movement bigger than this defence's own footprint?
        # Test orth against footprint_ORTH, not the full footprint norm.
        # The hypothesis is about OFF-AXIS movement, so both sides must be
        # measured after removing the attack axis; comparing against the
        # full norm mixes in the footprint's on-axis component.
        foot_orth = np.array([x["footprint_orth"] for x in m
                              if x.get("footprint_orth")], dtype=float)
        if (sps is not None and len(orth) >= 6 and len(foot_orth) == len(orth)
                and np.any(orth - foot_orth)):
            W, p = sps.wilcoxon(orth, foot_orth, zero_method="wilcox",
                                alternative="two-sided")
            e["p_raw"] = float(p)
            diff = orth - foot_orth
            nz = diff[diff != 0]
            e["rank_biserial"] = float((np.sum(nz > 0) - np.sum(nz < 0)) / nz.size)
        rows.append(e)

    if sps is not None:
        holm(rows, "p_raw", "p_holm")

    print(f"\n{'defence':26s} {'n':>4s} {'along_frac':>11s} {'orth':>9s} "
          f"{'footprint':>10s} {'orth/foot':>10s} {'cos(f,atk)':>11s} "
          f"{'orth/f_orth':>12s} {'p':>10s}")
    for e in rows:
        p = e.get("p_holm")
        star = " *" if (p is not None and p < 0.05) else "  "
        print(f"{e['defense']:26s} {e['n']:4d} {e['along_frac_median']:11.3f} "
              f"{e['orth_median']:9.2f} {e['footprint_median']:10.2f} "
              f"{e['orth_over_footprint_median']:10.2f} "
              f"{e['footprint_cos_attack_median']:11.3f} "
              f"{e['orth_over_footprint_orth_median']:12.2f} "
              f"{'n/a' if p is None else f'{p:.3g}':>10s}{star}")

    print("\nreading the table:")
    print("  along_frac  1.0 = defence left the state where the attack put it")
    print("              0.0 = returned to the clean position on the attack axis")
    print("  orth/foot   ~1  = off-axis movement is just this defence's own")
    print("                    footprint -> RESTORATION")
    print("              >>1 = moved into directions neither clean nor attack")
    print("                    explains -> SUPPRESSION STATE")
    print("  p           Wilcoxon, orth vs footprint, Holm-corrected across")
    print("              defences. Significant = off-axis movement is not")
    print("              explained by the defence's own footprint.")

    for e in rows:
        r = e["orth_over_footprint_median"]
        af = e["along_frac_median"]
        # 1.3 and 0.4 are descriptive cut-offs for labelling the table,
        # chosen a priori and NOT tuned. The reported quantities, not
        # these labels, are what the write-up should cite.
        if r < 1.3 and abs(af) < 0.4:
            v = "RESTORATION -- back toward clean, no novel direction"
        elif r >= 1.3:
            v = "SUPPRESSION -- substantial movement off the attack axis"
        else:
            v = "PARTIAL -- attack component persists"
        print(f"  {e['defense']:26s} -> {v}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"config": vars(a), "n_tasks": n_tasks,
                   "n_excluded_zero_delta": n_zero, "per_defense": rows,
                   "raw": {k: v for k, v in per.items()}},
                  open(a.out, "w"), indent=2, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
