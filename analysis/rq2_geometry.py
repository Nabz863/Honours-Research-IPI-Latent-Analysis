"""
analysis/rq2_geometry.py
------------------------
RQ2: when a defence is applied, does it RESTORE the clean-condition
representation, or produce a novel SUPPRESSION state?

THE GEOMETRIC FORMULATION
    Four states exist per task, all saved as activations:
        c  = clean                  (clean_<task>)
        a  = attacked               (<attack>_<task>)
        f  = defended clean         (<attack>_defended_<defense>_<task>, clean pass)
        d  = defended attacked      (<attack>_defended_<defense>_<task>)

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

    The significance test (audit 4.1) compares orth against the
    defence's OFF-AXIS footprint ||(f-c) - <f-c,u>u||, not the full
    ||f-c||: the hypothesis is about off-axis movement, so both sides are
    measured after removing the attack axis.

REACHABILITY GATE
    A large fraction of AgentDojo episodes have ||a - c|| == 0 exactly:
    the agent never retrieved the tool output carrying the injection, so
    the attacked forward pass is byte-identical to the clean one. For
    those the attack axis u is undefined and there is nothing to restore.
    They are excluded, and the count is reported -- restricting to the
    episodes where an injection actually entered the representation is the
    population RQ2 is about.

WHICH ATTACK
    --attack names the attacked condition whose defence phase is
    analysed. The final AgentDojo run carries more than one attack into
    defences, and VWA-Adv carries several image attacks, so --attack all
    discovers every attack_name that has at least one defended_attacked
    row and reports each separately. Each attack's p-values are
    Holm-corrected across its OWN defences, not pooled across attacks.

Usage
    python3 analysis/rq2_geometry.py \\
        --results_dir final_agentdojo_qwen3vl \\
        --attack all --layer 31 --out analysis_out/rq2_qwen3vl.json
"""

from __future__ import annotations

import argparse
import json
import os
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

# geometry.py sits next to this file; reuse its user-task grouping so the
# significance test can collapse the non-independent injection-task repeats.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import user_task_group, matched_rank_biserial  # noqa: E402

ZERO_TOL = 1e-6


def holm(entries, key_in, key_out):
    """Holm-Bonferroni step-down with monotonicity enforcement."""
    have = [e for e in entries if e.get(key_in) is not None]
    m = len(have)
    prev = 0.0
    for rank, e in enumerate(sorted(have, key=lambda x: x[key_in])):
        e[key_out] = float(min(1.0, e[key_in] * (m - rank)))
        e[key_out] = max(e[key_out], prev)
        prev = e[key_out]


def load(results_dir, layer):
    """{task_id: states} where states maps:
        'clean'                     -> vector
        ('attacked', attack)        -> vector
        ('def_clean', dn)           -> vector   (attack-independent footprint)
        ('def_att',  attack, dn)    -> vector

    The defended-clean footprint is the clean input processed under the
    defence, so it does not depend on which attack it is compared with; it
    is keyed by defence alone. If the harness writes one such row per
    (attack, defence) they are the same clean-under-defence input and
    coincide, so the last-writer-wins collision is harmless.

    Every attacked condition is loaded; each tensor is read at most once.
    Rows carrying an error are skipped (audit 2b.1): a failed episode may
    still have a loadable tensor, and it must not enter the analysis."""
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    cache = {}

    def vec(sample_id):
        if sample_id in cache:
            return cache[sample_id]
        p = os.path.join(results_dir, sample_id + ".pt")
        v = None
        if os.path.exists(p):
            try:
                try:
                    turns = torch.load(p, map_location="cpu", weights_only=True)
                except TypeError:
                    turns = torch.load(p, map_location="cpu")
                if isinstance(turns, list) and turns:
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
        cond, dn, atk = r.get("condition"), r.get("defense_name"), r.get("attack_name")
        t = r["task_id"]
        sid = r.get("sample_id")
        if not sid:
            continue
        # --dedup_defended_clean (P2a.1): a defended-clean episode's input
        # depends only on the user task, so each (suite, user_task) runs once
        # and its other pairs are POINTER rows -- error=None, no .pt of their
        # own, carrying deduplicated_to=<representative sample_id>. The
        # activation lives in the representative's file, so read that. Deferred
        # pointers (representative not yet complete) carry an error and are
        # already skipped above. Only defended_clean is deduped; other
        # conditions never set the field, so this is a no-op for them.
        read_sid = r.get("deduplicated_to") or sid
        if cond == "clean":
            v = vec(sid)
            if v is not None:
                states[t]["clean"] = v
        elif cond == "attacked":
            v = vec(sid)
            if v is not None:
                states[t][("attacked", atk)] = v
        elif cond == "defended_clean":
            v = vec(read_sid)
            if v is not None:
                states[t][("def_clean", dn)] = v
        elif cond == "defended_attacked":
            v = vec(read_sid)
            if v is not None:
                states[t][("def_att", atk, dn)] = v
    return states


def discover_attacks(states):
    """Every attack_name with at least one defended_attacked row."""
    return sorted({k[1] for s in states.values() for k in s
                   if isinstance(k, tuple) and k[0] == "def_att"})


def decompose(c, a, d, f):
    """Returns the RQ2 quantities for one (task, defence). Unchanged core
    (audit entry 4, verified)."""
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
        # A zero footprint has zero OFF-AXIS footprint: 0.0, not None, so the
        # significance test still counts this pair (audit-review finding 1).
        # None would drop it silently and skip the whole defence's p-value.
        "footprint_orth": (float(np.linalg.norm((f - c) - np.dot(f - c, u) * u))
                           if foot > ZERO_TOL else 0.0),
    }


def analyse_attack(states, attack):
    """All (task, defence) decompositions for one attacked condition, plus
    the per-defence summary rows with the footprint_orth significance test.
    Returns (rows, n_tasks_with_axis, n_zero)."""
    defenses = sorted({k[2] for s in states.values() for k in s
                       if isinstance(k, tuple) and k[0] == "def_att" and k[1] == attack})

    per = defaultdict(list)
    n_zero = 0
    n_axis = 0
    for t, s in states.items():
        c, av = s.get("clean"), s.get(("attacked", attack))
        if c is None or av is None:
            continue
        if np.linalg.norm(av - c) <= ZERO_TOL:
            n_zero += 1
            continue
        n_axis += 1
        for dn in defenses:
            d = s.get(("def_att", attack, dn))
            f = s.get(("def_clean", dn))
            if d is None or f is None:
                continue
            m = decompose(c, av, d, f)
            if m:
                m["task_id"] = t
                m["user_task_group"] = user_task_group(t)
                per[dn].append(m)

    rows = []
    for dn in defenses:
        m = per[dn]
        if not m:
            continue
        g = lambda k: np.array([x[k] for x in m if x[k] is not None], dtype=float)
        # median of a possibly-empty selection: None, never NaN-with-warning
        med = lambda arr: float(np.median(arr)) if np.size(arr) else None
        along_f, orth, foot = g("along_frac"), g("orth"), g("footprint")
        e = {"defense": dn, "n": len(m),
             "along_frac_median": float(np.median(along_f)),
             # A median alone cannot distinguish "most tasks retain ~57%"
             # from "half fully restored, half untouched" (audit 4.2).
             "along_frac_q1": float(np.percentile(along_f, 25)),
             "along_frac_q3": float(np.percentile(along_f, 75)),
             "along_frac_frac_below_0_2": float(np.mean(along_f < 0.2)),
             "along_frac_frac_above_0_8": float(np.mean(along_f > 0.8)),
             "orth_median": float(np.median(orth)),
             "footprint_median": float(np.median(foot)),
             # None when the defence's footprint is zero (on every task: the
             # median is None too). n_* says how many pairs the median covers.
             "orth_over_footprint_median": med(g("orth_over_footprint")),
             "n_orth_over_footprint": int(g("orth_over_footprint").size),
             "footprint_cos_attack_median": med(g("footprint_cos_attack")),
             # ratio only where the off-axis footprint is non-zero (a 0.0 is a
             # real observation for the TEST below, but cannot be divided by)
             "orth_over_footprint_orth_median": med(
                 [x["orth"] / x["footprint_orth"] for x in m
                  if x.get("footprint_orth")]),
             "attack_disp_median": float(np.median(g("attack_disp"))),
             "defended_disp_median": float(np.median(g("defended_disp")))}
        # Off-axis movement vs this defence's own OFF-AXIS footprint. Both
        # sides are measured after removing the attack axis (audit 4.1);
        # comparing against the full norm mixes in the on-axis component.
        #
        # Finding 8: AgentDojo repeats every user task once per injection task,
        # and the repeats share the user prompt and the clean / defended-clean
        # episode, so they are not independent. Collapse to ONE observation per
        # user task (the mean of orth and of footprint_orth over its injection
        # tasks) before the signed-rank test; the per-pair medians above stay
        # descriptive. The group key is geometry.user_task_group, the same key
        # stats.py's probe folds use; task_id carries the suite prefix, so
        # workspace_user_task_0 and slack_user_task_0 stay separate. A zero
        # footprint_orth is a real 0.0 and is kept (finding 1).
        by_group = defaultdict(list)
        for x in m:
            if x.get("footprint_orth") is not None:
                by_group[x["user_task_group"]].append((x["orth"], x["footprint_orth"]))
        g_orth = np.array([np.mean([o for o, _ in v]) for v in by_group.values()])
        g_foot = np.array([np.mean([f for _, f in v]) for v in by_group.values()])
        e["n_groups"] = len(by_group)
        e["n_test_pairs"] = int(sum(len(v) for v in by_group.values()))
        if sps is not None and len(by_group) >= 6 and np.any(g_orth - g_foot):
            W, p = sps.wilcoxon(g_orth, g_foot, zero_method="wilcox",
                                alternative="two-sided")
            e["p_raw"] = float(p)
            e["rank_biserial"] = matched_rank_biserial(g_orth - g_foot)
        rows.append(e)

    if sps is not None:
        holm(rows, "p_raw", "p_holm")       # across this attack's defences only
    return rows, n_axis, n_zero, {k: v for k, v in per.items()}


def _f(v, spec):
    """None-safe fixed-width format."""
    w = int(spec.lstrip("+").split(".")[0] or 0)
    return "n/a".rjust(w) if v is None else format(v, spec)


def print_attack(attack, rows, n_tasks, n_axis, n_zero, results_dir, layer):
    print(f"\n=== RQ2: restoration vs suppression  ({results_dir}, "
          f"attack={attack}, layer {layer}) ===")
    print(f"tasks with at least one recorded state: {n_tasks}")
    print(f"tasks with a defined attack axis (||a-c|| > 0): {n_axis}")
    print(f"excluded, attack displacement exactly zero (injection never "
          f"entered context): {n_zero}")
    if not rows:
        print("  (no defended pairs for this attack)")
        return

    print(f"\n{'defence':26s} {'n':>4s} {'grp':>4s} {'along_frac':>11s} {'orth':>9s} "
          f"{'footprint':>10s} {'orth/foot':>10s} {'cos(f,atk)':>11s} "
          f"{'orth/f_orth':>12s} {'r':>7s} {'p':>10s}")
    for e in rows:
        p = e.get("p_holm")
        star = " *" if (p is not None and p < 0.05) else "  "
        print(f"{e['defense']:26s} {e['n']:4d} {e['n_groups']:4d} {e['along_frac_median']:11.3f} "
              f"{e['orth_median']:9.2f} {e['footprint_median']:10.2f} "
              f"{_f(e['orth_over_footprint_median'], '10.2f')} "
              f"{_f(e['footprint_cos_attack_median'], '11.3f')} "
              f"{_f(e['orth_over_footprint_orth_median'], '12.2f')} "
              f"{_f(e.get('rank_biserial'), '+7.3f')} "
              f"{'n/a' if p is None else f'{p:.3g}':>10s}{star}")

    for e in rows:
        print(f"  {e['defense']:26s} -> {verdict(e)}")


def verdict(e):
    """Descriptive label for one defence row. 1.3 and 0.4 are a-priori cut-offs
    for labelling the table, NOT tuned (audit 4.4); the reported quantities,
    not these labels, are what the write-up cites. A defence with no footprint
    (defended-clean == clean on every task) has orth/foot undefined: say so
    rather than letting NaN fall through every comparison into PARTIAL."""
    r = e.get("orth_over_footprint_median")
    af = e["along_frac_median"]
    if r is None or not np.isfinite(r):
        return ("UNDEFINED -- defence has no footprint (orth/foot undefined); "
                "read orth and along_frac directly")
    if r < 1.3 and abs(af) < 0.4:
        return "RESTORATION -- back toward clean, no novel direction"
    if r >= 1.3:
        return "SUPPRESSION -- substantial movement off the attack axis"
    return "PARTIAL -- attack component persists"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--attack", default="all",
                    help="attacked condition to analyse, or 'all' to discover "
                         "every attack_name that has defended_attacked rows and "
                         "report each separately. Each attack's p-values are "
                         "Holm-corrected across its own defences only.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    states = load(a.results_dir, a.layer)
    n_tasks = len(states)

    if a.attack == "all":
        attacks = discover_attacks(states)
        if not attacks:
            sys.exit("no defended_attacked rows found -- is this a directory "
                     "with a defence phase? (RQ2 needs defended_clean and "
                     "defended_attacked conditions.)")
    else:
        attacks = [a.attack]

    out = {"config": vars(a), "n_tasks": n_tasks, "per_attack": {}}
    tables = []
    for atk in attacks:
        rows, n_axis, n_zero, raw = analyse_attack(states, atk)
        for e in rows:
            e["verdict"] = verdict(e)
        tables.append((atk, rows, n_axis, n_zero))
        out["per_attack"][atk] = {
            "n_axis_defined": n_axis,
            "n_excluded_zero_delta": n_zero,
            "per_defense": rows,
            "raw": raw,
        }

    # JSON first, atomically (the stats.py finding-4 rule): the console table
    # must never be able to cost the output file.
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        os.replace(tmp, a.out)

    print("\nreading the table:")
    print("  along_frac  1.0 = defence left the state where the attack put it")
    print("              0.0 = returned to the clean position on the attack axis")
    print("  orth/foot   ~1  = off-axis movement is just this defence's own")
    print("                    footprint -> RESTORATION")
    print("              >>1 = moved into directions neither clean nor attack")
    print("                    explains -> SUPPRESSION STATE")
    print("  orth/f_orth orth against the OFF-AXIS footprint (the tested")
    print("              quantity, audit 4.1)")
    print("  p           Wilcoxon on per-user-task means of orth vs footprint_orth")
    print("              (one observation per user task, finding 8; grp = n of")
    print("              them, min 6), two-sided, Holm-corrected across this")
    print("              attack's defences. Significant = orth differs from the")
    print("              off-axis footprint; the sign of r says which way:")
    print("  r           matched-pairs rank-biserial on the same per-user-task")
    print("              means. + = more off-axis movement than the footprint")
    print("              (suppression direction); - = less.")

    for atk, rows, n_axis, n_zero in tables:
        print_attack(atk, rows, n_tasks, n_axis, n_zero, a.results_dir, a.layer)
    if a.out:
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
