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
from geometry import (user_task_group, matched_rank_biserial, arrival_or_acts,  # noqa: E402
                      load_turns, same_path, normalise_asr)

ZERO_TOL = 1e-6


def holm(entries, key_in, key_out):
    """Holm-Bonferroni step-down with monotonicity enforcement."""
    have = [e for e in entries if e.get(key_in) is not None and np.isfinite(e[key_in])]
    m = len(have)
    prev = 0.0
    for rank, e in enumerate(sorted(have, key=lambda x: x[key_in])):
        e[key_out] = float(min(1.0, e[key_in] * (m - rank)))
        e[key_out] = max(e[key_out], prev)
        prev = e[key_out]


def load(results_dir, layer):
    """{task_id: states} at one layer -- see load_multi."""
    return load_multi(results_dir, [layer])[layer]


def load_multi(results_dir, layers, turn="mean", accept_step0=False, info=None):
    """{layer: {task_id: states}} where states maps:
        'clean'                     -> vector
        ('attacked', attack)        -> vector
        ('def_clean', dn)           -> vector   (attack-independent footprint)
        ('def_att',  attack, dn)    -> vector

    The defended-clean footprint is the clean input processed under the
    defence, so it does not depend on which attack it is compared with; it
    is keyed by defence alone. If the harness writes one such row per
    (attack, defence) they are the same clean-under-defence input and
    coincide, so the last-writer-wins collision is harmless.

    Every attacked condition is loaded; each tensor file is read at most once
    and every requested layer is taken from that one read (the layer sweep).
    Rows carrying an error are skipped (audit 2b.1): a failed episode may
    still have a loadable tensor, and it must not enter the analysis."""
    layers = list(layers)
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    if turn == "arrival":
        return _load_arrival(results_dir, layers, rows, accept_step0,
                             info if info is not None else {})
    cache = {}

    def vecs(sample_id):
        if sample_id in cache:
            return cache[sample_id]
        p = os.path.join(results_dir, sample_id + ".pt")
        out = {}
        if os.path.exists(p):
            try:
                try:
                    turns = torch.load(p, map_location="cpu", weights_only=True)
                except TypeError:
                    turns = torch.load(p, map_location="cpu")
                if isinstance(turns, list) and turns:
                    for L in layers:
                        vs = [t[L].float().numpy() for t in turns
                              if isinstance(t, dict) and L in t]
                        if vs:
                            out[L] = np.mean(vs, axis=0)
            except Exception:
                out = {}
        cache[sample_id] = out
        return out

    states = {L: defaultdict(dict) for L in layers}
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
            key, v = "clean", vecs(sid)
        elif cond == "attacked":
            key, v = ("attacked", atk), vecs(sid)
        elif cond == "defended_clean":
            key, v = ("def_clean", dn), vecs(read_sid)
        elif cond == "defended_attacked":
            key, v = ("def_att", atk, dn), vecs(read_sid)
        else:
            continue
        for L, x in v.items():
            if np.all(np.isfinite(x)):
                states[L][t][key] = x
            elif info is not None:
                info["nonfinite"] = info.get("nonfinite", 0) + 1
    return states


def _load_arrival(results_dir, layers, rows, accept_step0, info):
    """turn='arrival': every state at the payload's ARRIVAL step, the decision
    the RQ1 headline (geometry.displacement_by_outcome_payload) measures.
    Keys (per attack, because clean is taken at that attack's arrival step):
        ('clean', atk), ('attacked', atk)            at t  = arrival(clean, attacked)
        ('def_clean', atk, dn), ('def_att', atk, dn) at t' = arrival(def_clean, def_att)
    The defended pair is kept only if t' == t, so all four states are the same
    decision of the same task; otherwise it is counted (defended_arrival_differs).
    Unexposed / non-arrival / missing-transcript pairs are counted in info,
    never silently dropped. info[atk] = {status: n, ..., 'defended': {dn: {...}}}."""
    clean, att, dclean, datt = {}, {}, {}, {}
    for r in rows:
        if r.get("error") or not r.get("sample_id"):
            continue
        cond, t_id = r.get("condition"), r["task_id"]
        sid = r["sample_id"]
        read_sid = r.get("deduplicated_to") or sid
        if cond == "clean":
            clean[t_id] = sid
        elif cond == "attacked":
            att[(t_id, r.get("attack_name"))] = sid
        elif cond == "defended_clean":
            dclean[(t_id, r.get("defense_name"))] = read_sid
        elif cond == "defended_attacked":
            datt[(t_id, r.get("attack_name"), r.get("defense_name"))] = read_sid
    needs = defaultdict(set)
    plan = []                                  # (t_id, [(key, sid)], t, counter)
    t_att = {}
    for (t_id, atk), a_sid in att.items():
        c_sid = clean.get(t_id)
        ia = info.setdefault(atk, {"defended": {}})
        if c_sid is None:
            ia["no_clean"] = ia.get("no_clean", 0) + 1
            continue
        t, st = arrival_or_acts(results_dir, c_sid, a_sid, accept_step0=accept_step0)
        if t is None:
            ia[st] = ia.get(st, 0) + 1
            continue
        t_att[(t_id, atk)] = (t, c_sid)
        plan.append((t_id, [(("clean", atk), c_sid), (("attacked", atk), a_sid)], t, ia, st))
    for (t_id, atk, dn), da_sid in datt.items():
        ia = info.setdefault(atk, {"defended": {}})
        idd = ia["defended"].setdefault(dn, {})
        ta = t_att.get((t_id, atk))
        dc_sid = dclean.get((t_id, dn))
        if ta is None or dc_sid is None:
            k = "undefended_pair_unusable" if ta is None else "no_defended_clean"
            idd[k] = idd.get(k, 0) + 1
            continue
        t, c_sid = ta
        td, st = arrival_or_acts(results_dir, dc_sid, da_sid, accept_step0=accept_step0)
        if st == "not_exposed":
            # the defence kept the payload out entirely (e.g. a withheld tool
            # result): defended-attacked == defended-clean. That is complete
            # restoration, not a missing pair -- measure it at the undefended
            # arrival step (excluding it biased RQ2 against such defences).
            td, st = t, "blocked_identical"
        elif td is None:
            idd[st] = idd.get(st, 0) + 1
            continue
        if td != t:
            idd["defended_arrival_differs"] = idd.get("defended_arrival_differs", 0) + 1
            continue
        # the same decision only if the defended agent reached step t by the same
        # tool calls as the undefended one (review round 7, finding 5)
        sp = same_path(results_dir, c_sid, dc_sid, t)
        if not sp:
            k = "defended_path_differs" if sp is False else "no_transcript"
            idd[k] = idd.get(k, 0) + 1
            continue
        members = [(("def_clean", atk, dn), dc_sid), (("def_att", atk, dn), da_sid)]
        if st == "blocked_identical":     # its own stratum (review round 9)
            members.append((("blocked", atk, dn), None))
        plan.append((t_id, members, td, idd, "blocked_identical" if st == "blocked_identical" else "ok"))
    for _, members, t, _, _ in plan:
        for _, sid in members:
            if sid is not None:
                needs[sid].add(t)
    vec = load_turns(results_dir, needs, layers)
    states = {L: defaultdict(dict) for L in layers}
    for t_id, members, t, counter, status in plan:
        got = [vec.get((sid, t)) for _, sid in members if sid is not None]
        if any(g is None for g in got):
            counter["no_tensor"] = counter.get("no_tensor", 0) + 1   # never silently dropped
            continue
        counter[status] = counter.get(status, 0) + 1                 # counted once the pair loads
        for (key, sid), v in zip([m for m in members if m[1] is not None], got):
            for L, x in v.items():
                if np.all(np.isfinite(x)):
                    states[L][t_id][key] = x
                else:
                    info["nonfinite"] = info.get("nonfinite", 0) + 1
        if any(m[1] is None for m in members):                        # blocked flag, every layer
            for L in layers:
                states[L][t_id][("blocked", members[0][0][1], members[0][0][2])] = True
    return states


def _clean_of(s, attack):
    """clean state: per-attack (arrival mode) or shared (mean mode)."""
    return s.get(("clean", attack), s.get("clean"))


def _def_clean_of(s, attack, dn):
    return s.get(("def_clean", attack, dn), s.get(("def_clean", dn)))


def load_outcomes(results_dir):
    """{task_id: {('attacked', attack) | ('def_att', attack, dn): asr}} with asr
    True / False / None. None = unscored: asr_success missing, or
    scoring_error_final (audit register: excluded from ASR). Errored rows are
    skipped, as in load_multi."""
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    out = defaultdict(dict)
    for r in rows:
        if r.get("error"):
            continue
        asr = normalise_asr(r)
        if r.get("condition") == "attacked":
            out[r["task_id"]][("attacked", r.get("attack_name"))] = asr
        elif r.get("condition") == "defended_attacked":
            out[r["task_id"]][("def_att", r.get("attack_name"), r.get("defense_name"))] = asr
    return out


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
        # along the attack axis measured from the DEFENDED-CLEAN state:
        # 1 = the defence kept the whole attack displacement, 0 = the
        # defended-attacked state sits where this defence puts a clean input.
        # Unlike along_frac it does not carry the footprint's own on-axis part
        # (<f-c,u>/||a-c||), which scales with 1/attack size and confounds any
        # comparison between strong and weak attacks (RQ2 review, finding 4).
        "along_frac_def": float(np.dot(d - f, u)) / n_atk,
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
        c, av = _clean_of(s, attack), s.get(("attacked", attack))
        if c is None or av is None:
            continue
        if np.linalg.norm(av - c) <= ZERO_TOL:
            n_zero += 1
            continue
        n_axis += 1
        for dn in defenses:
            d = s.get(("def_att", attack, dn))
            f = _def_clean_of(s, attack, dn)
            if d is None or f is None:
                continue
            m = decompose(c, av, d, f)
            if m:
                m["blocked"] = bool(s.get(("blocked", attack, dn)))
                m["task_id"] = t
                m["user_task_group"] = user_task_group(t)
                per[dn].append(m)

    rows = []
    for dn in defenses:
        # 'blocked' pairs (the defence kept the payload out of BOTH defended
        # episodes, d == f) are their own stratum: complete restoration by
        # construction, but a blanket filter or an always-firing detector
        # produces the same picture. Reported separately, with the share whose
        # footprint is non-zero; the tests below use the other pairs.
        blocked = [x for x in per[dn] if x.get("blocked")]
        m = [x for x in per[dn] if not x.get("blocked")]
        if not m and not blocked:
            continue
        if not m:
            rows.append({"defense": dn, "n": 0, "n_groups": 0, "n_blocked": len(blocked),
                         "blocked_footprint_nonzero": int(sum(x["footprint"] > ZERO_TOL for x in blocked)),
                         "along_frac_median": None, "orth_median": None,
                         "orth_over_footprint_median": None, "footprint_median": None,
                         "footprint_cos_attack_median": None, "orth_over_footprint_orth_median": None})
            continue
        g = lambda k: np.array([x[k] for x in m if x[k] is not None], dtype=float)
        # median of a possibly-empty selection: None, never NaN-with-warning
        med = lambda arr: float(np.median(arr)) if np.size(arr) else None
        along_f, orth, foot = g("along_frac"), g("orth"), g("footprint")
        e = {"defense": dn, "n": len(m), "n_blocked": len(blocked),
             "blocked_footprint_nonzero": int(sum(x["footprint"] > ZERO_TOL for x in blocked)),
             "along_frac_median": float(np.median(along_f)),
             # A median alone cannot distinguish "most tasks retain ~57%"
             # from "half fully restored, half untouched" (audit 4.2).
             "along_frac_q1": float(np.percentile(along_f, 25)),
             "along_frac_q3": float(np.percentile(along_f, 75)),
             "along_frac_frac_below_0_2": float(np.mean(along_f < 0.2)),
             "along_frac_frac_above_0_8": float(np.mean(along_f > 0.8)),
             "along_frac_def_median": med(g("along_frac_def")),
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


# ---------------------------------------------------------------------------
# RQ2 extensions: behaviour link, probe-based restoration, layer sweep
# ---------------------------------------------------------------------------

OUTCOME_CLASSES = ("prevented", "persisted", "enabled", "never")


def _outcome_class(a_asr, d_asr):
    """Undefended attack outcome x defended outcome, per (task, defence)."""
    if a_asr is None or d_asr is None:
        return None
    if a_asr and not d_asr:
        return "prevented"
    if a_asr and d_asr:
        return "persisted"
    return "enabled" if d_asr else "never"


def outcome_split(per_dn, outcomes, attack, dn, min_paired=8, min_class=4):
    """Does the geometry differ when the defence actually STOPPED the attack?
    Each (task, defence) decomposition is labelled by behaviour:
        prevented  undefended attack succeeded, defended one did not
        persisted  both succeeded
        enabled    only the defended one succeeded
        never      neither did
    The compared quantity is along_frac_def (attack-axis component measured
    from the DEFENDED-CLEAN state): along_frac from clean carries the
    footprint's on-axis part / ||a-c|| and so differs between strong and weak
    attacks even when restoration is identical (RQ2 review, finding 4); the
    median attack size per class is reported so the reader can see the mix.
    PRIMARY test: user tasks with pairs in BOTH prevented and persisted --
    paired Wilcoxon on (prevented mean - persisted mean), >= min_paired tasks.
    SECONDARY: Mann-Whitney U on per-user-task means, >= min_class tasks per
    class (with tasks in both classes it is miscalibrated; exploratory).
    A defence that prevents attacks while along_frac_def stays high is
    suppressing, not restoring."""
    by = {c: [] for c in OUTCOME_CLASSES}
    n_unscored = 0
    for x in per_dn:
        o = outcomes.get(x["task_id"], {})
        c = _outcome_class(o.get(("attacked", attack)), o.get(("def_att", attack, dn)))
        if c is None:
            n_unscored += 1
            continue
        by[c].append(x)
    med = lambda v: float(np.median(v)) if len(v) else None
    res = {"n_unscored": n_unscored}
    task_means = {}
    for c, xs in by.items():
        g = defaultdict(list)
        for x in xs:
            g[x["user_task_group"]].append(x["along_frac_def"])
        task_means[c] = {k: float(np.mean(v)) for k, v in g.items()}
        res[c] = {
            "n": len(xs), "n_groups": len(g),
            "along_frac_def_median": med([x["along_frac_def"] for x in xs]),
            "along_frac_median": med([x["along_frac"] for x in xs]),
            "attack_disp_median": med([x["attack_disp"] for x in xs]),
            "orth_minus_footprint_orth_median": med([x["orth"] - x["footprint_orth"] for x in xs]),
            "orth_over_footprint_orth_median": med([x["orth"] / x["footprint_orth"]
                                                    for x in xs if x["footprint_orth"]]),
        }
    gp, gs = task_means["prevented"], task_means["persisted"]
    both = sorted(set(gp) & set(gs))
    res["n_tasks_in_both"] = len(both)
    if sps is not None and len(both) >= min_paired:
        diff = np.array([gp[k] - gs[k] for k in both])
        if np.any(diff):
            res["prevented_vs_persisted_p"] = float(sps.wilcoxon(diff, zero_method="wilcox",
                                                                 alternative="two-sided")[1])
            res["prevented_vs_persisted_r"] = matched_rank_biserial(diff)
    if sps is not None and len(gp) >= min_class and len(gs) >= min_class:
        xp, xs_ = list(gp.values()), list(gs.values())
        U, p = sps.mannwhitneyu(xp, xs_, alternative="two-sided")
        res["mw_p_exploratory"] = float(p)
        # common-language effect: P(prevented task's along_frac_def < persisted's)
        res["mw_auc"] = float(1.0 - U / (len(xp) * len(xs_)))
    return res


def _median_ci95(x):
    """Distribution-free 95% CI for a median: order statistics x_(k), x_(n-k+1)
    with k the largest integer such that P(Bin(n, .5) <= k-1) <= .025.
    Covers >= 95% for any continuous distribution (a percentile bootstrap of a
    median under-covers at small n -- review round 7, finding 4)."""
    from scipy.stats import binom
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    k = max([j for j in range(1, n // 2 + 1) if binom.cdf(j - 1, n, 0.5) <= 0.025], default=None)
    if k is None:
        return None
    return [float(x[k - 1]), float(x[n - k])]


def probe_restoration(states, attack, seed=0, max_splits=5, min_test=8,
                      min_logit_gap=1.0):
    """Is the attack's LINEAR SIGNATURE removed by the defence?
    A logistic probe is trained to separate clean from attacked states on the
    undefended reached pairs (grouped folds by user task), and each held-out
    user task's states are scored with the probe's LOGIT (decision function;
    probabilities saturate in 3840 dimensions, RQ2 review finding 2) by a
    probe that never saw that task. Every defence shifts every input, so the
    reference for a defended-attacked state is the same task's defended-clean
    state:
        signature_removed_fraction = (z_att - z_def_att) / (z_att - z_def_clean)
    per user task, defined where z_att - z_def_clean >= min_logit_gap (the
    dropped tasks are counted); median with a distribution-free 95% CI over
    user tasks (order statistics; needs >= min_test user tasks).
    1 = the signature is gone under the defence, 0 = untouched.
    WHAT IT CANNOT SHOW: the probe sees one direction only. A defence that
    removes the signature but moves the state somewhere new scores 1 here and
    is suppression, not restoration -- read with orth / footprint_orth (the
    decomposition test); restoration needs both (RQ2 review finding 1).
    p_raw: Wilcoxon of z_def_att vs z_def_clean (>= min_test user tasks, Holm
    across defences): rejecting = the signature is still DETECTABLE under the
    defence. Not rejecting is not evidence of removal -- the CI is.
    indist: a second probe trained def_clean vs def_att under this defence
    (grouped folds). For each held-out user task the SAME fold model scores
    both states; indist_paired_auc = share of user tasks with z_def_att >
    z_def_clean (ties 1/2). ~0.5 = the attack is no longer linearly decodable
    under the defence, without the out-of-distribution assumption the first
    probe makes. DESCRIPTIVE ONLY: cross-fitting makes the held-out
    differences dependent, so a Wilcoxon on them rejects 11-18% at a nominal
    5% (review round 8); a valid test needs a refitting permutation, which
    is not run. (A pooled AUC was biased toward 0.5 -- review round 7.)"""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"note": "scikit-learn missing"}

    def new_clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(solver="liblinear", dual=True,
                                                max_iter=500, random_state=seed))

    tasks = [t for t, s in states.items()
             if _clean_of(s, attack) is not None and s.get(("attacked", attack)) is not None
             and np.linalg.norm(s[("attacked", attack)] - _clean_of(s, attack)) > ZERO_TOL]
    groups = np.array([user_task_group(t) for t in tasks])
    ug = sorted(set(groups.tolist()))
    if len(ug) < 4:
        return {"note": f"{len(ug)} reached user tasks: too few for grouped folds"}
    defenses = sorted({k[2] for t in tasks for k in states[t]
                       if isinstance(k, tuple) and k[0] == "def_att" and k[1] == attack})
    scores = defaultdict(dict)                      # task -> what -> logit
    gkf = GroupKFold(n_splits=min(max_splits, len(ug)))
    idx = np.arange(len(tasks))
    for tr, te in gkf.split(idx, groups=groups):
        X = np.vstack([_clean_of(states[tasks[i]], attack) for i in tr] +
                      [states[tasks[i]][("attacked", attack)] for i in tr])
        y = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
        clf = new_clf()
        clf.fit(X, y)
        for i in te:
            t, s = tasks[i], states[tasks[i]]
            Z = lambda v: float(clf.decision_function(v[None, :])[0])
            scores[t]["clean"] = Z(_clean_of(s, attack))
            scores[t]["attacked"] = Z(s[("attacked", attack)])
            for dn in defenses:
                d, f = s.get(("def_att", attack, dn)), _def_clean_of(s, attack, dn)
                if d is not None and f is not None and not s.get(("blocked", attack, dn)):
                    scores[t][("def_att", dn)] = Z(d)
                    scores[t][("def_clean", dn)] = Z(f)
    acc = np.mean([(scores[t]["attacked"] > 0) + (scores[t]["clean"] <= 0) for t in tasks]) / 2
    out = {"probe_heldout_accuracy": float(acc), "n_tasks": len(tasks), "n_groups": len(ug),
           "per_defense": []}
    for dn in defenses:
        g = defaultdict(lambda: defaultdict(list))
        for t in tasks:
            sc = scores[t]
            if ("def_att", dn) in sc:
                for k, v in (("clean", sc["clean"]), ("attacked", sc["attacked"]),
                             ("def_att", sc[("def_att", dn)]), ("def_clean", sc[("def_clean", dn)])):
                    g[user_task_group(t)][k].append(v)
        if not g:
            continue
        keys = sorted(g)
        M = {k: np.array([np.mean(g[u][k]) for u in keys])
             for k in ("clean", "attacked", "def_att", "def_clean")}
        e = {"defense": dn, "n_groups": len(g),
             **{f"z_{k}_median": float(np.median(v)) for k, v in M.items()}}
        den = M["attacked"] - M["def_clean"]
        ok = den >= min_logit_gap
        e["n_fraction_defined"], e["n_fraction_dropped"] = int(ok.sum()), int((~ok).sum())
        if ok.any():
            fr = (M["attacked"][ok] - M["def_att"][ok]) / den[ok]
            e["signature_removed_fraction_median"] = float(np.median(fr))
            if fr.size >= min_test:
                e["signature_removed_fraction_ci95"] = _median_ci95(fr)
        diff = M["def_att"] - M["def_clean"]
        e["z_defatt_minus_defclean_median"] = float(np.median(diff))
        e["rank_biserial_defatt_vs_defclean"] = matched_rank_biserial(diff)
        if sps is not None and len(g) >= min_test and np.any(diff):
            e["p_raw"] = float(sps.wilcoxon(M["def_att"], M["def_clean"],
                                            zero_method="wilcox", alternative="two-sided")[1])
        # in-distribution: is the attack still linearly decodable under the defence?
        tk = [t for t in tasks if ("def_att", dn) in scores[t]]       # blocked pairs never scored
        gk = np.array([user_task_group(t) for t in tk])
        if len(set(gk.tolist())) >= 4:
            pdiff = defaultdict(list)
            for tr, te in GroupKFold(n_splits=min(max_splits, len(set(gk.tolist())))).split(
                    np.arange(len(tk)), groups=gk):
                X = np.vstack([_def_clean_of(states[tk[i]], attack, dn) for i in tr] +
                              [states[tk[i]][("def_att", attack, dn)] for i in tr])
                y = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
                clf = new_clf()
                clf.fit(X, y)
                for i in te:
                    zc = float(clf.decision_function(_def_clean_of(states[tk[i]], attack, dn)[None, :])[0])
                    za = float(clf.decision_function(states[tk[i]][("def_att", attack, dn)][None, :])[0])
                    pdiff[gk[i]].append(za - zc)
            dd = np.array([np.mean(v) for v in pdiff.values()])
            e["indist_paired_auc"] = float(np.mean(dd > 0) + 0.5 * np.mean(dd == 0))
            e["indist_n_groups"] = int(dd.size)
        out["per_defense"].append(e)
    if sps is not None:
        holm(out["per_defense"], "p_raw", "p_holm")
    return out


def layer_sweep(states_by_layer, attack):
    """The decomposition at every loaded layer: per defence and layer, the
    medians and the grouped Wilcoxon from analyse_attack. Descriptive -- the
    write-up's tests are at the analysis layer; p-values here are not
    corrected across layers."""
    out = defaultdict(list)
    for L in sorted(states_by_layer):
        rows, n_axis, _, _ = analyse_attack(states_by_layer[L], attack)
        for e in rows:
            out[e["defense"]].append({
                "layer": L, "n": e["n"], "n_groups": e["n_groups"], "n_axis": n_axis,
                "along_frac_median": e["along_frac_median"],
                "orth_over_footprint_orth_median": e["orth_over_footprint_orth_median"],
                "rank_biserial": e.get("rank_biserial"), "p_raw": e.get("p_raw")})
    return dict(out)


def _parse_layers(spec):
    """'24-42' or '18,24,31' -> sorted unique ints."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


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
        print(f"{e['defense']:26s} {e['n']:4d} {e['n_groups']:4d} {_f(e['along_frac_median'], '11.3f')} "
              f"{_f(e['orth_median'], '9.2f')} {_f(e['footprint_median'], '10.2f')} "
              f"{_f(e['orth_over_footprint_median'], '10.2f')} "
              f"{_f(e['footprint_cos_attack_median'], '11.3f')} "
              f"{_f(e['orth_over_footprint_orth_median'], '12.2f')} "
              f"{_f(e.get('rank_biserial'), '+7.3f')} "
              f"{'n/a' if p is None else f'{p:.3g}':>10s}{star}")

    for e in rows:
        print(f"  {e['defense']:26s} -> {verdict(e)}"
              + (f"   [+{e['n_blocked']} blocked pairs (payload kept out; "
                 f"{e['blocked_footprint_nonzero']} with a non-zero footprint), reported apart]"
                 if e.get("n_blocked") else ""))


def _write_json(path, obj):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    os.replace(tmp, path)


def _why_empty(a, info):
    """A hint naming the likely cause when nothing survived the pairing."""
    if getattr(a, "turn", "mean") != "arrival":
        return "is this a directory with a defence phase?"
    tot = defaultdict(int)
    for v in info.values():
        if isinstance(v, dict):
            for k, n in v.items():
                if isinstance(n, int):
                    tot[k] += n
    msg = "arrival-step pairing: " + ", ".join(f"{k}={n}" for k, n in sorted(tot.items()))
    if tot.get("not_payload_arrival", 0) and not a.accept_step0:
        msg += " -- payload in the first observation (VWA)? pass --accept_step0"
    return msg


def print_arrival(info):
    print("\npayload-arrival pairing (what was measured, and what was excluded):")
    for atk, v in sorted((k, v) for k, v in info.items() if isinstance(v, dict)):
        und = ", ".join(f"{k}={n}" for k, n in sorted(v.items()) if isinstance(n, int))
        print(f"  {atk:24s} undefended: {und}")
        for dn, dv in sorted(v.get("defended", {}).items()):
            print(f"  {'':24s}   {dn:22s} " + ", ".join(f"{k}={n}" for k, n in sorted(dv.items())))
    if info.get("nonfinite"):
        print(f"  non-finite vectors dropped: {info['nonfinite']}")


def print_extensions(rows, prb):
    if not rows:
        return
    print("\n  behaviour link (along_frac_def median by outcome; p: prevented vs persisted,"
          " paired within user tasks, Holm)")
    for e in rows:
        b = e.get("by_outcome", {})
        cells = "  ".join(f"{c}={_f(b.get(c, {}).get('along_frac_def_median'), '.2f')}"
                          f"(n={b.get(c, {}).get('n', 0)})" for c in OUTCOME_CLASSES)
        print(f"  {e['defense']:26s} {cells}  "
              f"p={_f(b.get('prevented_vs_persisted_p_holm'), '.3g')}")
    if prb and "per_defense" in prb:
        print(f"\n  attack signature under the defence (probe logits; held-out probe acc "
              f"{prb['probe_heldout_accuracy']:.3f}). removed: 1 = signature gone, 0 = intact,"
              f" [95% CI]; p: still detectable vs defended-clean (Holm); indist: share of user"
              f" tasks where the attack is still decodable under the defence (0.5 = not; no test)."
              f" Signature removal alone is NOT"
              f" restoration -- read with orth/f_orth.")
        for e in prb["per_defense"]:
            ci = e.get("signature_removed_fraction_ci95")
            print(f"  {e['defense']:26s} removed={_f(e.get('signature_removed_fraction_median'), '.2f')}"
                  f" [{_f(ci[0] if ci else None, '.2f')}, {_f(ci[1] if ci else None, '.2f')}]"
                  f" (n={e['n_fraction_defined']}, dropped {e['n_fraction_dropped']})  "
                  f"p={_f(e.get('p_holm'), '.3g')}  indist paired AUC={_f(e.get('indist_paired_auc'), '.2f')}"
                  f" (descriptive)")
    elif prb:
        print(f"\n  probe restoration: {prb.get('note')}")


def verdict(e):
    """Descriptive label for one defence row. 1.3 and 0.4 are a-priori cut-offs
    for labelling the table, NOT tuned (audit 4.4); the reported quantities,
    not these labels, are what the write-up cites. A defence with no footprint
    (defended-clean == clean on every task) has orth/foot undefined: say so
    rather than letting NaN fall through every comparison into PARTIAL."""
    r = e.get("orth_over_footprint_median")
    af = e.get("along_frac_median")
    if not e.get("n"):
        return ("BLOCKED -- the defence kept the payload out of every defended pair "
                "(d == f); complete restoration OR a blanket filter -- read the footprint")
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
    ap.add_argument("--sweep_layers", default=None,
                    help="also run the decomposition at these layers, e.g. '24-42' "
                         "(descriptive; tests are at --layer). Each file is read once.")
    ap.add_argument("--turn", default="arrival", choices=["arrival", "mean"],
                    help="arrival (default): every state at the payload's arrival step, "
                         "the RQ1 headline's decision; mean: mean over all turns")
    ap.add_argument("--accept_step0", action="store_true",
                    help="payload is in the first observation (VWA): accept arrival at step 0")
    ap.add_argument("--no_probe", action="store_true",
                    help="skip the probe-based restoration score")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sweep = _parse_layers(a.sweep_layers) if a.sweep_layers else []
    arrival_info = {}
    by_layer = load_multi(a.results_dir, sorted(set(sweep) | {a.layer}), turn=a.turn,
                          accept_step0=a.accept_step0, info=arrival_info)
    states = by_layer[a.layer]
    sweep_missing = [L for L in sweep if not any(by_layer.get(L, {}).values())]
    outcomes = load_outcomes(a.results_dir)
    n_tasks = len(states)

    if a.attack == "all":
        attacks = discover_attacks(states)
        if not attacks:
            _write_json(a.out, {"config": vars(a), "arrival": arrival_info, "per_attack": {}})
            sys.exit("no usable defended_attacked pairs -- " + _why_empty(a, arrival_info)
                     + " (RQ2 needs defended_clean and defended_attacked conditions.)")
    else:
        attacks = [a.attack]

    out = {"config": vars(a), "n_tasks": n_tasks, "per_attack": {},
           "arrival": arrival_info,           # what the arrival-step pairing excluded, and why
           "layer_sweep_missing": sweep_missing}  # requested layers with no data in the .pt files
    tables = []
    for atk in attacks:
        rows, n_axis, n_zero, raw = analyse_attack(states, atk)
        if a.turn == "arrival":         # unexposed pairs never enter states in this mode
            n_zero = arrival_info.get(atk, {}).get("not_exposed", 0)
        for e in rows:
            e["verdict"] = verdict(e)
            e["by_outcome"] = outcome_split([x for x in raw.get(e["defense"], []) if not x.get("blocked")],
                                            outcomes, atk, e["defense"])
        if sps is not None:            # behaviour-link tests: Holm across defences
            holm([e["by_outcome"] for e in rows], "prevented_vs_persisted_p",
                 "prevented_vs_persisted_p_holm")
        prb = {} if a.no_probe else probe_restoration(states, atk)
        tables.append((atk, rows, n_axis, n_zero, prb))
        out["per_attack"][atk] = {
            "n_axis_defined": n_axis,
            "n_excluded_zero_delta": n_zero,
            "per_defense": rows,
            "probe_restoration": prb,
            "layer_sweep": layer_sweep(by_layer, atk) if sweep else {},
            "raw": raw,
        }

    # JSON first, atomically (the stats.py finding-4 rule): the console table
    # must never be able to cost the output file.
    _write_json(a.out, out)

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

    if a.turn == "arrival":
        print_arrival(arrival_info)
    for atk, rows, n_axis, n_zero, prb in tables:
        print_attack(atk, rows, n_tasks, n_axis, n_zero, a.results_dir, a.layer)
        print_extensions(rows, prb)
    if a.out:
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
