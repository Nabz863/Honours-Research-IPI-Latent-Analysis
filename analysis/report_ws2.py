#!/usr/bin/env python3
"""analysis/report_ws2.py -- compact text report of the RQ1/RQ2/RQ3 outputs
(stats_*.json, rq2_*.json, homology_*.json in analysis_out/), for pasting
into a chat instead of copying the JSON files. Read-only.

    python3 analysis/report_ws2.py                 # all four run dirs
    python3 analysis/report_ws2.py --sweep          # + RQ2 layer sweep
    python3 analysis/report_ws2.py --dir analysis_out/meanturn   # the secondary
"""
import argparse
import json
import os
import time

LABELS = ["agentdojo_gemma4", "agentdojo_qwen3vl", "vwa_gemma4", "vwa_qwen3vl"]


def f(v, spec=".3g"):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def counts(d):
    if not isinstance(d, dict):
        return "-"
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items()) if not isinstance(v, dict)) or "-"


def load(path):
    if not os.path.exists(path):
        return None, f"(missing: {path})"
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        return json.load(open(path)), stamp
    except Exception as e:  # noqa: BLE001
        return None, f"(unreadable: {path}: {e})"


def cfg(c):
    c = c or {}
    keys = ("layer", "turn", "turn_strategy", "accept_step0", "n_perm", "sweep_layers")
    return " ".join(f"{k}={c[k]}" for k in keys if k in c)


def report_stats(d):
    print(f"  arrival: " + "; ".join(f"{a}: {counts(v)}" for a, v in sorted((d.get("arrival") or {}).items())))
    print("  reach:   attack | n  nonzero  rate [95% CI]  user tasks  n_eff | ||dh|| median (nonzero)")
    for a, v in sorted((d.get("reachability") or {}).items()):
        lo, hi = (v.get("rate_ci95") or [None, None])
        print(f"    {a:24s} {v.get('n')} {v.get('n_nonzero')} {f(v.get('rate'), '.3f')} "
              f"[{f(lo, '.3f')}, {f(hi, '.3f')}] {v.get('n_groups')} {f(v.get('n_eff'), '.1f')} "
              f"| {f(v.get('delta_median_nonzero'), '.2f')}")
    print("  compare: a vs b | pairs  user tasks  median a  median b  r  p_holm")
    for e in d.get("attack_comparison") or []:
        print(f"    {e.get('attack_a')} vs {e.get('attack_b')}: {e.get('n_pairs')} {e.get('n_groups')} "
              f"{f(e.get('median_a'), '.2f')} {f(e.get('median_b'), '.2f')} "
              f"{f(e.get('rank_biserial_r'), '+.3f')} {f(e.get('p_holm'))}")
    print("  probe:   attack | pairs  acc  null mean  p95  p_perm  p_holm")
    for a, v in sorted((d.get("probe") or {}).items()):
        if "accuracy" not in v:
            print(f"    {a:24s} {v.get('note')}")
            continue
        print(f"    {a:24s} {v.get('n_pairs')} {f(v.get('accuracy'), '.3f')} {f(v.get('null_mean'), '.3f')} "
              f"{f(v.get('null_p95'), '.3f')} {f(v.get('p_perm'))} {f(v.get('p_holm'))}")
    print("  reach->ASR: attack | table [[r&hit, r&miss],[u&hit, u&miss]]  task ASR reached/unreached"
          "  tasks  p_holm | p_within OR_MH | note")
    for e in d.get("reach_vs_asr") or []:
        print(f"    {e.get('attack'):24s} {e.get('table')} {f(e.get('asr_reached_tasks'), '.3f')}/"
              f"{f(e.get('asr_unreached_tasks'), '.3f')} {e.get('n_tasks', '-')} {f(e.get('p_holm'))} | "
              f"{f(e.get('p_within'))} {f(e.get('odds_ratio_mh'))} | {e.get('note', '')}")


def report_rq2(d, sweep):
    arr = d.get("arrival") or {}
    for a, v in sorted(arr.items()):
        if isinstance(v, dict):
            print(f"  arrival {a}: {counts(v)}")
            for dn, dv in sorted((v.get("defended") or {}).items()):
                print(f"      {dn}: {counts(dv)}")
    if arr.get("nonfinite"):
        print(f"  non-finite dropped: {arr['nonfinite']}")
    if d.get("layer_sweep_missing"):
        print(f"  sweep layers with no data: {d['layer_sweep_missing']}")
    for a, pa in sorted((d.get("per_attack") or {}).items()):
        print(f"  [{a}] axis defined {pa.get('n_axis_defined')}, excluded zero-delta/unexposed "
              f"{pa.get('n_excluded_zero_delta')}")
        print("    defence | n grp blk | along_frac med [q1,q3] | along_def | orth/f_orth | r  p_holm | verdict")
        for e in pa.get("per_defense") or []:
            print(f"    {e.get('defense'):22s} {e.get('n')} {e.get('n_groups')} {e.get('n_blocked', 0)}"
                  f"(fp>0:{e.get('blocked_footprint_nonzero', 0)}) | {f(e.get('along_frac_median'), '.2f')} "
                  f"[{f(e.get('along_frac_q1'), '.2f')},{f(e.get('along_frac_q3'), '.2f')}] | "
                  f"{f(e.get('along_frac_def_median'), '.2f')} | {f(e.get('orth_over_footprint_orth_median'), '.2f')} | "
                  f"{f(e.get('rank_biserial'), '+.2f')} {f(e.get('p_holm'))} | {(e.get('verdict') or '')[:40]}")
        print("    behaviour link: defence | prevented / persisted / enabled / never: n, along_def med, ||a-c|| med"
              " | paired tasks, r, p_holm | MW p, auc | unscored")
        for e in pa.get("per_defense") or []:
            b = e.get("by_outcome") or {}
            cells = "  ".join(f"{c}:{(b.get(c) or {}).get('n', 0)},{f((b.get(c) or {}).get('along_frac_def_median'), '.2f')},"
                              f"{f((b.get(c) or {}).get('attack_disp_median'), '.1f')}"
                              for c in ("prevented", "persisted", "enabled", "never"))
            print(f"    {e.get('defense'):22s} {cells} | {b.get('n_tasks_in_both', 0)}, "
                  f"{f(b.get('prevented_vs_persisted_r'), '+.2f')}, {f(b.get('prevented_vs_persisted_p_holm'))} | "
                  f"{f(b.get('mw_p_exploratory'))}, {f(b.get('mw_auc'), '.2f')} | {b.get('n_unscored', 0)}")
        pr = pa.get("probe_restoration") or {}
        if "per_defense" in pr:
            print(f"    signature: probe held-out acc {f(pr.get('probe_heldout_accuracy'), '.3f')} "
                  f"({pr.get('n_groups')} user tasks) | defence | removed [95% CI] (n, dropped) | "
                  f"z clean/att/defclean/defatt | p_holm | indist paired AUC (n)")
            for e in pr["per_defense"]:
                ci = e.get("signature_removed_fraction_ci95") or [None, None]
                print(f"    {e.get('defense'):22s} {f(e.get('signature_removed_fraction_median'), '.2f')} "
                      f"[{f(ci[0], '.2f')},{f(ci[1], '.2f')}] ({e.get('n_fraction_defined')},{e.get('n_fraction_dropped')}) | "
                      f"{f(e.get('z_clean_median'), '.1f')}/{f(e.get('z_attacked_median'), '.1f')}/"
                      f"{f(e.get('z_def_clean_median'), '.1f')}/{f(e.get('z_def_att_median'), '.1f')} | "
                      f"{f(e.get('p_holm'))} | {f(e.get('indist_paired_auc'), '.2f')} ({e.get('indist_n_groups', '-')})")
        elif pr:
            print(f"    signature: {pr.get('note')}")
        if sweep:
            for dn, rows in sorted((pa.get("layer_sweep") or {}).items()):
                print(f"    sweep {dn}: " + " ".join(
                    f"L{r.get('layer')}:{f(r.get('along_frac_median'), '.2f')}/{f(r.get('orth_over_footprint_orth_median'), '.1f')}"
                    for r in rows))


def report_homology(d):
    be = d.get("backend") or {}
    print(f"  backend ripser={be.get('ripser')} persim={be.get('persim')}")
    print(f"  arrival: " + "; ".join(f"{a}: {counts(v)}" for a, v in sorted((d.get("arrival") or {}).items())))
    cl = d.get("clouds") or {}
    for a, v in sorted(cl.items()):
        print(f"  cloud {a}: {counts(v)}")
    for a, r in sorted((d.get("per_attack") or {}).items()):
        pp = r.get("preprocess") or {}
        parts = []
        for dim, v in sorted((r.get("betti_l2") or {}).items()):
            parts.append(f"H{dim} L2={f(v.get('observed'), '.2f')} null={f(v.get('null_mean'), '.2f')} "
                         f"p={f(v.get('p_perm'))}")
        for dim, v in sorted((r.get("diagram_distance") or {}).items()):
            parts.append(f"H{dim} bottleneck={f(v.get('bottleneck'), '.3f')} p={f(v.get('bottleneck_p_perm'))}")
        print(f"  [{a}] pairs {pp.get('n_pairs')} (used {pp.get('n_pairs_used')}), pca {pp.get('pca_dim')} | "
              + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis_out")
    ap.add_argument("--sweep", action="store_true", help="include the RQ2 layer sweep")
    a = ap.parse_args()
    for lab in LABELS:
        print(f"\n######## {lab}")
        for kind, fn in (("stats", report_stats), ("rq2", None), ("homology", report_homology)):
            d, stamp = load(os.path.join(a.dir, f"{kind}_{lab}.json"))
            if d is None:
                print(f"== {kind}: {stamp}")
                continue
            print(f"== {kind} [{stamp}] {cfg(d.get('config'))}")
            try:
                if kind == "rq2":
                    report_rq2(d, a.sweep)
                else:
                    fn(d)
            except Exception as e:  # noqa: BLE001 -- a report must never hide the rest
                print(f"  (report error in {kind}: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
