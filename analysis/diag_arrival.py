#!/usr/bin/env python3
"""analysis/diag_arrival.py -- read-only diagnostic for two arrival exclusions
seen in the 10-29 RQ2/stats runs. Prints evidence only; changes nothing.

  delim: defended-clean vs defended-attacked pairs of one defence that
         arrival_step rejects as not_payload_arrival -- which step differs,
         check_arrival's reason, and the start of every differing message.
  vwa:   clean vs attacked pairs with status length_differs_no_diff -- step
         counts, the last shared completion, and the relative activation
         difference per shared turn.

  python3 analysis/diag_arrival.py delim final_agentdojo_gemma4 spotlighting_delimiting -n 3
  python3 analysis/diag_arrival.py vwa   final_vwa_gemma4 -n 3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
from experiments.attention_replay import (read_transcript, first_differing_step,  # noqa: E402
                                          check_arrival, step_conversation, _msg_text)
import geometry as g  # noqa: E402


def rows_of(d):
    return [r for r in json.load(open(os.path.join(d, "attack_results.json"))) if not r.get("error")]


def show(m, n=300):
    return repr(_msg_text(m)[:n])


def delim(d, defence, n):
    rows = rows_of(d)
    dclean = {r["task_id"]: r.get("deduplicated_to") or r["sample_id"] for r in rows
              if r.get("condition") == "defended_clean" and r.get("defense_name") == defence}
    shown = 0
    for r in rows:
        if r.get("condition") != "defended_attacked" or r.get("defense_name") != defence:
            continue
        dc = dclean.get(r["task_id"])
        da = r.get("deduplicated_to") or r["sample_id"]
        if dc is None:
            continue
        tc = read_transcript(Path(d) / f"{dc}.transcript.json.gz")
        ta = read_transcript(Path(d) / f"{da}.transcript.json.gz")
        if not tc or not ta:
            continue
        t = first_differing_step(tc, ta)
        why = check_arrival(tc, ta, t) if t is not None else "never differs"
        print(f"\n=== {da}\n    vs {dc}\n    steps {len(tc['steps'])}/{len(ta['steps'])}  first differing step {t}  reason: {why}")
        if t is not None:
            mc, ma = step_conversation(tc, t), step_conversation(ta, t)
            print(f"    messages at step {t}: {len(mc)} vs {len(ma)}")
            for i, (x, y) in enumerate(zip(mc, ma)):
                if x != y:
                    print(f"    [{i}] role {x.get('role')} / {y.get('role')}")
                    print(f"        clean   : {show(x)}")
                    print(f"        attacked: {show(y)}")
            if t > 0:
                pc, pa = tc["steps"][t - 1], ta["steps"][t - 1]
                print(f"    step {t-1} completion same: {pc.get('completion') == pa.get('completion')}, "
                      f"tool_calls same: {pc.get('tool_calls') == pa.get('tool_calls')}")
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print(f"no defended pairs found for {defence} in {d}")


def vwa(d, n):
    rows = rows_of(d)
    clean = {r["task_id"]: r["sample_id"] for r in rows if r.get("condition") == "clean"}
    shown = 0
    for r in rows:
        if r.get("condition") != "attacked" or r["task_id"] not in clean:
            continue
        c, a = clean[r["task_id"]], r["sample_id"]
        t, st = g.arrival_step(d, c, a, accept_step0=True)
        if st != "length_differs_no_diff":
            continue
        tc = read_transcript(Path(d) / f"{c}.transcript.json.gz")
        ta = read_transcript(Path(d) / f"{a}.transcript.json.gz")
        k = min(len(tc["steps"]), len(ta["steps"]))
        print(f"\n=== {a}  (asr {r.get('asr_success')})\n    steps clean {len(tc['steps'])} / attacked {len(ta['steps'])}")
        last = k - 1
        print(f"    last shared step {last} completion clean   : {tc['steps'][last].get('completion', '')[:160]!r}")
        print(f"    last shared step {last} completion attacked: {ta['steps'][last].get('completion', '')[:160]!r}")
        v = g.load_turns(d, {c: set(range(k)), a: set(range(k))}, g.ALL_LAYERS)
        rels = []
        for j in range(k):
            x, y = v.get((c, j)), v.get((a, j))
            if x is None or y is None:
                rels.append("-")
                continue
            sh = [L for L in x if L in y]
            rels.append(f"{max(np.linalg.norm(x[L] - y[L]) / max(np.linalg.norm(x[L]), 1e-12) for L in sh):.3g}")
        print(f"    max rel activation diff per shared turn: {rels}")
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print(f"no length_differs_no_diff pairs in {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["delim", "vwa"])
    ap.add_argument("results_dir")
    ap.add_argument("defence", nargs="?")
    ap.add_argument("-n", type=int, default=3)
    a = ap.parse_args()
    if a.mode == "delim":
        if not a.defence:
            ap.error("delim needs a defence name")
        delim(a.results_dir, a.defence, a.n)
    else:
        vwa(a.results_dir, a.n)


if __name__ == "__main__":
    main()
