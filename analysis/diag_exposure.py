#!/usr/bin/env python3
"""analysis/diag_exposure.py -- read-only cross-check of the two exposure
definitions on AgentDojo, undefended attacked episodes:

  reach   (geometry/stats): the attacked episode's prompt first differs from the
          clean episode's at a step accepted as the payload's arrival
          (geometry.arrival_or_acts); 'not_exposed' = prompts identical.
  matcher (agentdojo_behaviour.py, Table "compliance"): a 40-character
          normalised marker of the attacker-written text appears in the
          attacked transcript.

Prints the 2x2 table per attack (with the behaviour script's exclusion and
denominator rules applied, as in its table), counts of each disagreement by
suite, and for up to -n disagreeing episodes: the arrival step, the markers,
and the start of every message that differs from the clean episode there.

  python3 analysis/diag_exposure.py final_agentdojo_qwen3vl final_agentdojo_gemma4 -n 6
(the exclusion set is the union over ALL directories given, as in the table)
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import agentdojo_behaviour as ab  # noqa: E402
import geometry as g  # noqa: E402
from experiments.attention_replay import read_transcript, step_conversation, _msg_text  # noqa: E402


def classify(run, defaults, excluded):
    """[(row, reach_status, arrival_t, matcher_exposed or None, markers)]"""
    rows = [r for r in run["rows"] if ab.pair_key(r) not in excluded]
    clean = {r["task_id"]: r["sample_id"] for r in rows
             if r.get("condition") == "clean" and not r.get("error")}
    out = []
    for r in rows:
        if r.get("condition") != ab.ATTACKED or r.get("error") or (r.get("defense_name") or "none") != "none":
            continue
        if r.get("asr_success") is None:
            continue                                   # the table's denominator rule
        entry = run["payloads"].get((r.get("attack_name"), r.get("task_id")))
        marks = ab.payload_markers(entry, r.get("suite_name"), defaults) if entry else []
        txt = ab.transcript_text(run["dir"] / f"{r['sample_id']}.transcript.json.gz")
        exposed = None if (txt is None or not marks) else any(m in txt for m in marks)
        c = clean.get(r["task_id"])
        if c is None:
            t, st = None, "no_clean"
        else:
            t, st = g.arrival_or_acts(str(run["dir"]), c, r["sample_id"])
        out.append((r, st, t, exposed, marks, c))
    return out


def marker_at_arrival(d, c_sid, a_sid, t, marks):
    """True if a payload marker is in a message that differs at step t."""
    tc = read_transcript(Path(d) / f"{c_sid}.transcript.json.gz")
    ta = read_transcript(Path(d) / f"{a_sid}.transcript.json.gz")
    if t is None or not tc or not ta:
        return None
    diff = [y for x, y in zip(step_conversation(tc, t), step_conversation(ta, t)) if x != y]
    txt = ab.norm_rendered(" ".join(_msg_text(y) for y in diff))
    return any(m in txt for m in marks)


def show_diff(d, c_sid, a_sid, t, marks):
    tc = read_transcript(Path(d) / f"{c_sid}.transcript.json.gz")
    ta = read_transcript(Path(d) / f"{a_sid}.transcript.json.gz")
    print(f"    markers: {marks}")
    if t is None or not tc or not ta:
        return
    for i, (x, y) in enumerate(zip(step_conversation(tc, t), step_conversation(ta, t))):
        if x != y:
            a, b = _msg_text(x), _msg_text(y)
            k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print(f"    step {t} msg [{i}] role {y.get('role')}: first difference at char {k} "
                  f"(lengths {len(a)}/{len(b)})")
            print(f"      clean   : {a[max(0, k - 60):k + 200]!r}")
            print(f"      attacked: {b[max(0, k - 60):k + 300]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-n", type=int, default=6, help="disagreeing episodes to print per kind and directory")
    a = ap.parse_args()
    runs = [ab.load_dir(Path(d)) for d in a.dirs]
    defaults = ab.injection_defaults()
    excluded = set()
    for run in runs:
        for r in run["rows"]:
            if r.get("condition") == "clean" and not r.get("error") and r.get("asr_success") is True:
                excluded.add(ab.pair_key(r))
    print(f"excluded pairs (union over {len(runs)} dirs): {len(excluded)}")
    for run in runs:
        res = classify(run, defaults, excluded)
        print(f"\n######## {run['name']}")
        by_attack = defaultdict(list)
        for x in res:
            by_attack[x[0].get("attack_name")].append(x)
        for atk, xs in sorted(by_attack.items()):
            tab = Counter((x[1] == "ok", x[3]) for x in xs)
            stat = Counter(x[1] for x in xs)
            checked = [x for x in xs if x[3] is not None]
            print(f"\n== {atk}: {len(xs)} scored episodes; matcher checked {len(checked)}; reach statuses {dict(stat)}")
            print(f"   matcher exposed {sum(1 for x in checked if x[3])}/{len(checked)}; "
                  f"reached {sum(1 for x in xs if x[1] == 'ok')}/{len(xs)}")
            print("   reached & exposed {}, reached & NOT exposed {}, NOT reached & exposed {}, "
                  "neither {}, matcher unchecked {}".format(
                      tab[(True, True)], tab[(True, False)], tab[(False, True)], tab[(False, False)],
                      tab[(True, None)] + tab[(False, None)]))
            both = [x for x in xs if x[1] == "ok" and x[3]]
            at = [marker_at_arrival(run["dir"], x[5], x[0]["sample_id"], x[2], x[4]) for x in both]
            wrong = [x for x, ok in zip(both, at) if ok is False]
            print(f"   reached & exposed: payload marker in the differing message at the arrival step "
                  f"{sum(1 for ok in at if ok)}/{len(both)}; NOT there (arrival step is something else) {len(wrong)}"
                  + (f" by suite {dict(Counter(x[0].get('suite_name') for x in wrong))}" if wrong else ""))
            for x in wrong[: a.n]:
                print(f"  -- {x[0]['sample_id']}  (arrival step {x[2]})")
                show_diff(run["dir"], x[5], x[0]["sample_id"], x[2], x[4])
            for kind, sel in (("reached but NOT matched", lambda x: x[1] == "ok" and x[3] is False),
                              ("matched but NOT reached", lambda x: x[1] != "ok" and x[3] is True)):
                d = [x for x in xs if sel(x)]
                if not d:
                    continue
                print(f"   {kind}: {len(d)}  by suite {dict(Counter(x[0].get('suite_name') for x in d))}"
                      f"  ASR among them {sum(bool(x[0]['asr_success']) for x in d)}")
                for x in d[: a.n]:
                    print(f"  -- {x[0]['sample_id']}  (status {x[1]}, arrival step {x[2]})")
                    show_diff(run["dir"], x[5], x[0]["sample_id"], x[2], x[4])


if __name__ == "__main__":
    main()
