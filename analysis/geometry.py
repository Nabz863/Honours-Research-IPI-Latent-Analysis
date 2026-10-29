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
from pathlib import Path as _P

# repo root on the path so `from experiments...` resolves however this is run
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

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
    s = str(task_id)
    m = re.search(r"^(.*user_task_\d+)", s)
    if m:
        return m.group(1)
    # VWA-Adv (cleanup-plan A2): 'reddit_36_wrong_object_cap_1' and its
    # '_2' sibling are variants of ONE base task, 'reddit_36' (the runner's
    # task.base). Without this every variant is its own group, so the probe
    # can score a held-out variant by its sibling and RQ2's per-task collapse
    # does nothing on VWA. Anything matching neither pattern is its own group.
    m = re.match(r"^(.*?_\d+)_", s)          # first '<...>_<N>' prefix; multi-word sites ok
    return m.group(1) if m else s


def _avg_ranks(a):
    """1-based ranks with ties given their average rank (scipy.stats.rankdata
    'average'), numpy-only so the effect size needs no scipy."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def matched_rank_biserial(diff):
    """Matched-pairs rank-biserial r = (R+ - R-) / (R+ + R-) over the ranks of
    the non-zero |differences| -- the effect size that goes with the Wilcoxon
    signed-rank test (Kerby 2014), as chapter 3 states. NOT the sign statistic
    (n+ - n-)/n, which ignores magnitude and can disagree with the test in sign."""
    d = np.asarray(diff, dtype=float)
    if not np.all(np.isfinite(d)):
        return None        # scipy's Wilcoxon gives p=nan here; never a silent number
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = _avg_ranks(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


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


def per_condition_geometry_by_outcome(paired):
    """RQ1's central question: do attacks that FAIL behaviourally still move
    the representation? Same ||dh||/normalised/cosine as per_condition_geometry,
    but split by the attacked episode's asr_success (True / False / None=unscored).
    Keyed attack -> outcome -> layer. The reportable claim is that failed
    attacks displace the state well above zero, so an injection leaves a
    signature even when the agent does not act on it."""
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for _, conds in paired.items():
        clean = conds[CLEAN_NAME]["acts"]
        for attack, entry in conds.items():
            if attack == CLEAN_NAME:
                continue
            # audit register: scoring_error_final rows are unscored, not failures
            s = normalise_asr(entry["row"])
            outcome = "success" if s is True else ("failure" if s is False else "unscored")
            for layer, h_adv in entry["acts"].items():
                h_cln = clean.get(layer)
                if h_cln is None or h_cln.shape != h_adv.shape:
                    continue
                d = h_adv - h_cln
                nd = float(np.linalg.norm(d))
                base = float(np.linalg.norm(h_cln))
                buckets[attack][outcome][layer].append(
                    (nd, nd / base if base else float("nan"), cosine(h_adv, h_cln)))

    out = {}
    for attack, outcomes in buckets.items():
        out[attack] = {}
        for outcome, layers in outcomes.items():
            out[attack][outcome] = {}
            for layer, vals in sorted(layers.items()):
                arr = np.array(vals, dtype=np.float64)
                out[attack][outcome][layer] = {
                    "n": int(arr.shape[0]),
                    "delta_mean": float(np.nanmean(arr[:, 0])),
                    "delta_std": float(np.nanstd(arr[:, 0])),
                    "delta_median": float(np.nanmedian(arr[:, 0])),
                    "delta_norm_mean": float(np.nanmean(arr[:, 1])),
                    "cosine_mean": float(np.nanmean(arr[:, 2])),
                }
    return out


def _load_turn(path, t):
    """{layer: np.float32[d]} for a single turn index, or None."""
    try:
        turns = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        turns = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(turns, list) or not (0 <= t < len(turns)) or not isinstance(turns[t], dict):
        return None
    return {int(L): v.detach().float().cpu().numpy() for L, v in turns[t].items()}


# ----------------------------------------------------------------------------
# Payload-arrival turn, shared by rq2_geometry / stats / persistent_homology so
# every analysis measures the same decision as the RQ1 headline below.
# ----------------------------------------------------------------------------

_SPOTLIGHT_KEY = re.compile(r"SPOTLIGHT-[0-9a-f]{16}")
# A timestamp the environment writes at RUN time (datetime.now(), microsecond
# precision), e.g. AgentDojo workspace's `last_modified` on a file the agent
# creates. Fixture data carries no microseconds, so only run-time stamps match.
_RUNTIME_TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d{6}")


def _mask_text(x):
    return _RUNTIME_TS.sub("<ts>", _SPOTLIGHT_KEY.sub("SPOTLIGHT-#", x))


def _mask_volatile(tr):
    """Copy of a transcript with the two run-dependent strings masked:
    - spotlighting_delimiting's delimiter key (SPOTLIGHT-<16 hex>), derived from
      the tool output's content and named in the tool result AND the system
      message, so the system message changes when the payload arrives
      (diag_arrival, 10-29: every delimiting pair was rejected for it);
    - run-time timestamps (_RUNTIME_TS): two episodes run hours apart differ in
      them although nothing the attacker wrote is involved (diag_exposure,
      10-29: 24 Qwen3-VL workspace_user_task_31 pairs per attack were 'reached'
      only through a file's last_modified time, with no payload in the prompt).
    Everything else is compared as recorded."""
    def blk(b):
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            return dict(b, text=_mask_text(b["text"]))
        return b
    def msg(m):
        if not isinstance(m, dict):
            return m
        c = m.get("content")
        if isinstance(c, str):
            return dict(m, content=_mask_text(c))
        if isinstance(c, list):
            return dict(m, content=[blk(b) for b in c])
        return m
    return dict(tr, messages=[msg(m) for m in tr.get("messages", [])])


def arrival_step(results_dir, sid_ref, sid_cmp, accept_step0=False):
    """(t, status) for two episodes of one task that differ only by the payload
    (clean vs attacked, or defended-clean vs defended-attacked under the same
    defence). t = the first step whose prompt differs in more than run-time
    noise (prompt_sha256 differs AND the conversations still differ with
    _mask_volatile applied), accepted as the payload's arrival by
    attention_replay.check_arrival on the masked transcripts: the episodes did
    the same at t-1 and every differing message is a tool result.
    accept_step0: a benchmark whose payload is in the FIRST observation (VWA:
    the page) differs at step 0 by construction; accept t = 0 there.
    status: ok | not_exposed (prompts identical throughout) | volatile_only
    (prompts differ only by run-time timestamps / keys) | no_transcript |
    not_payload_arrival | length_differs_no_diff."""
    from pathlib import Path as _Path
    from experiments.attention_replay import (read_transcript, first_differing_step,
                                              check_arrival, step_conversation)
    d = _Path(results_dir)
    tr = read_transcript(d / (sid_ref + ".transcript.json.gz"))
    tc = read_transcript(d / (sid_cmp + ".transcript.json.gz"))
    if tr is None or tc is None:
        return None, "no_transcript"
    t = first_differing_step(tr, tc)
    if t is None:
        same = len(tr.get("steps", [])) == len(tc.get("steps", []))
        return None, ("not_exposed" if same else "length_differs_no_diff")
    mr, mc = _mask_volatile(tr), _mask_volatile(tc)
    n = min(len(tr["steps"]), len(tc["steps"]))
    while t is not None and step_conversation(mr, t) == step_conversation(mc, t):
        t = next((k for k in range(t + 1, n)
                  if tr["steps"][k].get("prompt_sha256") != tc["steps"][k].get("prompt_sha256")), None)
    if t is None:
        # nothing but run-time noise differs on the shared steps; if the episodes
        # still end at different steps, they diverged after it -- the same
        # anomaly as identical prompts with different lengths, not "unexposed"
        same = len(tr["steps"]) == len(tc["steps"])
        return None, ("volatile_only" if same else "length_differs_no_diff")
    if t == 0 and accept_step0:
        return 0, "ok"
    if check_arrival(mr, mc, t) is not None:
        return None, "not_payload_arrival"
    return t, "ok"


ALL_LAYERS = range(0, 512)      # every layer a .pt may hold


def arrival_or_acts(results_dir, sid_ref, sid_cmp, layers=None, accept_step0=False,
                    abs_tol=1e-6, vwa_rel_tol=1e-2):
    """arrival_step, plus a check that 'not_exposed' really means unexposed.
    prompt_sha256 hashes the TEXT prompt only: a payload carried by an image
    leaves every text prompt identical while the activations differ. For a
    text-identical pair the two episodes are compared turn by turn on EVERY
    stored layer (not the caller's analysis layer, so every analysis
    classifies a pair identically -- review round 9; `layers` is ignored):
      AgentDojo (accept_step0 False): greedy decoding on one GPU makes an
        unexposed pair bit-identical, so ANY difference (> abs_tol) is
        anomalous -> (None, 'prompt_same_acts_differ'), excluded and counted.
      VWA (accept_step0 True): the first turn whose relative difference
        ||a-b|| / ||a|| (max over layers) exceeds vwa_rel_tol is the image
        payload's arrival -> (t, 'ok_acts'). vwa_rel_tol is a fixed guard
        against numerical noise, NOT calibrated on replicate clean runs
        (none exist on the final VWA data); stated as a limitation.
      no turn differs -> (None, 'not_exposed').
    'volatile_only' (text differs only by run-time timestamps / keys): AgentDojo
    -> (None, 'not_exposed') without the activation check; VWA -> scanned.
    VWA also scans 'length_differs_no_diff' pairs (identical text on every
    shared step, different episode lengths): an image payload can change the
    last shared decision without changing any text before it, so the episodes
    end at different steps. Scanned over the shared turns; no turn above
    vwa_rel_tol keeps the status. (AgentDojo keeps it as an anomaly: text-only
    greedy decoding cannot diverge with identical prompts.)
    Missing tensors -> (None, 'no_tensor')."""
    t, st = arrival_step(results_dir, sid_ref, sid_cmp, accept_step0)
    if st == "volatile_only":
        if not accept_step0:
            # text differs only by run-time timestamps / keys: the payload never
            # arrived. Unreached, like identical prompts -- but the activations
            # legitimately differ (the timestamp is in the prompt), so the
            # bit-identity check below does not apply.
            return None, "not_exposed"
        st = "not_exposed"          # VWA: the payload may be the image -> scan
    if st != "not_exposed" and not (accept_step0 and st == "length_differs_no_diff"):
        return t, st
    from pathlib import Path as _Path
    from experiments.attention_replay import read_transcript
    tr = read_transcript(_Path(results_dir) / (sid_ref + ".transcript.json.gz"))
    n = len(tr.get("steps", [])) if tr else 1   # a shorter sid_cmp stops the scan (missing turn -> break)
    v = load_turns(results_dir, {sid_ref: set(range(n)), sid_cmp: set(range(n))}, ALL_LAYERS)
    if (sid_ref, 0) not in v or (sid_cmp, 0) not in v:
        return None, "no_tensor"
    for k in range(n):
        a, b = v.get((sid_ref, k)), v.get((sid_cmp, k))
        if a is None or b is None:
            break
        shared = [L for L in a if L in b]
        if accept_step0:
            rel = max((np.linalg.norm(a[L] - b[L]) / max(np.linalg.norm(a[L]), 1e-12) for L in shared),
                      default=0.0)
            if rel > vwa_rel_tol:
                return k, "ok_acts"
        elif max((np.abs(a[L] - b[L]).max() for L in shared), default=0.0) > abs_tol:
            return None, "prompt_same_acts_differ"
    return None, st


def same_path(results_dir, sid_a, sid_b, t):
    """True if the two episodes made the same tool calls at every step before
    t (the defended and undefended trajectories reached step t the same way).
    None if a transcript is missing."""
    from pathlib import Path as _Path
    from experiments.attention_replay import read_transcript
    d = _Path(results_dir)
    ta = read_transcript(d / (sid_a + ".transcript.json.gz"))
    tb = read_transcript(d / (sid_b + ".transcript.json.gz"))
    if ta is None or tb is None:
        return None
    sa, sb = ta.get("steps", []), tb.get("steps", [])
    if len(sa) < t or len(sb) < t:
        return False
    # The action is the tool call where there is one (AgentDojo: a defence
    # changes the free text of the completion without changing the call);
    # only when neither step has tool calls (VWA) is the completion the action.
    def act(step):
        return step.get("tool_calls") if step.get("tool_calls") else ("completion", step.get("completion"))
    return all(act(sa[k]) == act(sb[k]) for k in range(t))


def normalise_asr(row):
    """The one ASR rule every analysis uses: True / False, or None = unscored.
    scoring_error_final rows are unscored (audit register); 0/1 integers are
    scores; anything else (missing, strings) is unscored."""
    if row.get("scoring_error_final"):
        return None
    v = row.get("asr_success")
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)) and v in (0, 1):
        return bool(v)
    return None


def load_turns(results_dir, needs, layers):
    """{(sid, t): {layer: np.float32[d]}} reading each .pt ONCE for all the
    turns and layers asked of it. needs: {sid: set(t)}. A missing file, an
    unreadable one, or a turn beyond the episode's length is simply absent."""
    layers = set(int(L) for L in layers)
    out = {}
    for sid, ts in needs.items():
        p = os.path.join(results_dir, sid + ".pt")
        try:
            try:
                turns = torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:
                turns = torch.load(p, map_location="cpu")
        except Exception:
            continue
        if not isinstance(turns, list):
            continue
        for t in ts:
            if 0 <= t < len(turns) and isinstance(turns[t], dict):
                v = {int(L): x.detach().float().cpu().numpy() for L, x in turns[t].items()
                     if int(L) in layers}
                if v:
                    out[(sid, t)] = v
    return out


def displacement_by_outcome_payload(results_dir, attacks=None, accept_step0=False):
    """RQ1 headline, measured at the PAYLOAD'S ARRIVAL turn, not turn 0.

    For each undefended attacked episode, the arrival turn t is the first step
    whose prompt differs from its clean pair's (prompt_sha256) -- the same
    localisation the attention replay uses. Up to t the two episodes are
    identical, so clean[t] vs attacked[t] isolates the payload. Turn 0 would be
    pre-injection (||dh||=0) for indirect injection, so turn-strategy loading is
    wrong for this metric; this reads the specific turn instead.

    Returns (summary attack->outcome->layer, exposure counts). An episode whose
    prompts never differ was never exposed and is counted, not bucketed."""
    # Reuse the attention replay's arrival-turn logic so the geometry and the
    # attention metrics localise the payload identically: the first step whose
    # prompt differs, ACCEPTED only when that difference is the payload (not
    # turn 0, not the agent's own prior output, only a differing tool result).
    from pathlib import Path as _Path
    from experiments.attention_replay import read_transcript, first_differing_step, check_arrival

    rows = load_rows(results_dir)
    clean_by_task = {r["task_id"]: r for r in rows
                     if r.get("condition") == "clean" and not r.get("error")
                     and r.get("defense_name") in (None, "none")}
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    exposure = defaultdict(lambda: {"exposed": 0, "not_exposed": 0,
                                    "not_payload_arrival": 0, "no_transcript": 0, "no_clean": 0})

    for r in rows:
        if r.get("condition") != "attacked" or r.get("error"):
            continue
        if r.get("defense_name") not in (None, "none"):
            continue
        attack = r.get("attack_name")
        if attacks and attack not in attacks:
            continue
        clean = clean_by_task.get(r.get("task_id"))
        if clean is None:
            exposure[attack]["no_clean"] += 1
            continue
        # the same pairing rq2_geometry / stats / persistent_homology use,
        # including image-only payloads (text identical, activations differ)
        t, st = arrival_or_acts(results_dir, clean["sample_id"], r["sample_id"],
                                accept_step0=accept_step0)
        if t is None:
            exposure[attack][{"length_differs_no_diff": "not_exposed"}.get(st, st)] = \
                exposure[attack].get({"length_differs_no_diff": "not_exposed"}.get(st, st), 0) + 1
            continue
        exposure[attack]["exposed"] += 1
        adv_t = _load_turn(tensor_path(r), t)
        cln_t = _load_turn(tensor_path(clean), t)
        if adv_t is None or cln_t is None:
            continue
        s = normalise_asr(r)
        outcome = "success" if s is True else ("failure" if s is False else "unscored")
        for layer, h_adv in adv_t.items():
            h_cln = cln_t.get(layer)
            if h_cln is None or h_cln.shape != h_adv.shape:
                continue
            d = h_adv - h_cln
            nd = float(np.linalg.norm(d))
            base = float(np.linalg.norm(h_cln))
            buckets[attack][outcome][layer].append(
                (nd, nd / base if base else float("nan"), cosine(h_adv, h_cln)))

    summary = {}
    for attack, outcomes in buckets.items():
        summary[attack] = {}
        for outcome, layers in outcomes.items():
            summary[attack][outcome] = {}
            for layer, vals in sorted(layers.items()):
                arr = np.array(vals, dtype=np.float64)
                summary[attack][outcome][layer] = {
                    "n": int(arr.shape[0]),
                    "delta_mean": float(np.nanmean(arr[:, 0])),
                    "delta_median": float(np.nanmedian(arr[:, 0])),
                    "delta_std": float(np.nanstd(arr[:, 0])),
                    "delta_norm_mean": float(np.nanmean(arr[:, 1])),
                    "cosine_mean": float(np.nanmean(arr[:, 2])),
                }
    return summary, {k: dict(v) for k, v in exposure.items()}


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
    ap.add_argument("--accept_step0", action="store_true",
                    help="payload is in the first observation (VWA): accept arrival at step 0")
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
    by_outcome = per_condition_geometry_by_outcome(paired)
    probe = {} if args.skip_probe else probe_separability(paired)

    # RQ1 headline at the payload's arrival turn (per directory, needs transcripts).
    payload_outcome, payload_exposure = {}, {}
    for d in args.results_dir:
        s, e = displacement_by_outcome_payload(d, accept_step0=args.accept_step0)
        if s or e:
            payload_outcome[d] = s
            payload_exposure[d] = e

    layers = sorted({L for c in paired.values() for L in c[CLEAN_NAME]["acts"]})
    pca_layer = args.pca_layer if args.pca_layer is not None else layers[-1]
    pca = pca_coords(paired, pca_layer)

    payload = {
        "config": vars(args),
        "n_episodes_paired": len(paired),
        "layers": layers,
        "geometry": geom,
        "geometry_by_outcome": by_outcome,
        "geometry_by_outcome_payload_turn": payload_outcome,
        "payload_exposure": payload_exposure,
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

    # RQ1 headline: failed attacks still displace. Deepest layer, per attack.
    oc_csv = os.path.join(args.out_dir, "geometry_by_outcome.csv")
    with open(oc_csv, "w") as f:
        f.write("attack,outcome,layer,n,delta_mean,delta_median,delta_norm_mean,cosine_mean\n")
        for attack, outcomes in sorted(by_outcome.items()):
            for outcome, layers_d in sorted(outcomes.items()):
                for layer, m in sorted(layers_d.items()):
                    f.write("%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f\n" % (
                        attack, outcome, layer, m["n"], m["delta_mean"],
                        m["delta_median"], m["delta_norm_mean"], m["cosine_mean"]))

    # RQ1 headline CSV: displacement at the payload-arrival turn, by outcome.
    pt_csv = os.path.join(args.out_dir, "displacement_payload_turn.csv")
    with open(pt_csv, "w") as f:
        f.write("results_dir,attack,outcome,layer,n,delta_mean,delta_median,delta_norm_mean,cosine_mean\n")
        for d, summ in sorted(payload_outcome.items()):
            for attack, outcomes in sorted(summ.items()):
                for outcome, layers_d in sorted(outcomes.items()):
                    for layer, m in sorted(layers_d.items()):
                        f.write("%s,%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f\n" % (
                            os.path.basename(d.rstrip("/")), attack, outcome, layer, m["n"],
                            m["delta_mean"], m["delta_median"], m["delta_norm_mean"], m["cosine_mean"]))

    print("\n=== RQ1: displacement at the payload-arrival turn (deepest layer) ===")
    for d, summ in sorted(payload_outcome.items()):
        exp = payload_exposure.get(d, {})
        print("  [%s]" % os.path.basename(d.rstrip("/")))
        for attack in sorted(summ):
            deep = max((L for oc in summ[attack].values() for L in oc), default=None)
            if deep is None:
                continue
            parts = []
            for outcome in ("success", "failure", "unscored"):
                m = summ[attack].get(outcome, {}).get(deep)
                if m:
                    parts.append("%s n=%d ||dh||=%.2f (med %.2f)" % (
                        outcome, m["n"], m["delta_mean"], m["delta_median"]))
            e = exp.get(attack, {})
            print("    %-24s %s  [exposed %s, not %s]" % (
                attack, " | ".join(parts), e.get("exposed", "?"), e.get("not_exposed", "?")))

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
    print("\nwrote %s\n      %s\n      %s\n      %s" % (out, csv, oc_csv, pt_csv))


if __name__ == "__main__":
    main()
