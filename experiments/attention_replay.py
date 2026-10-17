#!/usr/bin/env python3
"""
experiments/attention_replay.py -- attention entropy and deflection angle for
the final AgentDojo runs, by replaying the recorded prompts (no generation).

WHAT IS REPLAYED
    For every undefended attacked episode (template_ipi, important_instructions)
    and its pair's clean episode: the FIRST step whose prompt differs between
    the two (prompt_sha256 in the transcripts). Greedy decoding makes the
    trajectories identical until the payload arrives, so at that step the two
    prompts differ only by the injected content -- the one point where clean
    and attacked attention can be compared key by key. An episode whose
    prompts never differ was never exposed and is recorded as such.

    The prompt is rebuilt exactly as the run built it (TranscriptRecorder:
    processor.apply_chat_template(messages, add_generation_prompt=True,
    tokenize=False)) and refused unless its sha256 equals the recorded one.
    One forward pass, the run's own eager attention. As a second check, the
    last-token hidden states of that pass are compared with the activations
    the run saved for that turn.

METRICS, per decoder layer and head, at the last prompt token (the position
the decision is generated from):
    entropy        Shannon entropy of the attention over the keys the query
                   can see, normalised by log(#visible keys) (sliding-window
                   layers: the window), as in ICON's recorder
    payload_mass   attention on the injected tokens (attacked prompt)
    deflection     angle between the clean and attacked attention patterns,
                   on the union of their keys aligned token by token (difflib
                   on token ids): equal tokens share a coordinate, injected
                   tokens are 0 in the clean pattern and removed ones 0 in the
                   attacked pattern. 0 deg = the same pattern; capture by the
                   payload or redistribution both raise it

    deflection_shared  the same angle restricted to keys visible in BOTH
                   prompts and renormalised there (redistribution with the
                   payload's capture taken out). Needed in sliding-window
                   layers (Gemma 4: 5 of 6), where the injected tokens push
                   shared keys out of the window and the whole-row angle
                   would measure position; n_shared_visible is recorded

The first differing step is accepted as the payload's arrival only if the
two episodes did the same thing at the step before and every message that
differs is a tool result; anything else is recorded and skipped.

OUTPUT  <out_dir>/<sample_id>.attn.pt per attacked episode (resumable: an
        existing file is skipped), and <out_dir>/replay_log.jsonl.

    python experiments/attention_replay.py --model gemma4 --hpc \
        --run_dir final_agentdojo_gemma4 --out_dir attn_replay_gemma4
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("attention_replay")

ATTACKS = ("important_instructions", "template_ipi")
_EPS = 1e-12
ACT_TOL = 1e-2     # median relative difference to the saved activations above which the replay stops


# ---------------------------------------------------------------- transcripts
def read_transcript(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(gzip.decompress(p.read_bytes()))
    except Exception:
        return None


def step_conversation(tr: Dict[str, Any], t: int) -> List[Any]:
    return [tr["messages"][i] for i in tr["steps"][t]["input"]]


def build_prompt(processor, tr: Dict[str, Any], t: int) -> str:
    return processor.apply_chat_template(step_conversation(tr, t), add_generation_prompt=True, tokenize=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _msg_text(m: Any) -> str:
    return " ".join(b.get("text", "") for b in (m.get("content") or []) if isinstance(b, dict))


def check_arrival(tc: Dict[str, Any], ta: Dict[str, Any], t: int) -> Optional[str]:
    """None if step t's difference is the payload arriving: the two episodes
    did the same at t-1 (completion and tool calls), and every message that
    differs at t is a tool result. Otherwise the reason it is not."""
    if t == 0:
        return "differs at step 0 (before any tool result)"
    pc, pa = tc["steps"][t - 1], ta["steps"][t - 1]
    if pc.get("completion") != pa.get("completion") or pc.get("tool_calls") != pa.get("tool_calls"):
        return "the model's own output differs at the previous step"
    mc, ma = step_conversation(tc, t), step_conversation(ta, t)
    if len(mc) != len(ma):
        return "different number of messages"
    for x, y in zip(mc, ma):
        if x != y and not _msg_text(y).startswith(("[TOOL RESULT]", "[TOOL ERROR]")):
            return f"a non-tool message differs (role {y.get('role')})"
    return None


def first_differing_step(tc: Dict[str, Any], ta: Dict[str, Any]) -> Optional[int]:
    """First step whose prompt differs; None if the attacked episode's prompts
    all equal the clean ones (never exposed). Steps beyond the clean episode's
    end are not comparable."""
    for t, (sc, sa) in enumerate(zip(tc["steps"], ta["steps"])):
        if sc.get("prompt_sha256") != sa.get("prompt_sha256"):
            return t
    return None


# ---------------------------------------------------------------- metrics
def entropy(p: torch.Tensor, n_visible: int) -> torch.Tensor:
    """p: [H, K] attention rows (sum to 1 over visible keys)."""
    p = p.float()
    p = p / p.sum(-1, keepdim=True).clamp_min(_EPS)
    ent = -(p * torch.log(p + _EPS)).sum(-1)
    return ent / math.log(max(n_visible, 2))


def align(clean_ids: List[int], adv_ids: List[int]):
    """Token alignment: (matched clean idx, matched adv idx, adv-only idx, clean-only idx)."""
    sm = difflib.SequenceMatcher(None, clean_ids, adv_ids, autojunk=False)
    mc, ma, oa, oc = [], [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            mc += range(i1, i2)
            ma += range(j1, j2)
        else:
            oc += range(i1, i2)
            oa += range(j1, j2)
    return (torch.tensor(mc, dtype=torch.long), torch.tensor(ma, dtype=torch.long),
            torch.tensor(oa, dtype=torch.long), torch.tensor(oc, dtype=torch.long))


def shared_visible(mc, ma, kc: int, ka: int, window: Optional[int]):
    """Matched key pairs visible from the last query in BOTH prompts. In a
    sliding-window layer the injected tokens push shared keys out of the
    attacked prompt's window, so comparing whole rows there measures
    position, not attention."""
    if not window:
        return mc, ma
    keep = (mc >= kc - window) & (ma >= ka - window)
    return mc[keep], ma[keep]


def deflection_shared(pc: torch.Tensor, pa: torch.Tensor, mc, ma) -> torch.Tensor:
    """Angle (degrees) between the two patterns restricted to the shared,
    visible keys, each renormalised there: redistribution among the keys both
    prompts contain, with the payload's capture taken out (payload_mass
    reports that). NaN where no shared key is visible."""
    if len(mc) == 0:
        return torch.full((pc.shape[0],), float("nan"))
    x = pc.float()[:, mc]
    y = pa.float()[:, ma]
    x = x / x.sum(-1, keepdim=True).clamp_min(_EPS)
    y = y / y.sum(-1, keepdim=True).clamp_min(_EPS)
    cos = (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(_EPS)
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def deflection(pc: torch.Tensor, pa: torch.Tensor, mc, ma) -> torch.Tensor:
    """Angle (degrees) between clean [H, Kc] and attacked [H, Ka] patterns on
    the union of keys; only matched keys contribute to the dot product."""
    pc, pa = pc.float(), pa.float()
    dot = (pc[:, mc] * pa[:, ma]).sum(-1)
    cos = dot / (pc.norm(dim=-1) * pa.norm(dim=-1)).clamp_min(_EPS)
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


# ---------------------------------------------------------------- recorder
class LastRowRecorder:
    """Wraps the model file's eager_attention_forward (the function the run
    used) and keeps, per decoder layer, the attention row of the LAST query
    position of a prefill call: [H, K]. Text decoder modules only (the
    vision tower calls the same function in the Qwen-VL files)."""

    def __init__(self, model):
        from defenses.icon_runtime import find_attention_module, text_attention_ids, assert_eager
        assert_eager(model)
        self.mod = find_attention_module(model)
        self.scope = text_attention_ids(model)
        self.rows: Dict[int, torch.Tensor] = {}
        self.windows: Dict[int, Optional[int]] = {}
        self._orig = None

    def __enter__(self):
        base = self.mod.eager_attention_forward
        rec = self

        def recording(module, query, key, value, *args, **kwargs):
            out, probs = base(module, query, key, value, *args, **kwargs)
            if probs is not None and id(module) in rec.scope and probs.shape[2] > 1:
                li = getattr(module, "layer_idx", None)
                if li is not None:
                    rec.rows[int(li)] = probs[0, :, -1, :].detach().float().cpu()
                    w = getattr(module, "sliding_window", None)
                    rec.windows[int(li)] = w if isinstance(w, int) and not isinstance(w, bool) and w > 0 else None
            return out, probs

        self._orig = base
        self.mod.eager_attention_forward = recording
        self.rows, self.windows = {}, {}
        return self

    def __exit__(self, *exc):
        self.mod.eager_attention_forward = self._orig
        return False


def visible(k: int, window: Optional[int]) -> int:
    return k if not window else max(1, min(k, window))


# ---------------------------------------------------------------- replay
def forward(model, processor, extractor, prompt: str):
    device = next(model.parameters()).device
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    rec = LastRowRecorder(model)
    with extractor:
        with rec, torch.no_grad():
            try:                      # logits for the last position only: the decision
                model(**inputs, logits_to_keep=1)
            except TypeError:
                model(**inputs)
        acts = extractor.get_activations()
    extractor.clear()
    torch.cuda.empty_cache()
    if not rec.rows:
        raise RuntimeError("recorded no attention: the eager patch did not take effect")
    return inputs["input_ids"][0].tolist(), rec.rows, rec.windows, acts


def act_check(acts: Dict[int, torch.Tensor], saved: Optional[Dict[int, torch.Tensor]]) -> Optional[float]:
    """Largest relative difference between the replayed and saved last-token
    hidden states of this turn, over layers."""
    if not saved:
        return None
    worst = 0.0
    for k, v in acts.items():
        if k in saved:
            a, b = v.float().flatten(), saved[k].float().flatten()
            if a.shape != b.shape:
                return float("inf")
            worst = max(worst, ((a - b).norm() / b.norm().clamp_min(_EPS)).item())
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gemma4", "qwen3vl", "llava", "internvl", "qwen2vl"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hpc", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="stop after N episodes (smoke test)")
    a = ap.parse_args()

    run = Path(a.run_dir)
    rows = json.loads((run / "attack_results.json").read_text())
    by_sid = {r["sample_id"]: r for r in rows}
    excluded = {r["task_id"] for r in rows if r.get("condition") == "clean" and not r.get("error")
                and r.get("asr_success") is True}
    def clean_ok(r):
        c = by_sid.get(f"clean_{r['task_id']}")
        return c is not None and not c.get("error")
    todo = [r for r in rows if r.get("condition") == "attacked" and r.get("attack_name") in ATTACKS
            and not r.get("error") and r.get("task_id") not in excluded and clean_ok(r)]
    # pairs together, so a clean prompt is replayed once for both attacks
    todo.sort(key=lambda r: (r["task_id"], r["attack_name"]))
    if a.limit:
        todo = todo[: a.limit]
    log.info("%d attacked episodes selected (%d pairs excluded)", len(todo), len(excluded))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "replay_log.jsonl", "a")

    from experiments.run_attack_matrix import _load_model_by_name
    from extraction.hooks import ActivationExtractor, all_layers
    model, processor = _load_model_by_name(a.model, a.hpc)
    model.eval()
    extractor = ActivationExtractor(model, target_layers=all_layers(model), token_aggregation="last")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log.info("model %s on %s", a.model, gpu)

    clean_cache: Dict[str, Tuple] = {}
    done = skipped = mismatches = 0
    checks: List[float] = []
    t0 = time.time()
    for i, r in enumerate(todo):
        sid = r["sample_id"]
        dest = out / f"{sid}.attn.pt"
        if dest.exists():
            skipped += 1
            continue
        csid = f"clean_{r['task_id']}"
        tc = read_transcript(run / f"{csid}.transcript.json.gz")
        ta = read_transcript(run / f"{sid}.transcript.json.gz")
        rec: Dict[str, Any] = {"sample_id": sid, "clean_sid": csid, "attack_name": r["attack_name"],
                               "task_id": r["task_id"], "suite_name": r.get("suite_name"),
                               "asr_success": r.get("asr_success"), "utility_success": r.get("utility_success"),
                               "gpu_name": gpu}
        if not tc or not ta:
            rec["status"] = "no_transcript"
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            continue
        t = first_differing_step(tc, ta)
        rec["step"] = t
        if t is None:
            # identical prompts at every shared step; a length difference
            # with no differing prompt would be a non-determinism, not exposure
            rec["status"] = "not_exposed" if len(tc["steps"]) == len(ta["steps"]) else "length_differs_no_diff"
            torch.save(rec, dest)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            continue
        why = check_arrival(tc, ta, t)
        if why:
            rec["status"], rec["reason"] = "not_payload_arrival", why
            torch.save(rec, dest)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log.warning("%s: step %d is not the payload's arrival (%s) -- skipped", sid, t, why)
            continue
        pc_txt, pa_txt = build_prompt(processor, tc, t), build_prompt(processor, ta, t)
        if sha(pc_txt) != tc["steps"][t]["prompt_sha256"] or sha(pa_txt) != ta["steps"][t]["prompt_sha256"]:
            rec["status"] = "prompt_mismatch"
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log.error("%s: rebuilt prompt does not match the recorded sha256 -- skipped", sid)
            mismatches += 1
            if mismatches >= 5 and done == 0:
                log.error("FATAL: the first %d prompts could not be rebuilt -- the processor or template "
                          "differs from the run's; stopping", mismatches)
                return 1
            continue
        try:
            key = sha(pc_txt)
            if key not in clean_cache:
                ids_c, rows_c, win_c, acts_c = forward(model, processor, extractor, pc_txt)
                saved_c = torch.load(run / f"{csid}.pt", weights_only=False)
                if len(clean_cache) >= 4:      # episodes are sorted by pair: a few recent clean prompts suffice
                    clean_cache.pop(next(iter(clean_cache)))
                clean_cache[key] = (ids_c, rows_c, win_c, act_check(acts_c, saved_c[t] if t < len(saved_c) else None))
            ids_c, rows_c, win_c, chk_c = clean_cache[key]
            ids_a, rows_a, win_a, acts_a = forward(model, processor, extractor, pa_txt)
            saved_a = torch.load(run / f"{sid}.pt", weights_only=False)
            chk_a = act_check(acts_a, saved_a[t] if t < len(saved_a) else None)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rec["status"] = "oom"
            rec["error"] = str(e).splitlines()[0][:200]
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            continue
        mc, ma, oa, oc = align(ids_c, ids_a)
        layers = sorted(set(rows_c) & set(rows_a))
        ent_c, ent_a, pmass, defl, defl_sh, n_sh = [], [], [], [], [], []
        for li in layers:
            pc, pa = rows_c[li], rows_a[li]
            w = win_a.get(li)
            ent_c.append(entropy(pc, visible(pc.shape[-1], win_c.get(li))))
            ent_a.append(entropy(pa, visible(pa.shape[-1], w)))
            pmass.append(pa.float()[:, oa].sum(-1) if len(oa) else torch.zeros(pa.shape[0]))
            defl.append(deflection(pc, pa, mc, ma))
            sc, sa = shared_visible(mc, ma, pc.shape[-1], pa.shape[-1], w)
            n_sh.append(len(sc))
            defl_sh.append(deflection_shared(pc, pa, sc, sa))
        rec.update(status="ok", layers=layers, n_tokens_clean=len(ids_c), n_tokens_adv=len(ids_a),
                   n_injected_tokens=len(oa), n_removed_tokens=len(oc),
                   act_check_clean=chk_c, act_check_adv=chk_a, n_shared_visible=n_sh,
                   windows={int(k): v for k, v in win_a.items()})
        torch.save({**rec, "entropy_clean": torch.stack(ent_c).half(), "entropy_adv": torch.stack(ent_a).half(),
                    "payload_mass": torch.stack(pmass).half(), "deflection_deg": torch.stack(defl).half(),
                    "deflection_shared_deg": torch.stack(defl_sh).half()}, dest)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        done += 1
        checks.extend(x for x in (chk_c, chk_a) if x is not None)
        if done == 20:
            med = sorted(checks)[len(checks) // 2] if checks else None
            if med is None or med > ACT_TOL:
                log.error("FATAL: replayed hidden states differ from the run's saved activations (median "
                          "relative difference %s > %s): the replay is not reproducing the run's inputs or "
                          "hardware; stopping", med, ACT_TOL)
                return 1
            log.info("activation check over the first 20 episodes: median relative difference %.2e", med)
        if done % 25 == 0:
            el = time.time() - t0
            log.info("%d/%d done (%d skipped) -- %.1fs/episode, act check %.2e/%.2e",
                     done + skipped, len(todo), skipped, el / done, chk_c or 0, chk_a or 0)
    log.info("finished: %d replayed, %d already done, of %d", done, skipped, len(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
