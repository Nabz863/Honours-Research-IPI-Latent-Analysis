"""
experiments/episode_utils.py
----------------------------
Episode helpers shared by BOTH runners (run_attack_matrix.py for AgentDojo,
run_vwa_adv_curated_matrix.py for VWA-Adv), moved here unchanged from
run_attack_matrix.py (review 4, M3). Before the move, the VWA runner
imported them from the AgentDojo runner, so the AgentDojo runner's source
was part of the VWA harness hash and any AgentDojo-only edit blocked VWA
resumes. Also: atomic file writes (review 4, H2).

FROZEN for the final runs: every row's harness hash covers this file.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic writes (review 4, H2): a walltime kill mid-write must leave either
# the old file or the new one, never a truncated file that breaks every
# resume. Write to a temp file in the SAME directory, fsync, os.replace.
# ---------------------------------------------------------------------------
_UMASK = os.umask(0); os.umask(_UMASK)      # read once, at import


def _atomic(path, write_fn) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        # mkstemp creates 0600; give the file the permissions an ordinary
        # open() would have (umask-respecting), as before (review 5, low).
        try:
            os.fchmod(fd, 0o666 & ~_UMASK)
        except OSError:
            pass                            # some network filesystems refuse; not fatal
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path, text: str) -> None:
    _atomic(path, lambda f: f.write(text.encode("utf-8")))


def atomic_write_json(path, obj, **kw) -> None:
    atomic_write_text(path, json.dumps(obj, **kw))


def atomic_write_json_gz(path, obj) -> None:
    """gzip-compressed JSON (per-step transcripts, P2a). mtime=0 so the same
    transcript always produces the same bytes."""
    import gzip
    data = gzip.compress(json.dumps(obj, default=str).encode("utf-8"), compresslevel=6, mtime=0)
    _atomic(path, lambda f: f.write(data))


def acquire_dir_lock(output_dir) -> None:
    """One runner process per output directory (P2a.1 review). Each process
    rewrites attack_results.json from its own in-memory rows, so two at once
    erase each other's. The lock records the SLURM job; a lock left by a job
    that is no longer queued or running is taken over, a live one stops this
    process before it touches anything. Released at exit."""
    import atexit
    import subprocess
    lock = Path(output_dir) / ".runner.lock"
    me = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as f:
                f.write(f"{me} {os.uname().nodename} {os.getpid()}\n")
            atexit.register(lambda: lock.unlink() if lock.exists() and
                            lock.read_text().split()[:1] == [me] else None)
            return
        except FileExistsError:
            parts = lock.read_text().split() or ["?"]
            owner = parts[0]
            if owner == me:
                return                               # this process (or job) already holds it
            alive = None
            if owner.startswith("pid") and len(parts) >= 3 and parts[1] == os.uname().nodename:
                try:
                    os.kill(int(parts[2]), 0); alive = True
                except ProcessLookupError:
                    alive = False
                except Exception:
                    alive = None
            elif owner.isdigit():
                # alive only on a clear answer; a transient squeue failure is
                # retried, and only "invalid job id" / empty output means gone
                for _try in range(3):
                    try:
                        r = subprocess.run(["squeue", "-h", "-j", owner, "-o", "%T"],
                                           capture_output=True, text=True, timeout=30)
                    except FileNotFoundError:
                        alive = None; break
                    except Exception:
                        alive = None
                    else:
                        if r.returncode == 0:
                            alive = bool(r.stdout.strip()); break
                        if "invalid job id" in (r.stderr or "").lower():
                            alive = False; break
                        alive = True                 # squeue erred: assume live, retry
                    import time as _t
                    _t.sleep(10)
            if alive is False or (alive is None and owner.isdigit()):
                # finished job, or squeue unavailable on this node: a resubmit
                # after a walltime kill must not be blocked by its own old lock
                logger.warning("taking over the lock from job %s (%s; %s)", owner, lock,
                               "no longer queued" if alive is False else "state unknown")
                lock.unlink()
                continue
            raise SystemExit(
                f"FATAL: {output_dir} is in use by job {owner} ({lock}). Two runners on one "
                "directory erase each other's rows. Wait for it, or delete the lock if that job "
                "is certainly gone.")
    raise SystemExit(f"FATAL: could not take {lock}")


def atomic_torch_save(obj, path) -> None:
    _atomic(path, lambda f: torch.save(obj, f))


def atomic_save_image(image, path, format: str = "PNG") -> None:
    _atomic(path, lambda f: image.save(f, format=format))


@contextlib.contextmanager
def _cuda_memory_guard():
    """
    Wraps ONE episode's execution: resets peak memory stats before,
    exposes a dict the caller reads peak_gb from after the `with` block,
    and unconditionally empty_cache()s + gc.collect()s afterward --
    belt-and-suspenders on top of LocalVLMPipelineElement's own per-TURN
    cleanup, targeting any cross-EPISODE accumulation (agentdojo's own
    pipeline/environment objects, or Python reference cycles that plain
    refcounting doesn't collect -- gc.collect() specifically exists for
    that) that per-turn cleanup wouldn't reach.

    Real motivation, not preemptive: job 23527 hit a genuine CUDA OOM
    roughly 40 episodes into a 629-task run, inside an attention softmax
    call (21.87 GiB in use on a 24GB card before a 2.36 GiB allocation
    failed). Two honest explanations are both physically plausible --
    memory accumulating across episodes that per-turn cleanup doesn't
    fully reach, OR that specific task's conversation simply being long
    enough to legitimately need that much -- and there wasn't enough
    data to tell which. peak_cuda_memory_gb (logged per row below) is
    what actually answers that on the next run, rather than guessing.
    """
    stats: Dict[str, Optional[float]] = {"peak_gb": None}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield stats
    finally:
        if torch.cuda.is_available():
            try:
                stats["peak_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                # A poisoned CUDA context (e.g. after a device-side assert)
                # can make even cleanup calls raise -- letting that REPLACE
                # whatever real exception was already propagating is exactly
                # backwards. Confirmed directly on a real run: empty_cache()
                # here raised its own AcceleratorError, masking the actual
                # device-side-assert traceback underneath a confusing second
                # one. Log and swallow the cleanup failure instead.
                logger.warning(
                    "Cleanup inside _cuda_memory_guard failed (%s: %s) -- swallowing so it "
                    "doesn't mask whatever exception triggered this finally block.",
                    type(cleanup_error).__name__, cleanup_error,
                )
        gc.collect()


# Failures that are terminal for the PROCESS, not for one task. After any of
# these, every subsequent episode in this process fails identically, so
# recording a row and continuing wastes the rest of the allocation.
#
# Two families:
#
#   Playwright/browser -- the browser env is constructed once and shared by
#   every episode. Job 61500's launch timed out on episode 0 and the run
#   spent 25h46m recording 599 further failures against a dead browser. The
#   "asyncio loop" message is the tell: a browser failure leaves Playwright's
#   event loop marked running, and every later sync_playwright() refuses.
#   These cannot fire during an AgentDojo run -- it uses no browser.
#
#   CUDA context corruption -- distinct from OutOfMemoryError, which IS
#   per-task and recoverable (that is what the OOM branch below is for).
#   An illegal access or a device-side assert leaves the context unusable, so
#   every later forward pass fails whatever the task.
_PROCESS_FATAL_MARKERS = (
    "asyncio loop",
    "BrowserType.launch",
    "Target page, context or browser has been closed",
    "browser has been closed",
    "Connection closed",
    "Target crashed",
    "CUDA error: unspecified launch failure",
    "CUDA error: illegal memory access",
    "device-side assert triggered",
)


def _ensure_inference_state(model) -> list:
    """
    Put a model back into inference state. Returns what had to change.

    Needed because the pixel-space attacks (adversarial_patch,
    pixel_perturbation, cross_modal_injection) call model.train() and
    gradient_checkpointing_enable() to backpropagate into the image, and do
    not restore either. Left that way, HuggingFace disables the KV cache and
    the numerical path of every later episode changes -- see audit entry 8:
    37 of 40 clean-defended episodes differed between two runs of the same
    condition.

    An empty return means the model was already in inference state.
    """
    changed = []
    if getattr(model, "is_gradient_checkpointing", False) and \
            hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        changed.append("gradient_checkpointing")
    if getattr(model, "training", False):
        model.eval()
        changed.append("train_mode")
    for cfg in (getattr(model, "config", None),
                getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and getattr(cfg, "use_cache", True) is False:
            cfg.use_cache = True
            changed.append("use_cache")
    return changed


def _run_episode_safely(fn: Callable, *args, **kwargs) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Wraps a run_agentdojo_episode() call, catching BOTH
    torch.OutOfMemoryError AND every other exception (see the second except
    clause below), so ONE pathological task can't take down the entire run.

    WARNING -- this converts ANY failure into a recorded row, including a
    failure that is terminal for the whole process rather than for one task.
    Job 61500's browser launch timed out on episode 0; this function
    recorded it and returned; the run then spent 25h46m recording 599
    further failures against a dead shared browser and exited COMPLETED 0:0.

    PROCESS-FATAL failures are handled HERE. An error matching
    _PROCESS_FATAL_MARKERS raises SystemExit instead of returning a row, so
    every caller inherits that protection -- run_vwa_adv_matrix.py and all
    four call sites in this file included. SystemExit is a BaseException and
    so passes through `except Exception` untouched.

    PER-TASK failure ACCUMULATION stays the caller's job, because only the
    caller knows which errors are legitimate for a single task.
    run_vwa_adv_curated_matrix has a 25-consecutive-failure backstop in its
    record(), with exemptions for its own benign per-task errors (no victim
    image, no clean reference activation, no cached payload). Other callers
    have no equivalent.

    On the OOM case: this is not
    a hypothetical: job 23613 was a completely FRESH process (model just
    loaded, zero real episodes executed -- 78 prior results were all
    skip-logic, no GPU work at all) and OOM'd on the very FIRST real
    episode it attempted, immediately after the skips. That's decisive:
    it rules out cross-episode accumulation (the earlier
    _cuda_memory_guard fix's target) and confirms this is ONE task's
    prompt being long enough to exceed available memory on its own, from
    a cold start, every time. Without this fix, resubmitting achieves
    NOTHING -- every fresh process reaches the same task and dies the
    same way, forever, regardless of how many times it's resubmitted.

    Returns (result, error) -- error is None on success, or a short
    description on failure, in which case result is a placeholder dict
    (asr_success/utility_success=None, empty turn_activations) so every
    call site's row-building code works unmodified whether the episode
    succeeded or not, rather than needing per-site special-casing.
    """
    # Every episode starts from a known inference state, whatever ran before
    # it in this process. The warning is the evidence: it fires only when an
    # earlier step (an attack's generate()) left the model in the wrong state.
    for _a in list(args) + list(kwargs.values()):
        if isinstance(_a, torch.nn.Module):
            _changed = _ensure_inference_state(_a)
            if _changed:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "restored inference state before episode (was: %s) -- "
                    "an earlier step left the model out of eval mode; see "
                    "audit entry 8", ", ".join(_changed))
    try:
        return fn(*args, **kwargs), None
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return (
            {
                "asr_success": None,
                "utility_success": None,
                "turn_activations": [],
                "turn_tool_calls": [],
                "turn_raw_completions": [],
            },
            f"CUDA_OOM: {str(e).splitlines()[0][:200]}",
        )
    except Exception as e:
        # The ROW keeps only the first line, to stay compact. The TRACEBACK
        # goes to the log. Job 61500 needed a whole separate probe job to
        # diagnose because only the first line survived, and the first line
        # of a Playwright error says what broke but not where.
        import traceback as _tb
        import logging as _logging
        _logging.getLogger(__name__).error(
            "episode raised %s -- full traceback follows:\n%s",
            type(e).__name__, _tb.format_exc())
        gc.collect()

        # Abort the whole run rather than record a row, when the failure is
        # terminal for the process. SystemExit derives from BaseException, so
        # it passes through callers' `except Exception` blocks untouched, and
        # every caller writes its results file incrementally -- so nothing
        # recorded so far is lost, and the failing episode is simply absent
        # rather than present-but-empty, which means a resubmit retries it.
        _msg = f"{type(e).__name__}: {e}"
        for _m in _PROCESS_FATAL_MARKERS:
            if _m in _msg:
                _logging.getLogger(__name__).error(
                    "ABORTING RUN: %r is terminal for this process, not for "
                    "one task. Every remaining episode would fail the same "
                    "way. Fix the cause and resubmit -- rows carrying an "
                    "error are not treated as done.", _m)
                raise SystemExit(
                    f"ABORT: process-fatal failure (matched {_m!r})\n"
                    f"  {_msg.splitlines()[0][:300]}\n"
                    f"  Every remaining episode would fail identically. "
                    f"Resubmit after fixing the cause; incomplete rows are "
                    f"retried automatically.")
        return (
            {
                "asr_success": None,
                "utility_success": None,
                "turn_activations": [],
                "turn_tool_calls": [],
                "turn_raw_completions": [],
            },
            f"{type(e).__name__}: {str(e).splitlines()[0][:200]}",
        )



from defenses.spotlighting import SpotlightingDefense


def _build_defense_registry() -> Dict[str, "BaseDefense"]:
    """
    Named per-variant, not one generic "spotlighting" entry -- the three
    variants (delimiting/datamarking/encoding) have meaningfully
    different robustness/utility profiles per Hines et al. 2024, worth
    comparing separately rather than picking one and reporting it as
    "spotlighting" broadly.
    """
    return {
        "spotlighting_delimiting": SpotlightingDefense(variant="delimiting"),
        "spotlighting_datamarking": SpotlightingDefense(variant="datamarking"),
        "spotlighting_encoding": SpotlightingDefense(variant="encoding"),
    }
