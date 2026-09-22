"""
defenses/icon_runtime.py
------------------------
Makes ICON actually run during generation.

WHY THIS FILE EXISTS
    defenses/icon.py implements FIS, the prober and the rectifier, and
    says plainly that apply() is a pass-through: "Registering ICON in a
    runner that only calls apply() would produce 'ICON-defended' rows
    with no defense applied". This module is the missing runner-side
    integration -- detect, then rectify, around a real generate() call.

WHY NOT output_attentions=True
    The obvious route is generate(output_attentions=True,
    return_dict_in_generate=True) and feeding output.attentions to
    ICON. That is not viable at this project's context lengths. The
    prefill step alone returns [1, H, L, L] per layer; at L~4000 and
    H=32 that is ~1 GB per layer, tens of GB across the stack, before
    a single token is generated. It OOMs on a 97 GB card.

    ICON does not need the tensors -- it needs the per-head entropy of
    attention from each GENERATED token. That is a scalar per (layer,
    head, step). So this module patches the same attention dispatch the
    rectifier already patches, computes the entropy inline, and throws
    the attention matrix away. Cost is layers x heads x steps floats:
    kilobytes.

    It also means only DECODE steps (query length 1) are recorded.
    Prefill rows are attention from prompt tokens to prompt tokens,
    which is not what Eq. 6 averages over.

TWO-PASS DESIGN
    Pass 1  record entropy over a SHORT generation (detect_tokens, default
            32). FIS is an average over generated tokens; 32 is ample for
            a stable statistic and avoids paying for a full 512-token
            generation just to decide whether to defend.
    detect  LSTP classifies the feature vector.
    Pass 2  if flagged: pick the over-focusing heads, install the
            Mitigating Rectifier, regenerate in full, uninstall.
            If not flagged: generate in full undefended.

    A clean episode therefore costs one short pass plus one full pass.
    That overhead is real and should be reported as ICON's inference
    cost, not hidden.

REQUIREMENTS AND FAILURE MODES
    - The model must be loaded with attn_implementation="eager".
      SDPA and flash-attention do not route through
      ALL_ATTENTION_FUNCTIONS["eager"], so the patch would be installed
      and never called -- silently no defense. assert_eager() checks
      this and RAISES rather than letting that happen.
    - The prober must be trained. An untrained LatentSpaceTraceProber
      outputs noise, so ICON would flag at chance. load_prober() refuses
      to proceed on random weights unless explicitly overridden.
"""

from __future__ import annotations

import os

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .icon import (
    ICONDefense,
    LatentSpaceTraceProber,
    build_feature_vector,
    focus_intensity,
    rank_layers_by_fis,
    steer_attention_scores,
)

logger = logging.getLogger(__name__)
_EPS = 1e-9


# ---------------------------------------------------------------------------

def find_attention_module(model):
    """Locate the modeling module that defines eager_attention_forward
    for this model. Every model file defines its own, so patching the
    wrong one installs cleanly and never runs."""
    import sys
    seen = []
    for m in (model, getattr(model, "model", None),
              getattr(model, "language_model", None)):
        if m is None:
            continue
        mod = sys.modules.get(type(m).__module__)
        seen.append(type(m).__module__)
        if mod is not None and hasattr(mod, "eager_attention_forward"):
            return mod
    # decoder layers are defined in the same file as the attention
    for sub in (model.modules() if hasattr(model, "modules") else []):
        if type(sub).__name__.endswith(("Attention", "DecoderLayer")):
            mod = sys.modules.get(type(sub).__module__)
            if mod is not None and hasattr(mod, "eager_attention_forward"):
                return mod
    raise RuntimeError(
        "could not find a module defining eager_attention_forward for this "
        f"model (looked in {seen}). ICON cannot install its hooks, and "
        "running without them would produce rows labelled ICON-defended "
        "that contain no defense."
    )


def assert_eager(model) -> None:
    """ICON's rectifier patches ALL_ATTENTION_FUNCTIONS['eager']. If the
    model dispatches to sdpa/flash instead, the patch installs cleanly
    and is never invoked -- the worst possible outcome, since rows would
    be labelled ICON-defended with no intervention applied."""
    impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if impl != "eager":
        raise RuntimeError(
            f"model attn_implementation={impl!r}, ICON requires 'eager'. "
            "sdpa/flash bypass ALL_ATTENTION_FUNCTIONS entirely, so the "
            "rectifier would install and never run. Reload with "
            "attn_implementation='eager'."
        )


class AttentionEntropyRecorder:
    """Records per-head attention entropy for DECODE steps only, without
    materialising attention tensors.

    Produces, per layer, a [H, N_Y] tensor of normalised entropies --
    exactly the shape defenses.icon.generation_entropy would compute
    from [H, N_Y, N] attention, but assembled a step at a time.

    Key slicing note: at decode step t the key axis has grown to
    prompt_len + t. Entropy is computed over the PROMPT portion only
    (the first `context_len` keys), so every step is normalised over the
    same key set and the statistic is comparable across steps. That is
    also the semantically right choice -- ICON is about how much the
    generation over-focuses on part of the retrieved CONTEXT, not on the
    model's own previous output.
    """

    def __init__(self, context_len: Optional[int] = None):
        self.context_len = context_len
        self.per_layer: Dict[int, List[torch.Tensor]] = {}
        self._installed = False
        self._original = None

    def _record(self, layer_idx: Optional[int], probs: torch.Tensor) -> None:
        # probs: [B, H, Q, K]; only decode steps (Q == 1) are generated tokens
        if layer_idx is None or probs.shape[2] != 1:
            return
        k = probs.shape[-1]
        n = min(self.context_len or k, k)
        p = probs[0, :, 0, :n].float()                    # [H, n]
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        log_n = torch.log(torch.tensor(float(max(n, 2)), device=p.device))
        ent = -(p * torch.log(p + _EPS)).sum(dim=-1) / log_n   # [H]
        self.per_layer.setdefault(layer_idx, []).append(ent.detach().cpu())

    def install(self, model=None):
        """Wraps the model's module-level eager_attention_forward.

        Wrapping rather than reimplementing: the original computes the
        attention, this reads the probabilities it returns and records
        entropy. No part of the attention maths is duplicated here, so
        it cannot diverge from what the model actually does."""
        mod = find_attention_module(model)
        base = getattr(mod, "eager_attention_forward")
        rec = self

        def recording(module, query, key, value, *args, **kwargs):
            out, probs = base(module, query, key, value, *args, **kwargs)
            if probs is not None:
                rec._record(getattr(module, "layer_idx", None), probs)
            return out, probs

        self._module = mod
        self._original = base
        mod.eager_attention_forward = recording
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return
        self._module.eager_attention_forward = self._original
        self._installed = False

    def attentions(self) -> List[torch.Tensor]:
        """[L] list of [H, 1, N_Y] tensors, shaped so icon.py's
        generation_entropy (which expects [H, N_Y, N]) is bypassed --
        callers should use entropies() instead. Kept for symmetry."""
        return [torch.stack(v, dim=-1).unsqueeze(1)
                for _, v in sorted(self.per_layer.items())]

    def entropies(self) -> List[torch.Tensor]:
        """[L] list of [H, N_Y] normalised entropies -- the direct input
        to FIS and to the prober's feature vector."""
        return [torch.stack(v, dim=-1) for _, v in sorted(self.per_layer.items())]


# ---------------------------------------------------------------------------
# FIS / features straight from recorded entropies
# ---------------------------------------------------------------------------

def fis_from_entropy(ent: torch.Tensor) -> torch.Tensor:
    """S_l,h = 1 - mean_i E(l,h,i)  (icon.py Eq. 7), on recorded entropy."""
    return 1.0 - ent.mean(dim=-1)


def features_from_entropy(entropies: Sequence[torch.Tensor],
                          layer_indices: Sequence[int]) -> torch.Tensor:
    """z (Eq. 9) as [3, H*K] -- [min, mean, std] per head, concatenated
    across the selected layers. Mirrors icon.head_features exactly,
    reading recorded entropies instead of attention tensors."""
    feats = []
    for l in layer_indices:
        e = entropies[l]                                   # [H, N_Y]
        std = (torch.zeros_like(e[:, 0]) if e.shape[-1] < 2
               else e.std(dim=-1))
        feats.append(torch.stack([e.min(dim=-1).values, e.mean(dim=-1), std], dim=0))
    return torch.cat(feats, dim=1)


def rank_layers_from_entropy(entropies: Sequence[torch.Tensor], top_k: int) -> List[int]:
    scores = [float(fis_from_entropy(e).mean()) for e in entropies]
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]


# ---------------------------------------------------------------------------

@contextmanager
def _recording(context_len: int, model=None):
    rec = AttentionEntropyRecorder(context_len=context_len).install(model)
    try:
        yield rec
    finally:
        rec.uninstall()


@contextmanager
def _rectifying(rectifier, model):
    path = rectifier.install(model)
    logger.info("ICON rectifier installed via %s", path)
    try:
        yield
    finally:
        rectifier.uninstall()


def icon_generate(model, inputs: Dict[str, Any], icon: ICONDefense,
                  max_new_tokens: int = 512, detect_tokens: int = 32,
                  gen_kwargs: Optional[Dict[str, Any]] = None):
    """Two-pass ICON generation. Returns (output_ids, info)."""
    assert_eager(model)
    gk = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    gk.update(gen_kwargs or {})
    ctx_len = int(inputs["input_ids"].shape[1])

    # --- pass 1: short generation, record entropy ----------------------
    with _recording(ctx_len, model) as rec:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=detect_tokens, **gk)
    ents = rec.entropies()

    info: Dict[str, Any] = {"icon_detect_tokens": detect_tokens,
                            "icon_n_layers_recorded": len(ents)}
    if not ents:
        raise RuntimeError(
            "ICON recorded no decode-step attention. The model is not "
            "dispatching through ALL_ATTENTION_FUNCTIONS['eager'] -- "
            "running now would apply NO defense."
        )

    # ORACLE MODE. ICON's prober did not train at n=60 (loss 0.648 vs
    # ln(2)=0.693, accuracy 0.542) even after the last-turn feature fix
    # made clean and attacked features differ at all (1.82e-03, up from
    # exactly 0.0). A detector that classifies at near-chance cannot
    # gate an intervention, but the Mitigating Rectifier is a separate
    # component and has never been evaluated. Forcing the flag isolates
    # the INTERVENTION from the DETECTOR: it answers "does pre-softmax
    # attention steering restore clean geometry", which is RQ2's
    # question, without depending on a detector this data cannot
    # calibrate. Report as rectifier-under-oracle-detection, never as
    # ICON end-to-end.
    force = os.environ.get("ICON_ORACLE_ATTACKED") == "1"
    layers = rank_layers_from_entropy(ents, icon.top_k_layers)
    z = features_from_entropy(ents, layers)
    icon.prober.eval()
    with torch.no_grad():
        probs = F.softmax(icon.prober(z.float()), dim=-1).squeeze(0)
    flagged = True if force else bool(probs.argmax().item() == 1)

    info.update({
        "icon_layers_used": layers,
        "icon_is_attacked": flagged,
        "icon_confidence": float(probs.max()),
        "icon_fis_per_layer": [float(fis_from_entropy(e).mean()) for e in ents],
    })

    # --- pass 2 --------------------------------------------------------
    if not flagged:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, **gk)
        info["icon_rectified"] = False
        return out, info

    icon.rectifier.flagged = {}
    for l in layers:
        f = fis_from_entropy(ents[l])
        icon.rectifier.flagged[l] = torch.topk(
            f, min(icon.top_heads, f.numel())).indices.tolist()
    info["icon_flagged_heads"] = {int(k): v for k, v in icon.rectifier.flagged.items()}

    with _rectifying(icon.rectifier, model):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, **gk)
    info["icon_rectified"] = True
    info["icon_tau"] = icon.rectifier.tau
    info["icon_gamma"] = icon.rectifier.gamma
    return out, info


def record_features(model, inputs: Dict[str, Any], top_k_layers: int = 4,
                    detect_tokens: int = 32,
                    gen_kwargs: Optional[Dict[str, Any]] = None):
    """One recording pass, returning (feature_vector, layer_indices).
    Used by the calibration script to build the prober's training set
    from real clean and attacked episodes."""
    assert_eager(model)
    gk = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    gk.update(gen_kwargs or {})
    ctx = int(inputs["input_ids"].shape[1])
    with _recording(ctx, model) as rec:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=detect_tokens, **gk)
    ents = rec.entropies()
    if not ents:
        raise RuntimeError("no decode-step attention recorded; model is not eager")
    layers = rank_layers_from_entropy(ents, top_k_layers)
    return features_from_entropy(ents, layers), layers


def load_prober(path: str, allow_untrained: bool = False) -> LatentSpaceTraceProber:
    p = LatentSpaceTraceProber()
    import os
    if os.path.exists(path):
        p.load_state_dict(torch.load(path, map_location="cpu"))
        p.eval()
        return p
    if not allow_untrained:
        raise FileNotFoundError(
            f"no trained prober at {path}. An untrained LatentSpaceTraceProber "
            "classifies at chance, so ICON would flag episodes randomly. Run "
            "experiments/calibrate_icon.py first."
        )
    logger.warning("using an UNTRAINED prober -- detection is random")
    return p
