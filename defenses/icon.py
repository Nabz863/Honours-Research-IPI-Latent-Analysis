"""
defenses/icon.py
------------------
ICON: Indirect Prompt Injection Defense for Agents based on Inference-Time
Correction (Table 3.3, inference-time). Wang et al. 2026, arXiv:2602.20708.

Implemented from the paper directly (fetched and read), not from a summary.

WHY ICON MATTERS FOR RQ2 SPECIFICALLY: every other defense wired into this
project either transforms input text (Spotlighting) or only measures
(TaskTracker, CircuitBreaker). ICON is the only one that RESTORES rather
than refuses -- it steers attention back toward the benign task instead of
aborting the trajectory. RQ2 asks whether defenses restore clean-condition
geometry or produce a novel suppression state; a refusal-based defense can
only ever demonstrate suppression, so ICON is the defense that makes the
question answerable at all.

THREE COMPONENTS, all implemented here:

  1. Focus Intensity Score (FIS), Eq. 6-7. Generation-normalised attention
     entropy per (layer, head, generated token):
         E(l,h,i) = -(1/log N) * sum_j a_ij log(a_ij + eps)
         S_l,h    = 1 - mean_i E(l,h,i)
         S_l      = mean_h S_l,h
     Note this is NOT the same statistic as defenses/attention_tracker.py
     (Hung et al.). The paper is explicit: prior work scores only the last
     input token or first generated token, whereas FIS averages over the
     ENTIRE generated sequence. Both are in Table 3.3 and they measure
     different things; do not treat one as a reimplementation of the other.

  2. Latent Space Trace Prober (LSTP), Eq. 8-10. Per head, a 3-vector
     [min, mean, std] of E over the generation; concatenated across all
     heads of the top-K FIS-ranked layers; classified by a small
     1D-CNN + MLP with adaptive max pooling (length-invariant, so response
     length does not affect the decision). The paper reports ~31.5k
     parameters trained on 255 samples in under 2 minutes -- this is a
     genuinely cheap component, unlike a fine-tuned guard model.

  3. Mitigating Rectifier (MR), Eq. 11-12. On heads flagged anomalous by
     FIS, mask the top-tau fraction of attention weights and scale them by
     gamma < 1 BEFORE the softmax, so renormalisation redistributes the
     freed mass onto the rest of the context:
         theta_l,h = (1-tau) percentile of attention weights in head h
         M_l,h(i,j) = 1[a_l,h(i,j) >= theta_l,h]
         A~_l,h = A_l,h * [1 + M_l,h * (gamma - 1)]

VERIFICATION STATUS, stated per component:
  - FIS and LSTP feature extraction: pure tensor maths from the paper's
    equations, unit-testable offline against synthetic attention (see the
    self-test at the bottom). No model required.
  - LSTP training/inference: standard supervised classification; runs on
    saved attention features.
  - MR pre-softmax intervention: THE ONE PIECE THAT CANNOT BE VERIFIED
    OFFLINE. It must patch attention computation inside the model, and the
    correct hook point depends on the installed transformers version's
    attention dispatch. Two paths are attempted (the pluggable
    ALL_ATTENTION_FUNCTIONS registry, then a direct module patch), and if
    neither works it RAISES rather than silently running unmitigated --
    a defense that quietly does nothing would produce results labelled
    "ICON-defended" that contain no defense.

REQUIRES output_attentions=True, which the activation-capture path in
extraction/hooks.py does not currently request (it hooks hidden states,
not attention). Wiring ICON into a real episode therefore needs the
generation call to pass output_attentions=True -- a real integration step,
not a config flag on this class.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDefense, DefenseResult

logger = logging.getLogger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. Focus Intensity Score (Eq. 6-7)
# ---------------------------------------------------------------------------
def generation_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    E(l,h,i) from Eq. 6, for one layer.

    attn : [H, N_Y, N] -- attention from each generated query token i to
           each context key token j, for H heads.
    returns : [H, N_Y] normalised entropy in roughly [0, 1].

    The 1/log(N) factor is the paper's generation-normalisation; without it
    entropy scales with context length and is not comparable across samples
    of different prompt sizes.
    """
    if attn.dim() != 3:
        raise ValueError(f"expected [H, N_Y, N] attention, got {tuple(attn.shape)}")
    n_keys = attn.shape[-1]
    log_n = torch.log(torch.tensor(float(max(n_keys, 2)), device=attn.device))
    ent = -(attn * torch.log(attn + _EPS)).sum(dim=-1) / log_n
    return ent


def focus_intensity(attn: torch.Tensor) -> torch.Tensor:
    """S_l,h from Eq. 7: 1 - mean entropy over the generated sequence.
    Higher = more concentrated attention = more injection-like."""
    return 1.0 - generation_entropy(attn).mean(dim=-1)


def rank_layers_by_fis(attentions: Sequence[torch.Tensor], top_k: int = 4) -> List[int]:
    """Selects the K most attack-sensitive layers by S_l (mean FIS over
    heads), as the paper does before building the prober's feature vector."""
    scores = [float(focus_intensity(a).mean()) for a in attentions]
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]


# ---------------------------------------------------------------------------
# 2. Latent Space Trace Prober (Eq. 8-10)
# ---------------------------------------------------------------------------
def head_features(attn: torch.Tensor) -> torch.Tensor:
    """
    v_l,h from Eq. 8 -- [min, mean, std] of E over the generated sequence.

    attn : [H, N_Y, N]  ->  returns [3, H]

    Laid out as (feature, head) rather than flattened so the prober's 1D
    convolution can treat the three statistics as CHANNELS and convolve
    along the head axis, which is what lets it pick up inter-head
    correlations rather than memorising per-head magnitudes.
    """
    ent = generation_entropy(attn)                     # [H, N_Y]
    if ent.shape[-1] < 2:
        # std is undefined for a single generated token; zero is the honest
        # value here (no volatility observed), not a NaN to propagate.
        std = torch.zeros_like(ent[:, 0])
    else:
        std = ent.std(dim=-1)
    return torch.stack([ent.min(dim=-1).values, ent.mean(dim=-1), std], dim=0)


def build_feature_vector(
    attentions: Sequence[torch.Tensor], layer_indices: Sequence[int]
) -> torch.Tensor:
    """z from Eq. 9 -- [3, H*K], concatenating head features across the
    selected layers."""
    feats = [head_features(attentions[l]) for l in layer_indices]
    return torch.cat(feats, dim=1)


class LatentSpaceTraceProber(nn.Module):
    """1D-CNN + MLP over [3, H*K] head-entropy features (Eq. 10).

    Adaptive max pooling makes the output length-invariant, so the same
    prober handles both short aggressive injections and long subtle ones
    -- the paper's stated reason for choosing it."""

    def __init__(self, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(0)                          # [1, 3, H*K]
        h = self.conv(z)
        h = self.pool(h).squeeze(-1)
        return self.head(h)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_prober(
    prober: LatentSpaceTraceProber,
    features: Sequence[torch.Tensor],
    labels: Sequence[int],
    epochs: int = 200,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Supervised training on (feature vector, is_attacked) pairs.

    The paper trains on 255 synthesised adaptive-attack samples in under two
    minutes. Here the natural training set is this project's OWN matrix
    output: clean episodes as label 0, attacked episodes as label 1. That
    has a real consequence worth stating in the write-up -- a prober trained
    on our attacks and evaluated on our attacks is in-distribution, whereas
    the paper's headline claim is OOD transfer (trained on TrojanTools,
    evaluated on AgentDojo at 97-98% detection). To make an OOD claim, train
    on one attack family and test on another.
    """
    device = device or torch.device("cpu")
    prober = prober.to(device).train()
    x = torch.stack([f.to(device).float() for f in features])
    y = torch.tensor(list(labels), device=device, dtype=torch.long)
    opt = torch.optim.Adam(prober.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(prober(x), y)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    prober.eval()
    with torch.no_grad():
        acc = float((prober(x).argmax(dim=-1) == y).float().mean())
    return {"final_loss": history[-1], "train_accuracy": acc,
            "n_samples": len(labels), "n_parameters": prober.n_parameters()}


# ---------------------------------------------------------------------------
# 3. Mitigating Rectifier (Eq. 11-12)
# ---------------------------------------------------------------------------
def steer_attention_scores(
    scores: torch.Tensor, head_indices: Sequence[int], tau: float, gamma: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Eq. 11-12, applied to PRE-SOFTMAX attention scores.

    scores : [B, H, Q, K] raw scores (post-QK^T, pre-softmax)
    head_indices : heads flagged anomalous by FIS
    tau : top-tau fraction of weights to suppress (steering scope)
    gamma : multiplicative suppression factor, < 1 (steering intensity)

    Suppressing peaks before the softmax is what makes this a RESTORATION
    rather than a deletion: renormalisation redistributes the freed
    probability mass across the rest of the context, which is where the
    benign task tokens are. Applying the same scaling after the softmax
    would simply shrink those weights and leave the distribution unnormalised.

    AUDIT 13.8 -- two corrections to the original implementation:
      * Scaling a raw score by gamma is not suppression: for a NEGATIVE
        score, gamma < 1 moves it toward zero and RAISES its attention
        weight. The sign-safe operation with the stated meaning ("scale
        the weight by gamma, then renormalise") is adding log(gamma) to the
        score: softmax(s + log g) is proportional to g * exp(s).
      * The top-tau threshold was taken over every key, including future
        positions the causal mask removes during prefill. It is now taken
        over the keys the mask leaves visible.
    """
    if not head_indices or gamma >= 1.0:
        return scores
    import math
    log_g = math.log(max(gamma, 1e-12))
    out = scores.clone()
    for h in head_indices:
        if h >= scores.shape[1]:
            continue
        head = out[:, h]                                # [B, Q, K]
        visible = head
        if attention_mask is not None:
            m = attention_mask[:, 0] if attention_mask.dim() == 4 else attention_mask
            visible = head + m[..., : head.shape[-1]].to(head.dtype)
        # theta = (1 - tau) percentile along the visible keys, per query row.
        k = max(1, int(head.shape[-1] * tau))
        thresh = visible.topk(k, dim=-1).values[..., -1:].expand_as(head)
        sel = (visible >= thresh) & torch.isfinite(visible)
        out[:, h] = head + sel.to(head.dtype) * log_g
    return out


class MitigatingRectifier:
    """Installs pre-softmax attention steering on flagged heads.

    The install path is version-dependent, so it is attempted rather than
    assumed, and FAILS LOUDLY if no path works -- silently running without
    mitigation would produce rows labelled ICON-defended that contain no
    intervention at all.
    """

    def __init__(self, tau: float = 0.05, gamma: float = 0.3):
        self.tau = tau
        self.gamma = gamma
        self.flagged: Dict[int, List[int]] = {}          # layer -> head indices
        self._installed = False
        self._original = None
        self._module = None

    def set_flagged_heads(self, attentions: Sequence[torch.Tensor],
                          layer_indices: Sequence[int], top_heads: int = 4) -> None:
        """Selects, per chosen layer, the heads with the highest FIS --
        the 'over-focusing' heads the paper targets."""
        self.flagged = {}
        for l in layer_indices:
            fis = focus_intensity(attentions[l])         # [H]
            self.flagged[l] = torch.topk(fis, min(top_heads, fis.numel())).indices.tolist()

    def install(self, model) -> str:
        """Patches the model's module-level eager_attention_forward.

        transformers 5.12 has no ALL_ATTENTION_FUNCTIONS['eager'] entry --
        eager is the fallback, defined per model file and selected by
        ALL_ATTENTION_FUNCTIONS.get(_attn_implementation,
        eager_attention_forward). The module global is the real hook.

        Unlike the recorder, this cannot wrap the original: Eq. 11-12
        modifies scores BEFORE the softmax and the original returns only
        post-softmax probabilities. So the forward pass is reimplemented,
        and a gamma=1.0 self-check runs on the first real call -- with
        gamma=1.0 the steering is a no-op, so the reimplementation must
        reproduce the original exactly. If it does not, this raises
        rather than evaluating attacks through subtly wrong attention.
        """
        from defenses.icon_runtime import find_attention_module
        mod = find_attention_module(model)
        base = getattr(mod, "eager_attention_forward")
        rectifier = self
        checked = {"done": False}

        def steered(module, query, key, value, attention_mask=None,
                    scaling=None, dropout=0.0, **kwargs):
            if scaling is None:
                scaling = query.shape[-1] ** -0.5
            n_rep = query.shape[1] // key.shape[1] if key.shape[1] else 1
            k = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key
            v = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value

            scores = torch.matmul(query, k.transpose(2, 3)) * scaling
            layer_idx = getattr(module, "layer_idx", None)
            heads = rectifier.flagged.get(layer_idx, [])

            if not checked["done"]:
                checked["done"] = True
                ref_out, _ = base(module, query, key, value, attention_mask,
                                  scaling, dropout=dropout, **kwargs)
                sc = scores
                if attention_mask is not None:
                    sc = sc + attention_mask[:, :, :, : k.shape[-2]]
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(query.dtype)
                mine = torch.matmul(pr, v).transpose(1, 2).contiguous()
                if not torch.allclose(ref_out, mine, atol=1e-2, rtol=1e-2):
                    raise RuntimeError(
                        "ICON MitigatingRectifier's reimplementation of eager "
                        "attention does not match this model's own at "
                        "gamma=1.0 (max abs diff "
                        f"{float((ref_out - mine).abs().max()):.3e}). The "
                        "rectifier would be steering attention computed "
                        "differently from the real forward pass. Fix the "
                        "reimplementation before running ICON."
                    )

            if heads:
                am = attention_mask[:, :, :, : k.shape[-2]] if attention_mask is not None else None
                scores = steer_attention_scores(scores, heads,
                                                rectifier.tau, rectifier.gamma,
                                                attention_mask=am)
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, : k.shape[-2]]
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            probs = F.dropout(probs, p=dropout, training=module.training)
            out = torch.matmul(probs, v).transpose(1, 2).contiguous()
            return out, probs

        self._module = mod
        self._original = base
        mod.eager_attention_forward = steered
        self._installed = True
        return f"{mod.__name__}.eager_attention_forward"

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._module.eager_attention_forward = self._original
        self._installed = False


# ---------------------------------------------------------------------------
# Defense wrapper
# ---------------------------------------------------------------------------
class ICONDefense(BaseDefense):
    """
    ICON as a whole: probe, then rectify.

    NOTE ON THE BaseDefense INTERFACE: ICON is not a content transformation.
    It inspects and modifies model internals at inference time, so apply()
    is a deliberate pass-through -- it returns the content unchanged and
    records why. The real entry points are detect() and the rectifier.
    Registering ICON in a runner that only calls apply() would produce
    "ICON-defended" rows with no defense applied, so the runner must call
    detect()/rectifier explicitly around generation.
    """

    name = "icon"

    def __init__(self, prober: Optional[LatentSpaceTraceProber] = None,
                 top_k_layers: int = 4, top_heads: int = 4,
                 tau: float = 0.05, gamma: float = 0.3):
        self.prober = prober or LatentSpaceTraceProber()
        self.top_k_layers = top_k_layers
        self.top_heads = top_heads
        self.rectifier = MitigatingRectifier(tau=tau, gamma=gamma)
        self.layer_indices: Optional[List[int]] = None

    def select_layers(self, attentions: Sequence[torch.Tensor]) -> List[int]:
        self.layer_indices = rank_layers_by_fis(attentions, self.top_k_layers)
        return self.layer_indices

    def features(self, attentions: Sequence[torch.Tensor]) -> torch.Tensor:
        idx = self.layer_indices or self.select_layers(attentions)
        return build_feature_vector(attentions, idx)

    def detect(self, attentions: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Returns {is_attacked, confidence, fis_per_layer}."""
        z = self.features(attentions)
        self.prober.eval()
        with torch.no_grad():
            logits = self.prober(z.float())
            probs = F.softmax(logits, dim=-1).squeeze(0)
        return {
            "is_attacked": bool(probs.argmax().item() == 1),
            "confidence": float(probs.max()),
            "fis_per_layer": [float(focus_intensity(a).mean()) for a in attentions],
            "layers_used": list(self.layer_indices or []),
        }

    def apply(self, content: str, **kwargs) -> DefenseResult:
        return DefenseResult(
            defense_name=self.name,
            transformed_content=content,
            system_prompt_addendum="",
            metadata={
                "note": "ICON does not transform content. It probes attention and steers it at "
                        "inference time. Call detect() and install the MitigatingRectifier around "
                        "generation; apply() alone applies NO defense.",
                "tau": self.rectifier.tau, "gamma": self.rectifier.gamma,
            },
        )


if __name__ == "__main__":
    # Offline self-test -- synthetic attention, no model, no GPU.
    torch.manual_seed(0)
    H, N_Y, N, L = 8, 12, 64, 6

    # Benign: diffuse attention. Attacked: one head collapsed onto a few keys.
    benign = [F.softmax(torch.randn(H, N_Y, N) * 0.5, dim=-1) for _ in range(L)]
    attacked = [b.clone() for b in benign]
    spike = torch.full((N_Y, N), -8.0)
    spike[:, 30:34] = 6.0
    attacked[3][2] = F.softmax(spike, dim=-1)

    fb = [float(focus_intensity(a).mean()) for a in benign]
    fa = [float(focus_intensity(a).mean()) for a in attacked]
    print(f"FIS per layer, benign  : {[round(v,4) for v in fb]}")
    print(f"FIS per layer, attacked: {[round(v,4) for v in fa]}")
    print(f"layer 3 FIS rose: {fa[3] > fb[3]}  (collapsed head should raise it)")

    print(f"top-K layers by FIS (attacked): {rank_layers_by_fis(attacked, 4)}")
    z = build_feature_vector(attacked, rank_layers_by_fis(attacked, 4))
    print(f"feature vector shape: {tuple(z.shape)} (expect [3, H*K] = [3, {H*4}])")

    prober = LatentSpaceTraceProber()
    print(f"prober parameters: {prober.n_parameters()} (paper reports ~31.5k)")

    feats = ([build_feature_vector(benign, [0,1,2,3]) for _ in range(8)]
             + [build_feature_vector(attacked, [0,1,2,3]) for _ in range(8)])
    stats = train_prober(prober, feats, [0]*8 + [1]*8, epochs=150)
    print(f"train: {stats}")

    scores = torch.randn(1, H, N_Y, N)
    steered = steer_attention_scores(scores, [2], tau=0.05, gamma=0.3)
    changed = (steered != scores).any(dim=-1).sum().item()
    print(f"steering altered head 2 only: "
          f"{bool((steered[:,2] != scores[:,2]).any()) and not bool((steered[:,0] != scores[:,0]).any())}")
    print("\nOffline checks cover FIS, features, prober, and the steering maths. "
          "The MR INSTALL path (patching attention dispatch) is version-dependent and "
          "must be verified on the cluster -- it raises rather than silently no-opping.")
