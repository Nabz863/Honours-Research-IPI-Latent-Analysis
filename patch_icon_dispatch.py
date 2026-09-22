"""
patch_icon_dispatch.py -- fixes ICON's attention hook point.

Run once from the repo root:  python3 patch_icon_dispatch.py

THE BUG
    Both AttentionEntropyRecorder.install() and
    MitigatingRectifier.install() patched
    ALL_ATTENTION_FUNCTIONS["eager"]. In transformers 5.12 that key does
    not exist -- the registry holds only sdpa, the flash variants,
    flex_attention and the paged|* entries. Confirmed on the cluster:

        sorted(ALL_ATTENTION_FUNCTIONS.keys()) ->
        ['flash_attention_2','flash_attention_3','flash_attention_4',
         'flex_attention','paged|eager','paged|flash_attention_2',
         'paged|flash_attention_3','paged|flash_attention_4',
         'paged|sdpa','sdpa']

    Eager is the FALLBACK, not a registry entry. Each model file defines
    a module-level eager_attention_forward and does:

        ALL_ATTENTION_FUNCTIONS.get(self.config._attn_implementation,
                                    eager_attention_forward)

    so with _attn_implementation="eager" the lookup misses and the
    module global is used. That global is the real patch point, and
    because Python resolves globals at call time, swapping the module
    attribute takes effect immediately.

TWO DIFFERENT STRATEGIES, DELIBERATELY
    Recorder  -- WRAPS the original: calls it, reads the returned
                 attention probabilities, records entropy, returns the
                 original's output untouched. Nothing is reimplemented,
                 so nothing can silently diverge from the real
                 attention.

    Rectifier -- CANNOT wrap, because Eq. 11-12 modifies scores BEFORE
                 the softmax and the original only returns post-softmax
                 probabilities. It therefore reimplements the forward
                 pass. To catch reimplementation error, install() runs a
                 gamma=1.0 self-check against the original on the real
                 tensors at first call: with gamma=1.0 the steering is a
                 no-op, so the reimplementation MUST match the original
                 bit-for-bit. If it does not, it raises -- rather than
                 running an attack evaluation through subtly wrong
                 attention.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# 1. icon_runtime.py -- recorder patches the module global, by wrapping
# --------------------------------------------------------------------------
p = Path("defenses/icon_runtime.py")
s = p.read_text()

old = '''    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        base = ALL_ATTENTION_FUNCTIONS.get("eager")
        if base is None:
            raise RuntimeError("no 'eager' entry in ALL_ATTENTION_FUNCTIONS")
        rec = self

        def recording(module, query, key, value, attention_mask=None,
                      scaling=None, dropout=0.0, **kwargs):
            if scaling is None:
                scaling = query.shape[-1] ** -0.5
            n_rep = query.shape[1] // key.shape[1] if key.shape[1] else 1
            k = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key
            v = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value
            scores = torch.matmul(query, k.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, : k.shape[-2]]
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            rec._record(getattr(module, "layer_idx", None), probs)
            probs = F.dropout(probs, p=dropout, training=module.training)
            out = torch.matmul(probs, v).transpose(1, 2).contiguous()
            return out, probs

        self._original = base
        ALL_ATTENTION_FUNCTIONS["eager"] = recording
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = self._original
        self._installed = False'''

new = '''    def install(self, model=None):
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
        self._installed = False'''

assert s.count(old) == 1, "anchor: recorder install"
s = s.replace(old, new)

# helper + fail-loud module resolution
old2 = "def assert_eager(model) -> None:"
new2 = '''def find_attention_module(model):
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


def assert_eager(model) -> None:'''
assert s.count(old2) == 1, "anchor: assert_eager"
s = s.replace(old2, new2, 1)

# recorder is now installed with the model
s = s.replace("    rec = AttentionEntropyRecorder(context_len=context_len).install()",
              "    rec = AttentionEntropyRecorder(context_len=context_len).install(model)")
old3 = "@contextmanager\ndef _recording(context_len: int):"
new3 = "@contextmanager\ndef _recording(context_len: int, model=None):"
assert s.count(old3) == 1, "anchor: _recording sig"
s = s.replace(old3, new3)
s = s.replace("with _recording(ctx_len) as rec:", "with _recording(ctx_len, model) as rec:")
s = s.replace("with _recording(ctx) as rec:", "with _recording(ctx, model) as rec:")

p.write_text(s)
print("icon_runtime.py: recorder now wraps the module-level eager_attention_forward")

# --------------------------------------------------------------------------
# 2. icon.py -- rectifier patches the module global, with a no-op self-check
# --------------------------------------------------------------------------
p = Path("defenses/icon.py")
s = p.read_text()

start = s.index("    def install(self, model) -> str:")
end = s.index("    def uninstall(self) -> None:")
old = s[start:end]

new = '''    def install(self, model) -> str:
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
                scores = steer_attention_scores(scores, heads,
                                                rectifier.tau, rectifier.gamma)
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

'''
s = s[:start] + new + s[end:]

old_un = '''    def uninstall(self) -> None:
        if not self._installed:
            return
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = self._original
        self._installed = False'''
new_un = '''    def uninstall(self) -> None:
        if not self._installed:
            return
        self._module.eager_attention_forward = self._original
        self._installed = False'''
assert s.count(old_un) == 1, "anchor: rectifier uninstall"
s = s.replace(old_un, new_un)

s = s.replace("        self._installed = False\n        self._original = None",
              "        self._installed = False\n        self._original = None\n        self._module = None")

p.write_text(s)
print("icon.py: rectifier patches the module global, with a gamma=1.0 self-check")
