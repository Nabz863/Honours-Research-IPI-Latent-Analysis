"""
extraction/model_loader.py
--------------------------
Loads open-source VLMs in a configuration suitable for activation
extraction on the Wits HPC cluster.

Supported models
----------------
  - LLaVA-NeXT  (7B / 13B)   [primary]
  - Qwen2-VL    (7B)          [co-primary]
  - InternVL2   (8B)          [cross-architecture validation]
  - Qwen3-VL    (8B)          [4th model -- added specifically to test
    whether sustained, multi-step agentic engagement (or the lack of it)
    is a LLaVA-NeXT-specific limitation or a broader one. Real AgentDojo
    data showed the model never exceeding 2 turns across 5580 episodes,
    correlating with near-zero ASR; Qwen3-VL-8B's own model card
    explicitly advertises multi-step tool-invoking agent behavior as a
    design focus, unlike the three existing targets, none of which were
    chosen for that property specifically. Verified importable against
    a real transformers install (transformers==5.14.1,
    Qwen3VLForConditionalGeneration) -- NOT verified against the real
    downloaded weights, which needs huggingface.co access this
    environment doesn't have; that verification needs to happen on the
    cluster, which already has it (Phase 1 baselines for the other three
    models were downloaded there).]

HPC notes
---------
  - bf16 for every model (audit 14.1, P2a: LLaVA-NeXT was the last one on
    fp16). 4-bit quantisation (load_in_4bit=True) alters residual-stream
    precision -- flag any result obtained under it.
  - device_map="auto" distributes across available GPUs automatically.
  - CPU fallback included for local dev / unit tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoModelForCausalLM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoadConfig:
    """
    Parameters controlling how a VLM is loaded.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID or local path.
    load_in_4bit : bool
        Use BitsAndBytes 4-bit quantisation. Saves VRAM but alters
        activation precision — note in any results.
    load_in_8bit : bool
        Use BitsAndBytes 8-bit quantisation. Better precision than 4-bit.
    dtype : torch.dtype
        Weight dtype when not quantising. bf16 for every model (14.1).
    device_map : str | dict
        Passed to from_pretrained. "auto" = distribute across GPUs.
    max_memory : dict | None
        Optional per-device memory cap, e.g. {"cuda:0": "20GiB", "cpu": "60GiB"}
    attn_implementation : str
        Attention backend. "flash_attention_2" if available, else "eager".
    """
    model_id: str
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    dtype: torch.dtype = torch.bfloat16
    device_map: Any = "auto"
    max_memory: Optional[Dict] = None
    attn_implementation: str = "eager"
    extra_kwargs: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class VLMLoader:
    """
    Loads a VLM and its processor, exposing a unified interface regardless
    of the underlying architecture.
    """

    # Map from architecture name → loader method
    _LOADERS = {
        "llava": "_load_llava",
        "qwen2_vl": "_load_qwen2vl",
        "internvl": "_load_internvl",
        "qwen3_vl": "_load_qwen3vl",
    }

    def __init__(self, config: LoadConfig):
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.processor = None
        self.arch: Optional[str] = None

    def load(self) -> Tuple[torch.nn.Module, Any]:
        """
        Auto-detect architecture from model_id and load.

        Returns
        -------
        (model, processor)
        """
        mid = self.config.model_id.lower()

        if "llava" in mid:
            self.arch = "llava"
            self.model, self.processor = self._load_llava()
        elif "qwen3" in mid and "vl" in mid:
            # Checked BEFORE the qwen2 branch deliberately -- "qwen2" is
            # not a substring of "qwen3", so order wouldn't actually
            # matter here, but keeping newer-first reads correctly if a
            # qwen4/qwen3.5 naming pattern ever gets added later.
            self.arch = "qwen3_vl"
            self.model, self.processor = self._load_qwen3vl()
        elif "qwen2" in mid and "vl" in mid:
            self.arch = "qwen2_vl"
            self.model, self.processor = self._load_qwen2vl()
        elif "internvl" in mid:
            self.arch = "internvl"
            self.model, self.processor = self._load_internvl()
        elif "gemma" in mid:
            # Gemma 4 reports model_type 'gemma4_unified' (12B/31B) or
            # 'gemma4' (E2B/E4B); both load through the same path here.
            self.arch = "gemma4"
            self.model, self.processor = self._load_gemma4()
        else:
            raise ValueError(
                f"Cannot infer architecture from model_id='{self.config.model_id}'. "
                "Set arch explicitly or add a loader."
            )

        self.model.eval()
        logger.info(
            "Loaded %s (%s) | device_map=%s | dtype=%s | 4bit=%s | 8bit=%s",
            self.config.model_id,
            self.arch,
            self.config.device_map,
            self.config.dtype,
            self.config.load_in_4bit,
            self.config.load_in_8bit,
        )
        return self.model, self.processor

    # ------------------------------------------------------------------
    # Architecture-specific loaders
    # ------------------------------------------------------------------

    def _quant_config(self) -> Optional[BitsAndBytesConfig]:
        if self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if self.config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _base_kwargs(self) -> Dict:
        kwargs: Dict = {
            "device_map": self.config.device_map,
            "attn_implementation": self.config.attn_implementation,
        }
        qc = self._quant_config()
        if qc is not None:
            kwargs["quantization_config"] = qc
        else:
            kwargs["torch_dtype"] = self.config.dtype

        if self.config.max_memory:
            kwargs["max_memory"] = self.config.max_memory

        kwargs.update(self.config.extra_kwargs)
        return kwargs

    def _load_llava(self):
        kwargs = self._base_kwargs()
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config.model_id, **kwargs
        )
        processor = LlavaNextProcessor.from_pretrained(self.config.model_id)
        return model, processor

    def _load_qwen2vl(self):
        kwargs = self._base_kwargs()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_id, **kwargs
        )
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        return model, processor

    def _load_internvl(self):
        """
        As of transformers 5.x, InternVL has native (non-remote-code) support
        via InternVLForConditionalGeneration / AutoModelForImageTextToText,
        for "-hf" suffixed checkpoints (e.g. OpenGVLab/InternVL3-8B-hf).
        This sidesteps the all_tied_weights_keys incompatibility that the
        original OpenGVLab/InternVL2-8B (trust_remote_code=True) checkpoint
        hits under transformers v5.x -- see SCOPING_DECISIONS.md Decision 3.
        No trust_remote_code needed; standard Auto* classes work directly.
        """
        kwargs = self._base_kwargs()
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id, **kwargs
        )
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        return model, processor

    def _load_qwen3vl(self):
        """
        Qwen3-VL-8B-Instruct -- dense 8B, Apache 2.0, released 2025-10-15.
        Native transformers support via Qwen3VLForConditionalGeneration
        (confirmed importable against transformers==5.14.1 in a real,
        current install; the actual weight download and forward-pass
        behavior are UNVERIFIED here -- this environment has no
        huggingface.co access, only pypi.org, so the class import could
        be checked but not a real load). Same standard from_pretrained
        kwargs as every other loader here (device_map, attn_implementation,
        torch_dtype/quantization_config) -- no special-casing expected,
        but confirm on the cluster before trusting it, same as every
        other "verified against real code, not yet real weights" caveat
        elsewhere in this project.

        Qwen's own docs suggest flash_attention_2 for better memory
        efficiency "especially in multi-image and video scenarios" --
        not switched to by default here since this project only ever
        passes a single static image per turn (no video), and eager is
        what's already validated end-to-end for the other three loaders
        in this same pipeline. Worth trying flash_attention_2 explicitly
        if VRAM becomes a constraint, not assumed necessary here.
        """
        kwargs = self._base_kwargs()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_id, **kwargs
        )
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        return model, processor

    def _load_gemma4(self):
        """
        Gemma 4 (Google DeepMind, released 2026-04-02, Apache 2.0).

        Loaded via AutoModelForImageTextToText rather than a named class,
        deliberately: the family ships TWO architectures --
        Gemma4ForConditionalGeneration for E2B/E4B and
        Gemma4UnifiedForConditionalGeneration for 12B/31B (confirmed by
        reading each checkpoint's own config: 12B-it reports model_type
        'gemma4_unified'). Hardcoding either class would break the other
        for no benefit, and Auto* resolves both from the config.

        Two properties that matter downstream, not just here:
          - 48 decoder layers vs Qwen3-VL's 32, hidden size 3840 vs 4096.
            extraction/hooks._resolve_target_layers takes the top 50-90%
            of the stack, so the captured band is ~27-43 here against
            18-31 for Qwen3-VL. ABSOLUTE layer indices are therefore NOT
            comparable across the two models -- any cross-architecture
            geometric claim must compare at matched RELATIVE depth.
          - Gemma4UnifiedImageProcessor exposes none of
            image_grid_pinpoints / crop_size / min_patches / max_patches,
            so attacks.geometry_adapters.get_adapter() will raise for it.
            Text attacks and non-gradient image attacks run unaffected;
            the three pixel-space attacks need a new adapter first.
        """
        kwargs = self._base_kwargs()
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id, **kwargs
        )
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        return model, processor


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def load_llavanext_7b(hpc: bool = True) -> Tuple[torch.nn.Module, Any]:
    """bf16, like every other loader (audit 14.1, P2a). It was fp16, which
    has a fraction of bf16's dynamic range: a precision difference between
    models is a confound for any cross-model geometric comparison."""
    cfg = LoadConfig(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        dtype=torch.bfloat16 if hpc else torch.float32,
        device_map="auto" if hpc else "cpu",
        attn_implementation="eager",
    )
    return VLMLoader(cfg).load()


def load_llavanext_13b(hpc: bool = True, quantise: bool = False) -> Tuple[torch.nn.Module, Any]:
    cfg = LoadConfig(
        model_id="llava-hf/llava-v1.6-vicuna-13b-hf",
        dtype=torch.bfloat16,
        device_map="auto" if hpc else "cpu",
        load_in_4bit=quantise,
        attn_implementation="eager",
    )
    return VLMLoader(cfg).load()


def load_qwen2vl_7b(hpc: bool = True) -> Tuple[torch.nn.Module, Any]:
    cfg = LoadConfig(
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto" if hpc else "cpu",
        attn_implementation="eager",
    )
    return VLMLoader(cfg).load()


def load_internvl_8b(hpc: bool = True) -> Tuple[torch.nn.Module, Any]:
    """
    InternVL3-8B-hf (native transformers support) -- see SCOPING_DECISIONS.md
    Decision 3 for why this replaces the original InternVL2-8B target.
    """
    cfg = LoadConfig(
        model_id="OpenGVLab/InternVL3-8B-hf",
        dtype=torch.bfloat16,
        device_map="auto" if hpc else "cpu",
        attn_implementation="eager",
    )
    return VLMLoader(cfg).load()


def load_gemma4_12b(hpc: bool = True) -> Tuple[torch.nn.Module, Any]:
    """
    Gemma 4 12B-it -- second defensible architecture, added so RQ1's
    cross-architecture claim rests on two current models rather than one.

    Differs from the other three in ways that matter downstream:
      - 48 transformer layers vs Qwen3-VL's 32, so _resolve_target_layers'
        top-50-90% band is ~27-43 here, not 18-31. Absolute layer indices
        are NOT comparable across models; compare at matched RELATIVE
        depth.
      - Gemma4UnifiedImageProcessor exposes none of image_grid_pinpoints,
        crop_size, min_patches or max_patches, so none of the existing
        geometry adapters match and get_adapter() will raise. Text-based
        and image-producing-but-non-gradient attacks run fine; the three
        pixel-space attacks need a new adapter first.
      - ~24GB at bf16: biggpu only, will not fit bigbatch's 24GB cards.

    bfloat16 and eager attention to match every other loader here --
    the fp16/bf16 split between this file and the VWA-Adv driver was a
    real confound and is not to be reintroduced.
    """
    cfg = LoadConfig(
        model_id="google/gemma-4-12B-it",
        dtype=torch.bfloat16,
        device_map="auto" if hpc else "cpu",
        attn_implementation="eager",
    )
    return VLMLoader(cfg).load()


def load_qwen3vl_8b(hpc: bool = True) -> Tuple[torch.nn.Module, Any]:
    """
    Qwen3-VL-8B-Instruct -- added as a 4th model specifically to test
    whether the ~zero-ASR / never-exceeds-2-turns pattern seen with
    LLaVA-NeXT 7B on real AgentDojo data is a base-model limitation or
    something more structural to the pipeline/benchmark. See
    VLMLoader._load_qwen3vl's docstring for what's verified (class
    import against a real transformers install) versus not (the actual
    weight download and forward pass, which needs cluster-side
    verification -- this environment has no huggingface.co access).
    """
    cfg = LoadConfig(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto" if hpc else "cpu",
        attn_implementation="eager",
    )
    return VLMLoader(cfg).load()
