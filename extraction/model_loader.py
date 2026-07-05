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

HPC notes
---------
  - fp16 by default; switch to 4-bit quantisation via load_in_4bit=True
    if VRAM is < 24 GB (note: 4-bit alters residual stream precision —
    flag any results obtained under quantisation).
  - device_map="auto" distributes across available GPUs automatically.
  - CPU fallback included for local dev / unit tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    Qwen2VLForConditionalGeneration,
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
        Weight dtype when not quantising. fp16 recommended for HPC.
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
    dtype: torch.dtype = torch.float16
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
        elif "qwen2" in mid and "vl" in mid:
            self.arch = "qwen2_vl"
            self.model, self.processor = self._load_qwen2vl()
        elif "internvl" in mid:
            self.arch = "internvl"
            self.model, self.processor = self._load_internvl()
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
                bnb_4bit_compute_dtype=torch.float16,
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
        InternVL uses AutoModelForCausalLM with trust_remote_code.
        """
        kwargs = self._base_kwargs()
        kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        return model, tokenizer


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def load_llavanext_7b(hpc: bool = True) -> Tuple[torch.nn.Module, Any]:
    cfg = LoadConfig(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        dtype=torch.float16 if hpc else torch.float32,
        device_map="auto" if hpc else "cpu",
        attn_implementation="eager",
    )
    return VLMLoader(cfg).load()


def load_llavanext_13b(hpc: bool = True, quantise: bool = False) -> Tuple[torch.nn.Module, Any]:
    cfg = LoadConfig(
        model_id="llava-hf/llava-v1.6-vicuna-13b-hf",
        dtype=torch.float16,
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
    cfg = LoadConfig(
        model_id="OpenGVLab/InternVL2-8B",
        dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="eager",
    )
    return VLMLoader(cfg).load()
