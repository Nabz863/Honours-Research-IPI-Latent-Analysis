"""
attacks/base.py
----------------
Common interface for all IPI attack implementations (Table 3.2).

Every attack module exposes a class implementing BaseAttack.generate(),
which takes ONE clean task input + a target behaviour and returns ONE
adversarial instance. The experiment runner (experiments/run_attack_matrix.py)
is what applies a given attack across many tasks and models to build
the full attack matrix -- individual attack modules stay single-purpose
and testable in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class AttackModality(str, Enum):
    TEXTUAL = "textual"
    VISUAL = "visual"
    CROSS_MODAL = "cross_modal"
    ADAPTIVE = "adaptive"


@dataclass
class AttackResult:
    """Container for a single generated attack instance."""

    attack_name: str
    modality: AttackModality
    adversarial_text: Optional[str] = None
    adversarial_image: Optional[Any] = None  # PIL.Image or tensor, for visual attacks
    target_behavior: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    """
    Abstract base every attack in Table 3.2 implements.

    Subclasses implement generate() only. Everything about WHICH tasks
    to run against, and how many, lives in the experiment runner --
    not here. This keeps each attack module a single, unit-testable
    piece of logic regardless of how large the eventual experimental
    matrix grows.
    """

    name: str
    modality: AttackModality

    @abstractmethod
    def generate(
        self,
        base_input: Dict[str, Any],
        target_behavior: str,
        **kwargs,
    ) -> AttackResult:
        """
        Produce one adversarial instance from a clean base task input.

        Parameters
        ----------
        base_input : dict
            The clean task input as provided by the benchmark environment
            (e.g. a VWA-Adv task dict with page content / screenshot, or
            an AgentDojo task dict with tool outputs). Field names should
            match whatever the real benchmark loader emits.
        target_behavior : str
            The adversary's goal for this instance -- drawn from the
            benchmark's own attacker-goal definitions, not invented
            per-attack, so ASR stays comparable to published baselines.
        """
        raise NotImplementedError
