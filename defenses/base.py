"""
defenses/base.py
------------------
Common interface for all IPI defense implementations (Table 3.3).

Mirrors attacks/base.py's design deliberately: every defense module
exposes a class implementing BaseDefense.apply(), which takes ONE piece
of untrusted content and returns a DefenseResult -- the transformed
content plus whatever the defense needs added to the surrounding prompt
(e.g. an explanatory system-prompt addendum) for the transformation to
actually mean anything to the model. The experiment runner decides WHICH
content gets defended and how the addendum gets spliced into a given
benchmark's pipeline -- individual defense modules stay single-purpose
and testable in isolation, same rationale as attacks/base.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DefenseResult:
    """Container for one defended transformation of untrusted content."""

    defense_name: str
    transformed_content: str
    # Text to append to the system/instruction prompt so the model knows
    # how to interpret transformed_content (e.g. what a marker character
    # means, or that it must decode before reading). Empty string for
    # defenses that don't need one.
    system_prompt_addendum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDefense(ABC):
    """
    Abstract base every defense in Table 3.3 implements. Subclasses
    implement apply() only -- everything about WHICH content gets
    defended, and how the returned system_prompt_addendum gets spliced
    into a specific benchmark's pipeline, lives in the experiment runner,
    not here.
    """

    name: str

    @abstractmethod
    def apply(self, content: str, **kwargs) -> DefenseResult:
        """
        Transform one piece of untrusted content before it reaches the
        model.

        Parameters
        ----------
        content : str
            The untrusted content to defend -- e.g. a tool-output string
            or page-text field, potentially already carrying an injected
            payload. Defenses operate on this regardless of whether it's
            actually adversarial (a defense is applied uniformly to
            untrusted content, not only when an attack is known to be
            present -- that's the whole point of a provenance-based
            defense over a content-based detector).
        """
        raise NotImplementedError
