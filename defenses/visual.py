"""
defenses/visual.py
------------------
Visual-channel defences: transformations applied to the IMAGE the model
receives, rather than to text.

WHY A SEPARATE INTERFACE IS NEEDED
    defenses.base.BaseDefense.apply() takes `content: str` and returns a
    DefenseResult wrapping transformed text. Every existing defence
    (spotlighting's three variants) is a text transformation, so that
    signature was sufficient. A visual defence cannot use it: there is
    no text to transform, and returning a PIL image through a field
    typed as str would be a lie the type system would not catch.

    ImageDefenseResult and BaseDefense.apply_image() below extend the
    interface rather than replacing it. apply_image() defaults to a
    pass-through, so every existing text defence keeps working unchanged
    and a runner can call both without branching on defence type.

SCOPE: VWA-ADV ONLY
    AgentDojo has no image channel at all -- its untrusted content
    arrives as tool-result text. Visual defences are therefore reported
    for VWA-Adv only, and that is a property of the environments, not an
    omission.

WHY GAUSSIAN BLUR IS SCIENTIFICALLY INTERESTING HERE
    It is not merely another defence to add to a table. Blur is a
    low-pass filter, so it makes a DIFFERENTIAL prediction across this
    project's attack set:

      should be degraded      steganographic (payload encoded in
                              least-significant bits, i.e. the highest
                              spatial frequencies present)
                              pixel_perturbation (Lp-bounded noise,
                              broadband but dominated by high frequency)
                              adversarial_patch (fine-grained texture)

      should survive          typographic (rendered text is large and
                              low-frequency; a legible caption stays
                              legible under moderate blur)
                              template_ipi, gcg, comet's text channel
                              (no image dependence at all)

    A defence that suppresses every attack equally tells you little. One
    that suppresses exactly the high-frequency attacks and leaves the
    low-frequency ones intact is evidence about the MECHANISM by which
    each attack carries its payload -- which is the question this thesis
    is about. If blur degrades typographic as much as steganographic,
    that prediction is wrong and the result is more interesting still.

SIGMA
    Default sigma=1.5 px at the victim image's native resolution
    (~335x497 for the curated Reddit tasks). Chosen a priori, not tuned:
    it is large enough that the kernel's effective support spans several
    pixels and therefore destroys per-pixel LSB structure, and small
    enough that text at typical caption sizes remains readable. It is
    NOT optimised against this project's own results -- doing so would
    mean selecting the defence strength that most flattered the defence.
    The utility cost it imposes is measured, not assumed, via the
    defended-clean pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageDefenseResult:
    """Image-channel analogue of defenses.base.DefenseResult.

    transformed_image : the image the model should receive instead of
        the original. Same type as the input (PIL.Image), so callers
        need no conversion.
    system_prompt_addendum : kept for symmetry with DefenseResult -- a
        visual defence may still want to tell the model what was done,
        e.g. "the image has been filtered".
    metadata : whatever the defence wants recorded on the row.
    """
    transformed_image: Any
    system_prompt_addendum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GaussianBlurDefense:
    """Low-pass filtering of the untrusted image.

    Implements the visual pre-processing defence described in the
    proposal's defence table. The mechanism is standard and predates
    prompt injection: high-frequency adversarial structure does not
    survive a low-pass filter, while the low-frequency content that
    carries semantics largely does.

    Deliberately NOT a subclass of BaseDefense's text path: apply() here
    returns text unchanged, because this defence has nothing to say
    about text. A runner that only calls apply() would therefore apply
    NO defence -- the same trap ICON documents -- so the runner must
    call apply_image(). The assertion in apply() below makes that
    failure loud rather than silent.
    """

    # PAGE-LEVEL (audit 13.4, decision D3, 30 Sep): the blur is applied to
    # the page's IMAGES -- in the browser before each screenshot (CSS
    # filter, sigma in rendered pixels) and to every image the
    # self-captioner reads -- never to the whole screenshot, whose text and
    # set-of-marks labels the agent needs to act. The backend skips
    # apply_image() for page-level defences; the runner calls page_css()
    # and blur_pil() instead.
    page_level = True

    def __init__(self, sigma: float = 1.5, announce: bool = False):
        self.sigma = float(sigma)
        self.announce = bool(announce)

    def page_css(self) -> str:
        """CSS applied to every <img> before each observation. CSS
        blur(r) takes r as the Gaussian's standard deviation in CSS
        pixels, i.e. sigma at the size the page renders the image."""
        return f"img {{ filter: blur({self.sigma:g}px) !important; }}"

    def blur_pil(self, image):
        """The same blur for an image the captioner reads. The victim
        images are served at the size the page renders them (A1: stored
        size == rendered box), so sigma means the same thing in both
        places."""
        from PIL import ImageFilter
        return image.filter(ImageFilter.GaussianBlur(radius=self.sigma))

    @property
    def name(self) -> str:
        return f"gaussian_blur_sigma{self.sigma:g}"

    # -- text path: explicitly a no-op, and says so ---------------------

    def apply(self, content: str):
        from .base import DefenseResult
        return DefenseResult(
            defense_name=self.name,
            transformed_content=content,
            system_prompt_addendum=None,
            metadata={
                "defense": self.name,
                "text_path": "no-op",
                "warning": ("GaussianBlurDefense transforms IMAGES. A runner "
                            "that only calls apply() applies no defence at "
                            "all. Call apply_image()."),
            },
        )

    # -- image path: the actual defence ---------------------------------

    def apply_image(self, image) -> ImageDefenseResult:
        from PIL import ImageFilter

        if image is None:
            return ImageDefenseResult(transformed_image=None,
                                      metadata={"defense": self.name,
                                                "applied": False,
                                                "reason": "no image"})

        original_size = tuple(getattr(image, "size", ()) or ())
        blurred = image.filter(ImageFilter.GaussianBlur(radius=self.sigma))

        return ImageDefenseResult(
            transformed_image=blurred,
            system_prompt_addendum=(
                "Images in this conversation have been filtered before "
                "being shown to you." if self.announce else None),
            metadata={
                "defense": self.name,
                "applied": True,
                "sigma": self.sigma,
                "image_size": original_size,
            },
        )


def _add_image_path_to_base() -> None:
    """Give BaseDefense a default apply_image() so text defences remain
    valid image defences (identity), and a runner can call apply_image()
    unconditionally without checking the defence's type.

    Done here rather than by editing base.py so the existing defence
    hierarchy is untouched: if this module is never imported, nothing in
    the project changes.
    """
    from .base import BaseDefense

    if hasattr(BaseDefense, "apply_image"):
        return

    def apply_image(self, image) -> ImageDefenseResult:
        # Identity. A text defence does not touch the image channel, and
        # saying so explicitly is better than a runner special-casing
        # which defences have an image path.
        return ImageDefenseResult(
            transformed_image=image,
            metadata={"defense": getattr(self, "name", type(self).__name__),
                      "applied": False, "reason": "text-only defence"},
        )

    BaseDefense.apply_image = apply_image


_add_image_path_to_base()


def build_visual_defense_registry() -> Dict[str, Any]:
    """Named per sigma, not one generic "gaussian_blur" entry, for the
    same reason spotlighting is named per variant: filter strength
    trades robustness against utility, and collapsing that into one
    label would hide the trade-off this study is trying to measure."""
    return {
        "gaussian_blur_1.5": GaussianBlurDefense(sigma=1.5),
        "gaussian_blur_3.0": GaussianBlurDefense(sigma=3.0),
    }


if __name__ == "__main__":
    # Offline self-test. No model, no cluster: checks that the filter
    # actually removes high-frequency content and preserves low, which
    # is the whole premise of the differential prediction above.
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    h, w = 128, 128

    # high-frequency: per-pixel noise, i.e. what an LSB or Lp attack looks like
    hi = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    # low-frequency: broad blocks, i.e. what rendered text looks like
    lo = np.zeros((h, w, 3), dtype=np.uint8)
    lo[:, : w // 2] = 255

    d = GaussianBlurDefense(sigma=1.5)
    for label, arr in (("high-frequency", hi), ("low-frequency", lo)):
        before = Image.fromarray(arr)
        after = d.apply_image(before).transformed_image
        a = np.asarray(before, dtype=float)
        b = np.asarray(after, dtype=float)
        # per-pixel gradient magnitude as a crude high-frequency proxy
        def energy(x):
            return float(np.abs(np.diff(x, axis=0)).mean()
                         + np.abs(np.diff(x, axis=1)).mean())
        print(f"{label:16s} hf energy {energy(a):8.3f} -> {energy(b):8.3f} "
              f"({100 * (1 - energy(b) / max(energy(a), 1e-9)):5.1f}% removed)")
    print("\nExpected: high-frequency energy largely removed, low-frequency "
          "energy mostly retained.")
