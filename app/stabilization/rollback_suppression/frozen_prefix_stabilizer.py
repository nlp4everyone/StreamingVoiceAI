"""
Frozen-prefix stabilizer — rollback suppression + progressive commit.

Splits the running transcript into two regions:
  - frozen prefix : text committed by N consecutive agreements; immutable.
  - unstable tail : text that may still change between ASR hypotheses.

Rollback suppression:
    Any hypothesis that contradicts the frozen prefix is silently rejected;
    the previous output is returned unchanged.

Progressive commit (freeze):
    When `freeze_threshold` consecutive hypotheses share a longer common
    prefix than the current frozen region, that longer prefix is committed.

Reset policy:
    Call reset() at the end of each utterance (on finalize or session start)
    so the frozen prefix from a previous speaker does not bleed into the next.
"""

from __future__ import annotations

from typing import List, Literal

from app.stabilization.base import BaseStabilizer
from app.stabilization.longest_common_prefix import CharacterLevelLCP, WordLevelLCP

_STRATEGIES = {
    "character_level": CharacterLevelLCP(),
    "word_level": WordLevelLCP(),
}


class FrozenPrefixStabilizer(BaseStabilizer):
    """
    Stabilizer that protects committed text from rollback and progressively
    extends the frozen region as consecutive hypotheses agree.

    Parameters
    ----------
    freeze_threshold:
        Number of consecutive hypotheses that must agree on a prefix before
        it is frozen.  Lower values freeze earlier (less latency, higher risk
        of premature commit); higher values are more conservative.  3 is a
        sensible default for Vietnamese streaming ASR at ~200 ms intervals.
    mode:
        LCP comparison granularity.  ``"word_level"`` (default) is recommended
        for Vietnamese; ``"character_level"`` for Latin-script languages.
    """

    def __init__(self,
                 freeze_threshold: int = 3,
                 mode: Literal["character_level", "word_level"] = "word_level") -> None:
        self.freeze_threshold = freeze_threshold
        self.mode = mode
        self._impl = _STRATEGIES[mode]
        self._frozen: str = ""
        self._last_output: str = ""
        self._history: List[str] = []

    @property
    def frozen_prefix(self) -> str:
        """Currently committed (immutable) prefix."""
        return self._frozen

    def stabilize(self,
                  new_hypothesis: str,
                  previous_text: str = "") -> str:
        """
        Stabilize a new hypothesis.

        Args:
            new_hypothesis: Latest rolling hypothesis from the ASR engine.
            previous_text:  Unused by this stabilizer (state is tracked
                            internally); kept for interface compatibility.

        Returns:
            ``new_hypothesis`` if it respects the frozen prefix, otherwise
            the last accepted output.
        """
        if not new_hypothesis:
            return self._last_output

        impl = self._impl

        # --- Rollback suppression ----------------------------------------
        # Reject any hypothesis that contradicts the frozen prefix.
        if self._frozen and not impl.starts_with(new_hypothesis, self._frozen):
            return self._last_output

        # --- History tracking --------------------------------------------
        self._history.append(new_hypothesis)
        # Keep only the window needed for freeze decisions plus a small buffer.
        if len(self._history) > self.freeze_threshold + 2:
            self._history.pop(0)

        # --- Progressive commit ------------------------------------------
        # When enough consecutive hypotheses agree on a longer common prefix,
        # extend the frozen region.  Never shrink it.
        if len(self._history) >= self.freeze_threshold:
            recent = self._history[-self.freeze_threshold:]
            candidate = recent[0]
            for h in recent[1:]:
                candidate = impl.lcp(candidate, h)
                if not candidate:
                    break

            if candidate:
                if not self._frozen:
                    self._frozen = candidate
                elif (
                    impl.starts_with(candidate, self._frozen)
                    and candidate != self._frozen
                ):
                    self._frozen = candidate

        self._last_output = new_hypothesis
        return new_hypothesis

    def reset(self) -> None:
        """Clear all state; call between utterances or on session start."""
        self._frozen = ""
        self._last_output = ""
        self._history.clear()
