"""
Hard-length stabilizer — monotonic word-count guard.

Rejects any hypothesis whose word count is strictly less than the last
accepted output.  This is the simplest rollback suppression strategy:
the transcript length is only allowed to grow, never shrink.

Trade-off:
    Legitimate ASR corrections that produce a shorter (but correct) output
    are also blocked.  Use FrozenPrefixStabilizer when correction tolerance
    is needed; use this as a lightweight first stage in a StabilizerPipeline
    when you want an unconditional length floor.
"""

from __future__ import annotations

from typing import Literal

from app.stabilization.base import BaseStabilizer


class HardLengthStabilizer(BaseStabilizer):
    """
    Monotonic length guard: hypothesis word count must not decrease.

    Parameters
    ----------
    mode:
        Comparison unit.  ``"word_level"`` (default) counts whitespace-
        separated tokens; ``"character_level"`` counts Unicode code points.
        Word-level is recommended for Vietnamese.
    """

    def __init__(self,
                 mode: Literal["character_level", "word_level"] = "word_level") -> None:
        self.mode = mode
        self._last_output: str = ""

    def _length(self, text: str) -> int:
        if self.mode == "word_level":
            return len(text.split())
        return len(text)

    def stabilize(self,
                  new_hypothesis: str,
                  previous_text: str = "") -> str:
        """
        Accept the hypothesis only if it is at least as long as the last output.

        Args:
            new_hypothesis: Latest rolling hypothesis from the ASR engine.
            previous_text:  Fallback baseline when no output has been accepted
                            yet in this utterance (e.g. first call after reset).

        Returns:
            ``new_hypothesis`` if its length >= current baseline, otherwise
            the current baseline unchanged.
        """
        if not new_hypothesis:
            return self._last_output

        baseline = self._last_output or previous_text

        if baseline and self._length(new_hypothesis) < self._length(baseline):
            return baseline

        self._last_output = new_hypothesis
        return new_hypothesis

    def reset(self) -> None:
        """Clear internal state; call between utterances."""
        self._last_output = ""
