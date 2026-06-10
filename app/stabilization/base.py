"""
Base interface and pipeline combinator for transcript stabilizers.

To add a new strategy: subclass BaseStabilizer, implement stabilize() and
reset(), then drop the file here.  No other files need changing unless you
want it wired as the session default.

To combine strategies (e.g. hard-length guard then frozen-prefix):

    stabilizer = StabilizerPipeline(
        HardLengthStabilizer(),
        FrozenPrefixStabilizer(freeze_threshold=3),
    )

Each stage receives the previous stage's output as new_hypothesis; the
original committed text (previous_text) is passed unchanged to every stage
so each can make independent decisions against the same baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseStabilizer(ABC):
    """Common interface for all transcript stabilizer implementations."""

    @abstractmethod
    def stabilize(self, new_hypothesis: str, previous_text: str = "") -> str:
        """
        Decide what to emit given a new ASR hypothesis.

        Args:
            new_hypothesis: Latest rolling hypothesis from the ASR engine.
            previous_text:  Last committed stable output.

        Returns:
            The text to use as the new stable output.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state; call between utterances."""


class StabilizerPipeline(BaseStabilizer):
    """
    Chains multiple stabilizers left-to-right.

    The output of stage N becomes new_hypothesis for stage N+1.
    previous_text (the committed baseline) is forwarded unchanged to every
    stage so that each one can independently compare against the last-known
    good output.

    Example::

        pipeline = StabilizerPipeline(
            HardLengthStabilizer(),
            FrozenPrefixStabilizer(freeze_threshold=3),
        )
    """

    def __init__(self,
                 *stabilizers: BaseStabilizer) -> None:
        if not stabilizers:
            raise ValueError("StabilizerPipeline requires at least one stabilizer")
        self._stabilizers: List[BaseStabilizer] = list(stabilizers)

    def stabilize(self,
                  new_hypothesis: str,
                  previous_text: str = "") -> str:
        current = new_hypothesis
        for s in self._stabilizers:
            current = s.stabilize(current, previous_text)
        return current

    def reset(self) -> None:
        for s in self._stabilizers:
            s.reset()
