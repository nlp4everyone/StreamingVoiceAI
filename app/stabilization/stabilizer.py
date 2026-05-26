"""
TranscriptStabilizer — high-level coordinator for streaming ASR stabilization.

Streaming ASR engines emit a rapid sequence of rolling hypotheses for the
same audio segment.  Each new hypothesis may extend, correct, or partially
contradict the previous one.  This module provides ``TranscriptStabilizer``,
which uses a pluggable LCP (longest-common-prefix) strategy to decide which
part of the latest hypothesis is safe to commit as stable output.

Typical usage::

    stabilizer = TranscriptStabilizer()
    stable = ""
    for hypothesis in asr_stream():
        stable = stabilizer.stabilize(hypothesis, stable)
"""

from typing import List, Literal
from .longest_common_prefix import (CharacterLevelLCP,
                                    WordLevelLCP)

_STRATEGIES = {
    "character_level": CharacterLevelLCP(),
    "word_level": WordLevelLCP(),
}


class TranscriptStabilizer:
    """
    Stabilizes streaming ASR transcripts using longest common prefix (LCP)
    and rolling hypothesis comparison.

    Acts as a wrapper that delegates LCP logic to a strategy adapter selected
    by the ``mode`` parameter on each call.  Add a new entry to ``_STRATEGIES``
    to introduce additional LCP approaches without changing this class.
    """

    def __init__(self, history_size: int = 5):
        """
        Args:
            history_size: Maximum number of recent hypotheses to retain.
                Kept for potential future use (e.g. majority-vote logic);
                does not affect the current stabilization algorithm.
        """
        self.history_size = history_size
        self.hypothesis_history: List[str] = []

    def stabilize(self,
                  new_hypothesis: str,
                  previous_text: str = "",
                  mode: Literal["character_level", "word_level"] = "word_level") -> str:
        """
        Stabilize a new transcript hypothesis against previously committed text.

        Args:
            new_hypothesis: Latest rolling hypothesis from the ASR engine.
            previous_text: Last committed stable transcript.
            mode: LCP granularity — ``"word_level"`` (default) is
                recommended for Vietnamese; ``"character_level"`` gives
                finer precision for Latin-script languages.

        Returns:
            Updated stable transcript string.
        """
        if not new_hypothesis:
            return previous_text
        if not previous_text:
            return new_hypothesis

        self.hypothesis_history.append(new_hypothesis)
        if len(self.hypothesis_history) > self.history_size:
            self.hypothesis_history.pop(0)

        impl = _STRATEGIES[mode]
        prefix = impl.lcp(new_hypothesis, previous_text)

        if impl.starts_with(new_hypothesis, previous_text):  # case 3
            return new_hypothesis
        if impl.starts_with(previous_text, prefix):           # case 4
            return new_hypothesis
        if prefix:                                             # case 5
            return prefix + impl.suffix_after(new_hypothesis, prefix)
        return new_hypothesis                                  # case 6

    def get_stable_prefix(self,
                          hypotheses: List[str],
                          mode: Literal["character_level", "word_level"] = "word_level") -> str:
        """
        Return the longest prefix shared by every hypothesis in *hypotheses*.

        Useful when the caller has collected several consecutive engine
        outputs and wants to commit only the portion that has stayed
        stable across all of them.  Reduces progressively: folds the
        first two hypotheses into a prefix, then intersects that with
        the third, and so on — short-circuits as soon as the prefix
        becomes empty.

        Args:
            hypotheses: Ordered list of transcript hypotheses.
            mode: LCP granularity, same values as :meth:`stabilize`.

        Returns:
            Shared stable prefix; empty string if the first two
            hypotheses share no common tokens/characters.
        """
        if not hypotheses:
            return ""
        if len(hypotheses) == 1:
            return hypotheses[0]

        impl = _STRATEGIES[mode]
        stable = hypotheses[0]
        for hypothesis in hypotheses[1:]:
            stable = impl.lcp(stable, hypothesis)
            if not stable:
                break
        return stable

    def reset(self) -> None:
        """Clear rolling hypothesis history.

        Call between utterances or sessions so stale context from a
        previous speaker or recording does not influence stabilization.
        """
        self.hypothesis_history.clear()
