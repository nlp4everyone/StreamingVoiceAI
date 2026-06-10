from typing import List
from app.stabilization.base import BaseStabilizer
from app.utils.logger import setup_logger

logger = setup_logger("StabilizationService")


class StabilizationService:
    """
    Stateless coordination layer for transcript stabilization.

    Stabilizers are per-session objects (stored on TranscriptState) so this
    service never holds mutable state of its own — it delegates every call to
    the stabilizer the caller supplies.  Swap the stabilizer by passing a
    different BaseStabilizer subclass to TranscriptState; this service
    requires no changes.
    """

    def stabilize(self, stabilizer: BaseStabilizer, new_hypothesis: str) -> str:
        """
        Stabilize a new hypothesis using the provided per-session stabilizer.

        Args:
            stabilizer:     The session's own stabilizer instance.
            new_hypothesis: Latest rolling hypothesis from the ASR engine.

        Returns:
            Stabilized transcript string.
        """
        result = stabilizer.stabilize(new_hypothesis)
        logger.debug("Stabilize: '%s' -> '%s'", new_hypothesis, result)
        return result

    def get_stable_prefix(self, hypotheses: List[str]) -> str:
        """
        Return the longest prefix shared by all hypotheses in the list.

        Stateless utility — does not depend on a session stabilizer.  Useful
        when the caller has collected several consecutive engine outputs and
        wants the portion stable across all of them.

        Args:
            hypotheses: Ordered list of transcript hypotheses.

        Returns:
            Shared stable prefix; empty string if none.
        """
        from app.stabilization.stabilizer import TranscriptStabilizer
        prefix = TranscriptStabilizer().get_stable_prefix(hypotheses)
        logger.debug("Stable prefix from %d hypotheses: '%s'", len(hypotheses), prefix)
        return prefix
