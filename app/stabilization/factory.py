"""
Factory for creating stabilizer instances from application config.

Supported strategies (STABILIZER_STRATEGY):
    frozen_prefix     — rollback suppression + progressive commit (default)
    hard_length       — monotonic word-count guard; no corrections allowed
    hard_then_frozen  — hard-length gate first, then frozen-prefix commit

Usage::

    from app.stabilization.factory import create_stabilizer
    stabilizer = create_stabilizer()
"""

from __future__ import annotations

from app.stabilization.base import BaseStabilizer, StabilizerPipeline
from app.stabilization.rollback_suppression import FrozenPrefixStabilizer, HardLengthStabilizer

_REGISTRY: dict[str, type[BaseStabilizer]] = {
    "frozen_prefix": FrozenPrefixStabilizer,
    "hard_length": HardLengthStabilizer,
}


def create_stabilizer() -> BaseStabilizer:
    """
    Instantiate the stabilizer configured in settings.

    Reads STABILIZER_STRATEGY, STABILIZER_MODE, and STABILIZER_FREEZE_THRESHOLD
    from the application settings.  Raises ValueError for unknown strategies.

    Returns:
        A ready-to-use BaseStabilizer instance.
    """
    from app.core.config import settings

    strategy = settings.STABILIZER_STRATEGY
    mode = settings.STABILIZER_MODE
    freeze_threshold = settings.STABILIZER_FREEZE_THRESHOLD

    if strategy == "frozen_prefix":
        return FrozenPrefixStabilizer(freeze_threshold=freeze_threshold, mode=mode)

    if strategy == "hard_length":
        return HardLengthStabilizer(mode=mode)

    if strategy == "hard_then_frozen":
        return StabilizerPipeline(
            HardLengthStabilizer(mode=mode),
            FrozenPrefixStabilizer(freeze_threshold=freeze_threshold, mode=mode),
        )

    known = list(_REGISTRY) + ["hard_then_frozen"]
    raise ValueError(
        f"Unknown STABILIZER_STRATEGY '{strategy}'. Choose from: {known}"
    )
