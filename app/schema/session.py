from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

class SessionStatusResponse(BaseModel):
    """Typed output of SessionService.get_session_status()."""

    model_config = ConfigDict(from_attributes=True)

    session_id: str
    is_active: bool

    # Audio buffer fullness in seconds
    audio_buffer_seconds: float = Field(ge=0.0)

    # VAD state
    is_speaking: bool
    silence_duration_ms: float = Field(ge=0.0)

    # Transcript state
    partial_transcript: str
    final_transcript: str

    # Timestamps
    created_at: datetime
    last_activity: datetime

    # Inference stats
    inference_count: int = Field(ge=0)


# Backward-compat alias
SessionStatus = SessionStatusResponse
