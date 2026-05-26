from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class AudioMessage(BaseModel):
    """WebSocket message for audio data."""
    type: Literal["audio"]
    data: str  # Base64 encoded audio data
    sample_rate: int = 16000
    timestamp: Optional[datetime] = None
