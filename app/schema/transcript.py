from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class TranscriptMessage(BaseModel):
    """WebSocket message for transcript results."""
    type: Literal["transcript"]
    text: str
    is_final: bool = False
    confidence: Optional[float] = None
    timestamp: Optional[datetime] = None

class TranscriptResponse(BaseModel):
    """Response model for transcript data."""
    text: str
    is_final: bool
    confidence: Optional[float] = None
    session_id: str
    timestamp: datetime
