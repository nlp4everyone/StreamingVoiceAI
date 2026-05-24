"""Per-session mutable state: audio buffer, VAD FSM, and transcript accumulator."""

from typing import Optional
from datetime import datetime
from app.audio.buffer import RingAudioBuffer
from app.core.config import settings


class VADState:
    """Tracks speaking / silence transitions for one session using a time-based silence gate."""
    
    def __init__(self):
        self.is_speaking = False
        self.speech_start_time: Optional[datetime] = None
        self.speech_end_time: Optional[datetime] = None
        self.last_speech_time: Optional[datetime] = None
        self.silence_duration_ms = 0
    
    def update(self,
               is_speech: bool,
               current_time: datetime) -> None:
        """Flip is_speaking to False only after silence exceeds SILENCE_THRESHOLD_MS, adding a grace period."""
        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
            self.last_speech_time = current_time
            self.silence_duration_ms = 0
        else:
            if self.is_speaking and self.last_speech_time is not None:
                self.silence_duration_ms = (
                    (current_time - self.last_speech_time).total_seconds() * 1000
                )
                if self.silence_duration_ms >= settings.SILENCE_THRESHOLD_MS:
                    self.is_speaking = False
                    self.speech_end_time = current_time
    
    def reset(self) -> None:
        """Clear all VAD state; called on 'start' control message or session teardown."""
        self.is_speaking = False
        self.speech_start_time = None
        self.speech_end_time = None
        self.last_speech_time = None
        self.silence_duration_ms = 0


class TranscriptState:
    """Accumulates rolling partial hypotheses and promotes them to a final transcript on silence."""
    
    def __init__(self):
        self.partial_transcript = ""
        self.final_transcript = ""
        self.stable_prefix = ""
        self.unstable_suffix = ""
        self.hypothesis_history = []
        self.max_history_size = 10
    
    def update_partial(self, new_hypothesis: str) -> None:
        """Replace the current partial with a stabilized hypothesis and append it to the rolling history."""
        self.hypothesis_history.append(new_hypothesis)
        if len(self.hypothesis_history) > self.max_history_size:
            self.hypothesis_history.pop(0)
        
        self.partial_transcript = new_hypothesis
    
    def finalize(self) -> None:
        """Promote partial_transcript into final_transcript; triggered when VAD detects end of speech."""
        if self.partial_transcript:
            self.final_transcript += " " + self.partial_transcript.strip()
            self.partial_transcript = ""
            self.hypothesis_history.clear()
    
    def reset(self) -> None:
        """Discard all transcript state; called on 'start' control message."""
        self.partial_transcript = ""
        self.final_transcript = ""
        self.stable_prefix = ""
        self.unstable_suffix = ""
        self.hypothesis_history.clear()


class StreamingSession:
    """Complete state container for one connected WebSocket client."""

    def __init__(self,
                 session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Audio buffer
        self.audio_buffer = RingAudioBuffer()
        
        # VAD state
        self.vad_state = VADState()
        
        # Transcript state
        self.transcript_state = TranscriptState()

        # Tracks when the last VAD+STT cycle ran to enforce INFERENCE_INTERVAL_MS pacing.
        self.last_inference_time: Optional[datetime] = None
        self.inference_count = 0
    
    def update_activity(self) -> None:
        """Refresh the idle timestamp; used by cleanup_inactive_sessions to detect stale sessions."""
        self.last_activity = datetime.now()
    
    def is_active(self,
                  timeout_seconds: int = 300) -> bool:
        """Check if session is still active."""
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed < timeout_seconds
    
    def reset(self) -> None:
        """Reset session state."""
        self.audio_buffer.clear()
        self.vad_state.reset()
        self.transcript_state.reset()
        self.last_inference_time = None
        self.inference_count = 0
