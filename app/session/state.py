"""Per-session mutable state: audio buffer, VAD FSM, and transcript accumulator."""

import asyncio
from typing import Optional
from datetime import datetime
from app.audio.buffer import RingAudioBuffer
from app.core.config import settings
from app.stabilization.factory import create_stabilizer


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

    def __init__(self) -> None:
        self.partial_transcript = ""
        self.final_transcript = ""
        # Per-session stabilizer — each session owns its own frozen-prefix state
        # so sessions never share hypothesis history or frozen regions.
        self.stabilizer = create_stabilizer()

    def update_partial(self, stabilized_text: str) -> None:
        """Store the latest stabilized hypothesis as the current partial transcript."""
        self.partial_transcript = stabilized_text

    def finalize(self) -> None:
        """Promote partial_transcript into final_transcript; triggered when VAD detects end of speech."""
        if self.partial_transcript:
            self.final_transcript += " " + self.partial_transcript.strip()
            self.partial_transcript = ""
            # Reset stabilizer so the frozen prefix from this utterance
            # does not carry over into the next one.
            self.stabilizer.reset()

    def reset(self) -> None:
        """Discard all transcript state; called on 'start' control message."""
        self.partial_transcript = ""
        self.final_transcript = ""
        self.stabilizer.reset()


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

        # Per-session inference pipeline: bounded queue + background worker task.
        # Audio windows are snapshot-enqueued by the receive loop; the worker
        # drains them independently so receive is never blocked by ASR latency.
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.INFERENCE_QUEUE_MAXSIZE)
        self.inference_task: Optional[asyncio.Task] = None

        # Backpressure tracking: count dropped windows and rate-limit client signals.
        self.dropped_windows: int = 0
        self.last_backpressure_signal: Optional[datetime] = None
    
    def should_signal_backpressure(self, now: datetime, min_interval_s: float = 1.0) -> bool:
        """True if enough time has passed since the last backpressure signal to the client."""
        if self.last_backpressure_signal is None:
            return True
        return (now - self.last_backpressure_signal).total_seconds() >= min_interval_s

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
        self.dropped_windows = 0
        self.last_backpressure_signal = None
        # Drain the queue so the worker doesn't process stale windows after reset.
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
