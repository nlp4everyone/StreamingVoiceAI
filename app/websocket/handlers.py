import numpy as np
from datetime import datetime
from app.websocket.manager import ConnectionManager
from app.session.state import StreamingSession
from app.session.manager import SessionManager
from app.services.streaming_service import StreamingService
from app.services.transcription_service import TranscriptionService
from app.services.session_service import SessionService
from app.services.stabilization_service import StabilizationService
from app.vad.silero_vad import SileroVAD
from app.core.config import settings

class StreamingHandler:
    """Handles streaming audio processing and transcription."""
    
    def __init__(self,
                 connection_manager: ConnectionManager,
                 session_manager: SessionManager):
        """
        Args:
            connection_manager: Manages active WebSocket connections and message dispatch.
            session_manager: Manages per-session state lifecycle.
        """
        self.connection_manager = connection_manager
        self.session_manager = session_manager
        self.session_service = SessionService(session_manager, connection_manager)
        self.streaming_service = StreamingService()
        self.transcription_service = TranscriptionService()
        self.stabilization_service = StabilizationService()
        self.vad = SileroVAD()
    
    async def handle_audio_packet(self,
                                  session_id: str,
                                  audio_data: np.ndarray) -> None:
        """
        Process incoming audio packet.
        
        Args:
            session_id: Session identifier
            audio_data: Audio data (int16)
        """
        session = self.session_service.get_session(session_id)

        if not session:
            return
        
        # Process audio packet using streaming service
        self.streaming_service.process_audio_packet(session, audio_data)
        
        # Check if we should run inference
        if self.streaming_service.should_run_inference(session):
            await self._run_inference(session)
            session.last_inference_time = datetime.now()
            session.inference_count += 1
    
    def _trim_to_speech(self,
                        audio_window: np.ndarray) -> np.ndarray:
        """Return the sub-array of audio_window that spans detected speech.

        Expands the detected speech boundaries by SPEECH_PADDING_MS on each
        side so the ASR model receives a small amount of acoustic context
        around the speech region.  Boundaries are clamped to the window so
        no zero-padding is introduced.  Falls back to the original window
        when no segments are found.
        """
        # Detect speech segments via VAD
        segments = self.vad.detect_speech_segments(audio_window)
        if not segments:
            return audio_window

        # Convert padding from ms to samples
        padding = int(settings.SPEECH_PADDING_MS / 1000 * settings.SAMPLE_RATE)
        # Clamp start/end to window boundaries
        start_sample = max(0, int(segments[0][0] / 1000 * settings.SAMPLE_RATE) - padding)
        end_sample = min(len(audio_window), int(segments[-1][1] / 1000 * settings.SAMPLE_RATE) + padding)
        return audio_window[start_sample:end_sample]

    async def _run_inference(self,
                             session: StreamingSession) -> None:
        """Run VAD and STT inference on the session."""
        # Get latest audio window using streaming service
        audio_window = self.streaming_service.get_inference_window(session)
        
        if len(audio_window) == 0:
            return
        
        # Run VAD
        is_speech = self.vad.is_speech(audio_window, strategy=settings.VAD_TRIGGER_STRATEGY)
        current_time = datetime.now()
        session.vad_state.update(is_speech, current_time)
        
        # Only run STT if speech is detected
        if is_speech or session.vad_state.is_speaking:
            # Trim to the actual speech region so the ASR model receives
            # clean input rather than a fixed-size window padded with silence.
            audio_to_transcribe = self._trim_to_speech(audio_window)

            transcript = await self.transcription_service.atranscribe(audio_to_transcribe)
            
            if transcript:
                # Stabilize transcript using stabilization service
                stabilized = self.stabilization_service.stabilize(
                    transcript,
                    session.transcript_state.partial_transcript
                )

                # Only update and send if transcript actually changed
                if stabilized.strip() != session.transcript_state.partial_transcript.strip():
                    session.transcript_state.update_partial(stabilized)
                    await self.connection_manager.send_transcript(
                        session.session_id,
                        stabilized,
                        is_final=False
                    )

        # Check if speech ended and finalize
        if not session.vad_state.is_speaking and session.transcript_state.partial_transcript:
            session.transcript_state.finalize()

            # Send final transcript
            await self.connection_manager.send_transcript(
                session.session_id,
                session.transcript_state.final_transcript.strip(),
                is_final=True
            )
    
    async def handle_control_message(self,
                                     session_id: str,
                                     action: str) -> None:
        """
        Handle control messages.
        
        Args:
            session_id: Session identifier
            action: Control action (start, stop, pause, resume)
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return
        
        if action == "stop":
            # Finalize any pending transcript
            if session.transcript_state.partial_transcript:
                session.transcript_state.finalize()
                await self.connection_manager.send_transcript(
                    session.session_id,
                    session.transcript_state.final_transcript.strip(),
                    is_final=True
                )
        elif action == "start":
            # Reset session state for new recording
            session.audio_buffer.clear()
            session.vad_state.reset()
            session.transcript_state.reset()
    
    async def cleanup_session(self,
                              session_id: str) -> None:
        """Clean up session resources."""
        session = self.session_manager.get_session(session_id)
        if session:
            # Finalize any pending transcript
            if session.transcript_state.partial_transcript:
                session.transcript_state.finalize()
                await self.connection_manager.send_transcript(
                    session.session_id,
                    session.transcript_state.final_transcript.strip(),
                    is_final=True
                )
        
        # Remove session
        self.session_manager.remove_session(session_id)
        self.connection_manager.disconnect(session_id)
