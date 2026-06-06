import asyncio
import concurrent.futures
import numpy as np
from datetime import datetime
from typing import Optional
from app.websocket.manager import ConnectionManager
from app.session.state import StreamingSession
from app.session.manager import SessionManager
from app.services.streaming_service import StreamingService
from app.services.transcription_service import TranscriptionService
from app.services.session_service import SessionService
from app.services.stabilization_service import StabilizationService
from app.vad.silero_vad import SileroVAD
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("StreamingHandler")

class StreamingHandler:
    """Handles streaming audio processing and transcription."""

    def __init__(self,
                 connection_manager: ConnectionManager,
                 session_manager: SessionManager,
                 vad_pool: Optional[asyncio.Queue] = None,
                 vad_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
                 inference_semaphore: Optional[asyncio.Semaphore] = None):
        self.connection_manager = connection_manager
        self.session_manager = session_manager
        self.session_service = SessionService(session_manager, connection_manager)
        self.streaming_service = StreamingService()
        self.transcription_service = TranscriptionService()
        self.stabilization_service = StabilizationService()
        self.inference_semaphore = inference_semaphore or asyncio.Semaphore(settings.ASR_SEMAPHORE_LIMIT)

        if vad_pool is None:
            # Fallback for tests / standalone use: build a single-instance pool.
            vad_pool = asyncio.Queue()
            vad_pool.put_nowait(SileroVAD())
            vad_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vad"
            )
        self.vad_pool = vad_pool
        self.vad_executor = vad_executor
        # Reference for non-inference calls (segments_from_probs is pure Python,
        # no lock or ONNX — safe to share while inference runs on another instance).
        self._vad_ref: SileroVAD = self.vad_pool.get_nowait()
        self.vad_pool.put_nowait(self._vad_ref)

        logger.info("StreamingHandler initialized")

    # ------------------------------------------------------------------
    # Public API — called from websocket_router
    # ------------------------------------------------------------------

    def start_inference_worker(self, session: StreamingSession) -> None:
        """Spawn the per-session inference worker task."""
        session.inference_task = asyncio.create_task(
            self._inference_worker(session),
            name=f"inference-{session.session_id}",
        )
        logger.info(f"[{session.session_id}] Inference worker started")

    async def handle_audio_packet(self,
                                  session_id: str,
                                  audio_data: np.ndarray) -> None:
        """
        Receive an audio packet and enqueue an inference window if the
        pacing interval has elapsed.

        The receive loop never awaits ASR; it only snapshot-enqueues a
        pre-captured audio window.  The background worker drains the queue.
        """
        session = self.session_service.get_session(session_id)
        if not session:
            logger.warning(f"Audio packet received for unknown session: {session_id}")
            return

        self.streaming_service.process_audio_packet(session, audio_data)

        if self.streaming_service.should_run_inference(session):
            audio_window = self.streaming_service.get_inference_window(session)
            if len(audio_window) == 0:
                return

            try:
                session.audio_queue.put_nowait(audio_window)
                # Update pacing timestamp at enqueue time so the interval
                # is measured from when inference was requested, not completed.
                session.last_inference_time = datetime.now()
                session.inference_count += 1
                logger.debug(
                    f"[{session_id}] Enqueued inference #{session.inference_count} "
                    f"(queue size: {session.audio_queue.qsize()})"
                )
            except asyncio.QueueFull:
                # Worker is falling behind — drop oldest window to stay real-time.
                logger.debug(f"[{session_id}] Inference queue full — dropping window")

    async def handle_control_message(self,
                                     session_id: str,
                                     action: str) -> None:
        """Handle control messages (start / stop)."""
        session = self.session_manager.get_session(session_id)
        if not session:
            logger.warning(f"Control message '{action}' for unknown session: {session_id}")
            return

        logger.info(f"[{session_id}] Control action: '{action}'")

        if action == "stop":
            if session.transcript_state.partial_transcript:
                session.transcript_state.finalize()
                final_text = session.transcript_state.final_transcript.strip()
                logger.info(f"[{session_id}] Finalized on stop: '{final_text}'")
                await self.connection_manager.send_transcript(
                    session.session_id, final_text, is_final=True
                )

        elif action == "start":
            # Reset audio + transcript state and drain stale queue items.
            session.audio_buffer.clear()
            session.vad_state.reset()
            session.transcript_state.reset()
            logger.info(f"[{session_id}] Session state reset for new recording")

        else:
            logger.warning(f"[{session_id}] Unknown control action: '{action}'")

    async def cleanup_session(self, session_id: str) -> None:
        """Cancel the inference worker and clean up session resources."""
        logger.info(f"[{session_id}] Cleaning up session")
        session = self.session_manager.get_session(session_id)

        if session:
            # Cancel the worker first so it doesn't race with teardown.
            await self._stop_inference_worker(session)

            if session.transcript_state.partial_transcript:
                session.transcript_state.finalize()
                final_text = session.transcript_state.final_transcript.strip()
                logger.info(f"[{session_id}] Finalized on cleanup: '{final_text}'")
                await self.connection_manager.send_transcript(
                    session.session_id, final_text, is_final=True
                )

        self.session_manager.remove_session(session_id)
        self.connection_manager.disconnect(session_id)
        logger.info(f"[{session_id}] Session cleaned up")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _stop_inference_worker(self, session: StreamingSession) -> None:
        """Cancel and await the inference worker task for a clean shutdown."""
        task = session.inference_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        session.inference_task = None

    async def _inference_worker(self, session: StreamingSession) -> None:
        """
        Per-session background coroutine that drains the audio_queue and
        runs VAD + ASR inference under the global semaphore.

        Runs until cancelled (on session cleanup or 'start' reset).
        Individual inference errors are logged and swallowed so the worker
        stays alive for subsequent windows.
        """
        session_id = session.session_id
        logger.debug(f"[{session_id}] Inference worker running")
        try:
            while True:
                audio_window = await session.audio_queue.get()
                try:
                    async with self.inference_semaphore:
                        await self._run_inference(session, audio_window)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"[{session_id}] Inference error: {e}")
        except asyncio.CancelledError:
            logger.debug(f"[{session_id}] Inference worker cancelled")

    async def _run_vad(self, audio_window: np.ndarray) -> tuple:
        """
        Checkout a VAD instance from the pool, run is_speech() on a dedicated
        thread, then return the instance to the pool.

        Using run_in_executor releases the event loop during ONNX inference so
        all 200 receive loops remain responsive.  VAD_POOL_SIZE instances run
        truly in parallel — no shared threading.Lock contention between them.
        """
        vad = await self.vad_pool.get()
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.vad_executor,
                vad.is_speech,
                audio_window,
                settings.VAD_TRIGGER_STRATEGY,
            )
        finally:
            self.vad_pool.put_nowait(vad)

    def _trim_to_speech(self,
                        audio_window: np.ndarray,
                        probs: list) -> np.ndarray:
        """Return the sub-array of audio_window that spans detected speech."""
        segments = self._vad_ref.segments_from_probs(probs)
        if not segments:
            logger.debug("No speech segments detected — using full audio window")
            return audio_window

        padding = int(settings.SPEECH_PADDING_MS / 1000 * settings.SAMPLE_RATE)
        start_sample = max(0, int(segments[0][0] / 1000 * settings.SAMPLE_RATE) - padding)
        end_sample = min(len(audio_window), int(segments[-1][1] / 1000 * settings.SAMPLE_RATE) + padding)
        return audio_window[start_sample:end_sample]

    async def _run_inference(self,
                             session: StreamingSession,
                             audio_window: np.ndarray) -> None:
        """Run VAD and STT inference on a pre-captured audio window."""
        if len(audio_window) == 0:
            return

        is_speech, probs = await self._run_vad(audio_window)
        current_time = datetime.now()
        was_speaking = session.vad_state.is_speaking
        session.vad_state.update(is_speech, current_time)

        if is_speech and not was_speaking:
            logger.info(f"[{session.session_id}] Speech started")
        elif not is_speech and was_speaking:
            logger.info(f"[{session.session_id}] Speech ended")

        logger.debug(f"[{session.session_id}] VAD: is_speech={is_speech} is_speaking={session.vad_state.is_speaking}")

        if is_speech or session.vad_state.is_speaking:
            audio_to_transcribe = self._trim_to_speech(audio_window, probs)
            logger.debug(
                f"[{session.session_id}] Sending {len(audio_to_transcribe)} samples to ASR "
                f"(trimmed from {len(audio_window)})"
            )

            transcript = await self.transcription_service.atranscribe(audio_to_transcribe)

            if transcript:
                stabilized = self.stabilization_service.stabilize(
                    transcript,
                    session.transcript_state.partial_transcript
                )

                if stabilized.strip() != session.transcript_state.partial_transcript.strip():
                    session.transcript_state.update_partial(stabilized)
                    logger.info(f"[{session.session_id}] Partial transcript: '{stabilized}'")
                    await self.connection_manager.send_transcript(
                        session.session_id, stabilized, is_final=False
                    )
            else:
                logger.debug(f"[{session.session_id}] ASR returned empty transcript")

        if not session.vad_state.is_speaking and session.transcript_state.partial_transcript:
            session.transcript_state.finalize()
            final_text = session.transcript_state.final_transcript.strip()
            logger.info(f"[{session.session_id}] Final transcript: '{final_text}'")
            await self.connection_manager.send_transcript(
                session.session_id, final_text, is_final=True
            )
