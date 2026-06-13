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

        # 1. Append audio to ring buffer; returns True once buffer >= INFERENCE_WINDOW_SECONDS.
        buffer_ready = self.streaming_service.process_audio_packet(session, audio_data)

        # 2. Both gates must pass: enough buffered audio AND enough time since last inference.
        if buffer_ready and self.streaming_service.should_run_inference(session):
            # 3. Snapshot the latest window and hand it off to the background worker.
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
                    "[%s] Enqueued inference #%d (queue size: %d)",
                    session_id, session.inference_count, session.audio_queue.qsize(),
                )
            except asyncio.QueueFull:
                # 4. Worker is falling behind — drop window and signal client if needed.
                session.dropped_windows += 1
                now = datetime.now()
                if session.should_signal_backpressure(now):
                    session.last_backpressure_signal = now
                    await self.connection_manager.send_backpressure(
                        session_id, "queue_full", session.dropped_windows
                    )
                logger.debug(
                    "[%s] Inference queue full — dropping window (total drops: %d)",
                    session_id, session.dropped_windows,
                )

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
            # Flush partial transcript and send final result to client.
            await self._finalize_transcript(session)

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
            # 1. Cancel the worker first so it doesn't race with teardown.
            await self._stop_inference_worker(session)

            # 2. Flush any remaining partial transcript before removing state.
            await self._finalize_transcript(session)

        # 3. Remove from registry and close the WebSocket connection.
        self.session_manager.remove_session(session_id)
        self.connection_manager.disconnect(session_id)
        logger.info(f"[{session_id}] Session cleaned up")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_final_window(self, session: StreamingSession) -> np.ndarray:
        """
        Extract audio spanning the complete utterance with precise boundaries:
        [speech_start_time - SPEECH_PADDING_MS, last_speech_time + FINALIZE_RIGHT_PADDING_MS].

        Returns an empty array when timing info is unavailable or the segment
        has been evicted from the ring buffer.
        """
        vad = session.vad_state

        # 1. Bail out if VAD never recorded timing boundaries for this utterance.
        if not vad.last_speech_time or not vad.speech_start_time:
            return np.array([], dtype=np.int16)

        now = datetime.now()
        right_pad_s = settings.FINALIZE_RIGHT_PADDING_MS / 1000
        left_pad_s  = settings.SPEECH_PADDING_MS / 1000

        # 2. Convert wall-clock timestamps to "seconds ago" offsets for get_range.
        #    end_ago is clamped to 0 so we never request audio from the future.
        # get_range(start_ago, end_ago): start_ago > end_ago (start is further back)
        end_ago   = max(0.0, (now - vad.last_speech_time).total_seconds() - right_pad_s)
        start_ago = (now - vad.speech_start_time).total_seconds() + left_pad_s

        # 3. Guard against zero-length or inverted range (e.g. utterance too short).
        if start_ago <= end_ago:
            return np.array([], dtype=np.int16)

        # 4. Slice the ring buffer to extract the exact utterance window.
        return session.audio_buffer.get_range(start_ago, end_ago)

    async def _finalize_transcript(self, session: StreamingSession) -> None:
        """
        Finalize the current partial transcript and send it as the final result.

        When FINALIZE_RIGHT_PADDING_ENABLED is True, runs one dedicated ASR
        pass over the complete utterance (speech_start → last_speech_time +
        FINALIZE_RIGHT_PADDING_MS) before committing.  Falls back silently to
        the existing partial if extraction fails or ASR returns empty.

        Safe to call even when partial_transcript is empty — becomes a no-op.
        """
        # 1. No-op if there is nothing to finalize.
        if not session.transcript_state.partial_transcript:
            return

        # 2. Optional right-padding pass: re-run ASR over the full utterance boundary
        #    to capture words that the rolling window may have cut off at the tail.
        if settings.FINALIZE_RIGHT_PADDING_ENABLED:
            audio = self._extract_final_window(session)
            if len(audio) > 0:
                transcript = await self.transcription_service.atranscribe(audio)
                if transcript:
                    logger.debug(
                        "[%s] Right-finalize ASR result: '%s'",
                        session.session_id, transcript,
                    )
                    # Overwrite partial with the more complete right-padded result.
                    session.transcript_state.update_partial(transcript)

        # 3. Promote partial → final and send to client.
        session.transcript_state.finalize()
        final_text = session.transcript_state.final_transcript.strip()
        logger.info("[%s] Final transcript: '%s'", session.session_id, final_text)
        await self.connection_manager.send_transcript(
            session.session_id, final_text, is_final=True
        )

    async def _handle_intra_commit(self, session: StreamingSession) -> None:
        """
        Commit the current partial transcript when a mid-utterance pause is
        detected, without ending the utterance.

        Fires once per pause event (guarded by vad_state.intra_committed) when:
          - still inside an utterance (is_speaking=True)
          - silence has exceeded INTRA_SILENCE_MS but not SILENCE_THRESHOLD_MS
          - there is a partial transcript to commit
        """
        vad = session.vad_state

        # All four conditions must hold:
        #   - still inside an utterance (VAD hasn't declared end-of-speech)
        #   - silence has lasted long enough to treat as a clause boundary
        #   - not already committed this pause (intra_committed resets on next speech)
        #   - there is actually something to commit
        if (
            vad.is_speaking
            and vad.silence_duration_ms >= settings.INTRA_SILENCE_MS
            and not vad.intra_committed
            and session.transcript_state.partial_transcript
        ):
            # Mark as committed so this pause only fires once even if silence continues.
            vad.intra_committed = True
            session.transcript_state.finalize()
            committed_text = session.transcript_state.final_transcript.strip()
            logger.info(
                "[%s] Intra-utterance commit at %.0f ms pause: '%s'",
                session.session_id, vad.silence_duration_ms, committed_text,
            )
            await self.connection_manager.send_transcript(
                session.session_id, committed_text, is_final=True
            )

    async def _stop_inference_worker(self, session: StreamingSession) -> None:
        """Cancel and await the inference worker task for a clean shutdown."""
        task = session.inference_task
        if task and not task.done():
            task.cancel()
            try:
                # Await so cancellation completes before the caller proceeds with teardown.
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
                # 1. Block until handle_audio_packet enqueues a window.
                audio_window = await session.audio_queue.get()
                try:
                    # 2. Acquire semaphore to cap concurrent ASR calls across all sessions.
                    async with self.inference_semaphore:
                        await self._run_inference(session, audio_window)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # 3. Swallow per-window errors so the worker stays alive for the next window.
                    logger.error(f"[{session_id}] Inference error: {e}")
        except asyncio.CancelledError:
            logger.debug(f"[{session_id}] Inference worker cancelled")

    async def _run_vad(self, session: StreamingSession, audio_window: np.ndarray) -> tuple:
        """
        Checkout a VAD instance from the pool, run is_speech() on a dedicated
        thread, then return the instance to the pool.

        Using run_in_executor releases the event loop during ONNX inference so
        all 200 receive loops remain responsive.  VAD_POOL_SIZE instances run
        truly in parallel — no shared threading.Lock contention between them.
        """
        # 1. Checkout a VAD instance; timeout prevents indefinite blocking when all are busy.
        try:
            vad = await asyncio.wait_for(self.vad_pool.get(), timeout=5.0)
        except asyncio.TimeoutError:
            # All VAD instances occupied — drop this window and signal backpressure.
            session.dropped_windows += 1
            now = datetime.now()
            if session.should_signal_backpressure(now):
                session.last_backpressure_signal = now
                await self.connection_manager.send_backpressure(
                    session.session_id, "vad_pool_exhausted", session.dropped_windows
                )
            logger.error(
                f"[{session.session_id}] VAD pool exhausted — dropping inference window "
                f"(total drops: {session.dropped_windows})"
            )
            return False, []

        try:
            # 2. Run ONNX inference on a thread so the event loop stays unblocked.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.vad_executor,
                vad.is_speech,
                audio_window,
                settings.VAD_TRIGGER_STRATEGY,
            )
        finally:
            # 3. Always return the instance to the pool, even if inference raised.
            self.vad_pool.put_nowait(vad)

    def _trim_to_speech(self,
                        audio_window: np.ndarray,
                        probs: list) -> np.ndarray:
        """Return the sub-array of audio_window that spans detected speech."""
        # 1. Convert per-frame VAD probabilities to (start_ms, end_ms) speech segments.
        segments = self._vad_ref.segments_from_probs(probs)
        if not segments:
            logger.debug("No speech segments detected — using full audio window")
            return audio_window

        # 2. Span from the first segment start to the last segment end, with padding
        #    on both sides to avoid clipping the first/last phoneme.
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

        # 1. Run VAD on the audio window to detect speech activity.
        is_speech, probs = await self._run_vad(session, audio_window)
        current_time = datetime.now()
        was_speaking = session.vad_state.is_speaking

        # 2. Advance VAD state machine; tracks speech_start_time / last_speech_time.
        session.vad_state.update(is_speech, current_time)

        if is_speech and not was_speaking:
            logger.info(f"[{session.session_id}] Speech started")
        elif not is_speech and was_speaking:
            logger.info(f"[{session.session_id}] Speech ended")

        logger.debug(f"[{session.session_id}] VAD: is_speech={is_speech} is_speaking={session.vad_state.is_speaking}")

        # 3. Commit mid-utterance segment when a long-enough pause is detected.
        if settings.INTRA_SILENCE_COMMIT_ENABLED:
            await self._handle_intra_commit(session)

        # 4. Run ASR while speech is active or the utterance hasn't fully ended yet
        #    (is_speaking stays True during short silences within an utterance).
        if is_speech or session.vad_state.is_speaking:
            # 5. Trim window to speech boundaries to reduce noise fed to ASR.
            audio_to_transcribe = self._trim_to_speech(audio_window, probs)
            logger.debug(
                f"[{session.session_id}] Sending {len(audio_to_transcribe)} samples to ASR "
                f"(trimmed from {len(audio_window)})"
            )

            transcript = await self.transcription_service.atranscribe(audio_to_transcribe)

            if transcript:
                # 6. Stabilize to suppress ASR rollbacks before emitting partial.
                stabilized = self.stabilization_service.stabilize(
                    session.transcript_state.stabilizer,
                    transcript,
                )

                # 7. Only push update when text actually changed to avoid redundant WS messages.
                if stabilized.strip() != session.transcript_state.partial_transcript.strip():
                    session.transcript_state.update_partial(stabilized)
                    logger.info(f"[{session.session_id}] Partial transcript: '{stabilized}'")
                    await self.connection_manager.send_transcript(
                        session.session_id, stabilized, is_final=False
                    )
            else:
                logger.debug(f"[{session.session_id}] ASR returned empty transcript")

        # 8. Finalize utterance once VAD confirms end of speech.
        if not session.vad_state.is_speaking:
            await self._finalize_transcript(session)
