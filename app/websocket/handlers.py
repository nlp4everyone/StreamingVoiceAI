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
        # Step 1: Look up the session — drop silently if it no longer exists
        session = self.session_service.get_session(session_id)
        if not session:
            logger.warning(f"Audio packet received for unknown session: {session_id}")
            return

        # Step 2: Feed packet into the rolling audio buffer
        self.streaming_service.process_audio_packet(session, audio_data)

        # Step 3: Check pacing interval — skip enqueue if too soon
        if self.streaming_service.should_run_inference(session):
            audio_window = self.streaming_service.get_inference_window(session)
            if len(audio_window) == 0:
                return

            # Step 4: Snapshot the inference window and enqueue for the background worker
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
                # Step 5: Queue full — drop window and signal backpressure to the client
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
        # Step 1: Look up the session — ignore stale control messages
        session = self.session_manager.get_session(session_id)
        if not session:
            logger.warning(f"Control message '{action}' for unknown session: {session_id}")
            return

        logger.info(f"[{session_id}] Control action: '{action}'")

        # Step 2: Dispatch on action type
        if action == "stop":
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

        # Step 1: Look up the session
        session = self.session_manager.get_session(session_id)

        if session:
            # Step 2: Cancel the worker first so it doesn't race with teardown
            await self._stop_inference_worker(session)

            # Step 3: Attempt final transcript flush — best-effort, must not block cleanup
            try:
                await self._finalize_transcript(session)
            except Exception as e:
                logger.error(f"[{session_id}] Error finalizing transcript during cleanup: {e}")

        # Step 4: Deregister from session manager and connection map regardless of above
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
        # Step 1: Guard — VAD timestamps are required to compute boundaries
        vad = session.vad_state
        if not vad.last_speech_time or not vad.speech_start_time:
            return np.array([], dtype=np.int16)

        # Step 2: Compute time boundaries relative to now, including padding
        now = datetime.now()
        right_pad_s = settings.FINALIZE_RIGHT_PADDING_MS / 1000
        left_pad_s  = settings.SPEECH_PADDING_MS / 1000

        # get_range(start_ago, end_ago): start_ago > end_ago (start is further back)
        end_ago   = max(0.0, (now - vad.last_speech_time).total_seconds() - right_pad_s)
        start_ago = (now - vad.speech_start_time).total_seconds() + left_pad_s

        # Step 3: Guard — degenerate window (speech_start is more recent than last_speech)
        if start_ago <= end_ago:
            return np.array([], dtype=np.int16)

        # Step 4: Read the audio range from the ring buffer
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
        # Step 1: Log turn-level ASR count on first finalize call (reset prevents double-log)
        if session.asr_call_count > 0:
            logger.info(
                "[%s] Turn ended — ASR calls this turn: %d",
                session.session_id, session.asr_call_count,
            )
            session.asr_call_count = 0

        # Step 2: Guard — nothing to finalize
        if not session.transcript_state.partial_transcript:
            return

        # Step 2: Optionally re-run ASR over the full utterance window for accuracy
        if settings.FINALIZE_RIGHT_PADDING_ENABLED:
            audio = self._extract_final_window(session)
            if len(audio) > 0:
                session.asr_call_count += 1
                transcript = await self.transcription_service.atranscribe(audio)
                if transcript:
                    logger.debug(
                        "[%s] Right-finalize ASR result: '%s'",
                        session.session_id, transcript,
                    )
                    session.transcript_state.update_partial(transcript)

        # Step 3: Promote partial to final and send the committed text to the client
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
        # Step 1: Check all intra-commit conditions — guard fires at most once per pause
        vad = session.vad_state
        if (
            vad.is_speaking
            and vad.silence_duration_ms >= settings.INTRA_SILENCE_MS
            and not vad.intra_committed
            and session.transcript_state.partial_transcript
        ):
            # Step 2: Mark committed and promote partial to final
            vad.intra_committed = True
            session.transcript_state.finalize()
            committed_text = session.transcript_state.final_transcript.strip()
            logger.info(
                "[%s] Intra-utterance commit at %.0f ms pause: '%s'",
                session.session_id, vad.silence_duration_ms, committed_text,
            )
            # Step 3: Send committed text to client as a final segment
            await self.connection_manager.send_transcript(
                session.session_id, committed_text, is_final=True
            )

    async def _stop_inference_worker(self, session: StreamingSession) -> None:
        """Cancel and await the inference worker task for a clean shutdown."""
        # Step 1: Cancel the task if it is still running
        task = session.inference_task
        if task and not task.done():
            task.cancel()
            # Step 2: Await cancellation so the coroutine fully unwinds before teardown
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
                # Step 1: Block until the next audio window is enqueued
                audio_window = await session.audio_queue.get()
                try:
                    # Step 2: Acquire global semaphore to cap concurrent ASR calls, then infer
                    async with self.inference_semaphore:
                        await self._run_inference(session, audio_window)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
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
        # Step 1: Checkout a VAD instance from the pool (5 s timeout guards against pool exhaustion)
        try:
            vad = await asyncio.wait_for(self.vad_pool.get(), timeout=5.0)
        except asyncio.TimeoutError:
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
            # Step 2: Run is_speech() on a dedicated thread to keep the event loop free
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.vad_executor,
                vad.is_speech,
                audio_window,
                settings.VAD_TRIGGER_STRATEGY,
            )
        finally:
            # Step 3: Always return the instance to the pool
            self.vad_pool.put_nowait(vad)

    def _trim_to_speech(self,
                        audio_window: np.ndarray,
                        probs: list) -> np.ndarray:
        """Return the sub-array of audio_window that spans detected speech."""
        # Step 1: Derive speech segment boundaries from VAD frame probabilities
        segments = self._vad_ref.segments_from_probs(probs)

        # Step 2: Fall back to the full window if no speech segments were detected
        if not segments:
            logger.debug("No speech segments detected — using full audio window")
            return audio_window

        # Step 3: Apply padding around the first/last segment and clamp to window bounds
        padding = int(settings.SPEECH_PADDING_MS / 1000 * settings.SAMPLE_RATE)
        start_sample = max(0, int(segments[0][0] / 1000 * settings.SAMPLE_RATE) - padding)
        end_sample = min(len(audio_window), int(segments[-1][1] / 1000 * settings.SAMPLE_RATE) + padding)
        return audio_window[start_sample:end_sample]

    async def _run_inference(self,
                             session: StreamingSession,
                             audio_window: np.ndarray) -> None:
        """Run VAD and STT inference on a pre-captured audio window."""
        # Step 1: Guard — skip empty windows
        if len(audio_window) == 0:
            return

        # Step 2: RMS energy gate — skip VAD+ASR when the window is clearly silent and
        # the session is not mid-utterance.  Protects the shared VAD pool from idle sessions
        # so active-speech sessions can check out a slot without queuing.
        if not session.vad_state.is_speaking:
            rms = np.sqrt(np.mean(audio_window.astype(np.float32) ** 2))
            if rms < settings.RMS_SILENCE_THRESHOLD:
                logger.debug(
                    "[%s] RMS gate: energy=%.1f < threshold=%d — skipping VAD+ASR",
                    session.session_id, rms, settings.RMS_SILENCE_THRESHOLD,
                )
                return

        # Step 3: Run VAD to detect speech presence and get per-frame probabilities
        is_speech, probs = await self._run_vad(session, audio_window)
        current_time = datetime.now()
        was_speaking = session.vad_state.is_speaking

        # Trailing-window correction: the inference window may still contain old
        # speech after the user stopped talking, causing VAD to return is_speech=True
        # even during silence. Override when the last speech segment in the window
        # ended >= SILENCE_THRESHOLD_MS ago.
        if is_speech and probs:
            window_ms = len(audio_window) / settings.SAMPLE_RATE * 1000
            segments = self._vad_ref.segments_from_probs(probs)
            if segments and (window_ms - segments[-1][1]) >= settings.TRAILING_SILENCE_MS:
                logger.debug(
                    "[%s] Trailing-window correction: last speech ended %.0f ms ago (>= %d ms) — overriding is_speech=False",
                    session.session_id,
                    window_ms - segments[-1][1],
                    settings.TRAILING_SILENCE_MS,
                )
                is_speech = False

        # Step 4: Update VAD state machine (silence gate applies SILENCE_THRESHOLD_MS grace)
        session.vad_state.update(is_speech, current_time)

        if is_speech and not was_speaking:
            session.asr_call_count = 0
            logger.info(f"[{session.session_id}] Speech started")
        elif not is_speech and was_speaking:
            logger.info(f"[{session.session_id}] Speech ended")

        logger.debug(f"[{session.session_id}] VAD: is_speech={is_speech} is_speaking={session.vad_state.is_speaking}")

        # Step 5: Optionally commit partial transcript on mid-utterance pause
        if settings.INTRA_SILENCE_COMMIT_ENABLED:
            await self._handle_intra_commit(session)

        # Step 6: Run ASR when speech is active or the silence gate hasn't closed yet
        if is_speech or session.vad_state.is_speaking:
            # Delta gate: skip ASR if last_speech_time hasn't advanced since the last call.
            # VAD only updates last_speech_time on speech frames, so equality means the
            # window contains only silence — ASR would return a duplicate result.
            current_speech_ts = session.vad_state.last_speech_time
            if current_speech_ts is not None and current_speech_ts == session.last_asr_speech_time:
                logger.debug(
                    "[%s] Delta gate: no new speech frames since last ASR — skipping",
                    session.session_id,
                )
                # Skip ASR but fall through to Step 8 so silence detection still finalizes.
            else:
                session.last_asr_speech_time = current_speech_ts
                audio_to_transcribe = self._trim_to_speech(audio_window, probs)

                # Minimum trimmed audio gate: skip ASR when the post-trim window is too
                # short to contain intelligible speech.  Falls through to Step 8 so
                # silence detection can still finalize the utterance.
                min_samples = int(settings.MIN_TRIMMED_AUDIO_MS / 1000 * settings.SAMPLE_RATE)
                if len(audio_to_transcribe) < min_samples:
                    logger.debug(
                        "[%s] Trimmed audio gate: %d samples (%.0f ms) < min %d ms — skipping ASR",
                        session.session_id,
                        len(audio_to_transcribe),
                        len(audio_to_transcribe) / settings.SAMPLE_RATE * 1000,
                        settings.MIN_TRIMMED_AUDIO_MS,
                    )
                else:
                    logger.debug(
                        f"[{session.session_id}] Sending {len(audio_to_transcribe)} samples to ASR "
                        f"(trimmed from {len(audio_window)})"
                    )
                    session.asr_call_count += 1
                    transcript = await self.transcription_service.atranscribe(audio_to_transcribe)

                    if transcript:
                        # Step 7: Stabilize hypothesis and send to client only if it changed
                        stabilized = self.stabilization_service.stabilize(
                            session.transcript_state.stabilizer,
                            transcript,
                        )

                        if stabilized.strip() != session.transcript_state.partial_transcript.strip():
                            session.transcript_state.update_partial(stabilized)
                            logger.info(f"[{session.session_id}] Partial transcript: '{stabilized}'")
                            await self.connection_manager.send_transcript(
                                session.session_id, stabilized, is_final=False
                            )
                    else:
                        logger.debug(f"[{session.session_id}] ASR returned empty transcript")

        # Step 8: Finalize transcript when silence threshold closes the utterance
        if not session.vad_state.is_speaking:
            await self._finalize_transcript(session)
