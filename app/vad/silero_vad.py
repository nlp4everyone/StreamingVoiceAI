from typing import List, Literal
from app.core.config import settings
from app.vad.trigger_strategies import VADTriggerStrategies
from app.utils.logger import setup_logger
import onnxruntime as ort
import threading
import os
import numpy as np

logger = setup_logger("SileroVAD")

class SileroVAD:
    """
    Wrapper around the Silero VAD ONNX model for voice activity detection.

    Silero VAD is a lightweight GRU-based model that scores 512-sample frames
    (32 ms at 16 kHz) with a speech probability in [0, 1].  This class handles
    model loading, audio normalisation, frame-level inference, and delegates the
    final speech/silence decision to a pluggable :class:`VADTriggerStrategies`
    method.

    Threading:
        The underlying model is stateful (GRU hidden state).  A ``threading.Lock``
        serialises all inference calls so that concurrent WebSocket sessions cannot
        corrupt each other's state.  Hidden state is reset at the start of every
        inference batch to ensure each audio clip is scored independently.

    Usage::

        vad = SileroVAD()
        is_speech = vad.is_speech(audio_np, strategy="ema_smoothed")
        prob      = vad.get_speech_probability(audio_np)
        segments  = vad.detect_speech_segments(audio_np)
    """

    def __init__(self,
                 threshold: float = settings.VAD_THRESHOLD,
                 sample_rate: int = settings.VAD_SAMPLE_RATE,
                 window_size_samples: int = settings.VAD_WINDOW_SIZE_SAMPLES):
        """
        Args:
            threshold: Speech-probability cutoff used by all detection methods.
                Frames above this value are considered speech.
                Defaults to ``settings.VAD_THRESHOLD`` (env: ``VAD_THRESHOLD``).
            sample_rate: Expected sample rate of all audio passed to this
                instance.  Silero VAD supports 8 kHz and 16 kHz; Parakeet
                requires 16 kHz, so the default matches that.
                Defaults to ``settings.VAD_SAMPLE_RATE``.
            window_size_samples: Number of samples per inference frame.
                Must be 256 (16 ms) or 512 (32 ms) at 16 kHz per Silero docs.
                Defaults to ``settings.VAD_WINDOW_SIZE_SAMPLES``.
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_size_samples = window_size_samples
        self.model: ort.InferenceSession | None = None
        self._state: np.ndarray | None = None
        # OnnxWrapper prepends 64 samples of context from the previous frame
        # before calling the model; without it probabilities are near zero.
        self._context_size = 64 if sample_rate == 16000 else 32
        self._context: np.ndarray | None = None
        self._sr = np.array(sample_rate, dtype=np.int64)  # shape [] (scalar)
        # Silero VAD is stateful (GRU hidden state); serialize concurrent calls
        # so internal state from one clip does not bleed into another.
        self._lock = threading.Lock()
        self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Fallback locations searched when VAD_MODEL_PATH doesn't exist on disk.
    _FALLBACK_PATHS = [
        os.path.expanduser(
            "~/.cache/torch/hub/snakers4_silero-vad_master"
            "/src/silero_vad/data/silero_vad.onnx"
        ),
    ]

    def _resolve_model_path(self) -> str | None:
        """Return the first model path that exists, preferring INT8 over FP32 when VAD_USE_INT8 is set."""
        base = settings.VAD_MODEL_PATH

        # Build candidate list: INT8 variants are prepended when VAD_USE_INT8 is set
        # so the quantized model is preferred without requiring a separate config path.
        if settings.VAD_USE_INT8:
            candidates = (
                [base.replace(".onnx", "_int8.onnx"), base]
                + [p.replace(".onnx", "_int8.onnx") for p in self._FALLBACK_PATHS]
                + self._FALLBACK_PATHS
            )
        else:
            candidates = [base] + self._FALLBACK_PATHS

        # Return the first path that actually exists on disk.
        for path in candidates:
            if os.path.isfile(path):
                if "_int8" in path:
                    logger.info(f"Using INT8 quantized VAD model: {path}")
                return path

        logger.error(
            f"Silero VAD ONNX model not found. Tried: {candidates}"
        )
        return None

    def _load_model(self) -> None:
        """
        Load the Silero VAD ONNX model via ``ort.InferenceSession``.
        Tries ``settings.VAD_MODEL_PATH`` first, then falls back to the
        torch-hub cache so local dev works without extra setup.

        Sets ``self.model`` to ``None`` on failure so callers can degrade
        gracefully instead of raising at inference time.
        """
        # 1. Resolve model path — prefers INT8 quantized variant if VAD_USE_INT8 is set.
        path = self._resolve_model_path()
        if path is None:
            self.model = None
            return
        try:
            # 2. Configure single-threaded execution to avoid lock contention when
            #    multiple VAD instances run concurrently in the thread pool.
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # 3. Load the ONNX model.
            self.model = ort.InferenceSession(
                path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            # 4. Allocate GRU state buffer — shape [2, batch=1, hidden] derived from model metadata.
            state_meta = next(i for i in self.model.get_inputs() if i.name == "state")
            self._state = np.zeros(
                (state_meta.shape[0], 1, state_meta.shape[2]), dtype=np.float32
            )

            # 5. Pre-allocate input buffer [1, context + window] to avoid per-frame allocation.
            self._input_buf = np.zeros(
                (1, self._context_size + self.window_size_samples), dtype=np.float32
            )
            logger.debug("Silero VAD model loaded (onnxruntime, path=%s)", path)
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            self.model = None

    def _reset_states(self) -> None:
        """Reset GRU state and context to zeros for a fresh audio clip."""
        self._state[:] = 0.0
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalise *audio* to float32 in [-1.0, 1.0] as required by Silero VAD.

        Args:
            audio: Input array — int16 (raw PCM) or any float dtype.

        Returns:
            float32 numpy array in the range [-1.0, 1.0].
        """
        if audio.dtype == np.int16:
            # int16 range is [-32768, 32767] — divide by 32768 to map to [-1.0, ~1.0].
            return audio.astype(np.float32) / 32768.0
        if audio.dtype != np.float32:
            # Any other numeric dtype (float64, int32, …) — cast directly.
            return audio.astype(np.float32)
        # Already float32 — return as-is to avoid a redundant copy.
        return audio

    def _compute_frame_probs(self, audio: np.ndarray) -> List[float]:
        """
        Run Silero VAD on *audio* and return per-frame speech probabilities.

        The array is split into non-overlapping windows of
        ``window_size_samples`` samples.  Incomplete trailing windows are
        discarded so every scored frame has the same length.

        .. warning::
            Must be called while holding ``self._lock``.  The model's GRU state
            is reset at entry, making each call independent of previous clips.

        Args:
            audio: 1-D float32 numpy array at ``self.sample_rate`` Hz.

        Returns:
            List of speech probabilities, one per complete frame.
            Empty list if the audio is shorter than one window.
        """
        # Reset GRU hidden state so this clip is scored independently of the
        # previous call (important when the same instance serves multiple sessions).
        self._reset_states()

        probs = []
        for i in range(0, len(audio), self.window_size_samples):
            chunk = audio[i:i + self.window_size_samples]
            # Skip the last partial window — Silero requires a fixed frame size.
            if len(chunk) < self.window_size_samples:
                break
            # Prepend context from previous frame — model expects [1, context+window].
            self._input_buf[:, :self._context_size] = self._context
            self._input_buf[:, self._context_size:] = chunk
            ort_outs = self.model.run(
                None,
                {
                    "input": self._input_buf,
                    "state": self._state,
                    "sr": self._sr,
                },
            )
            # output: [1, 1], stateN: updated GRU state
            prob_arr, self._state = ort_outs[0], ort_outs[1]
            # Slide context forward: keep last context_size samples of full input.
            self._context[:] = self._input_buf[:, -self._context_size:]
            probs.append(float(prob_arr[0, 0]))

        return probs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_speech(self,
                  audio: np.ndarray,
                  strategy: Literal["consecutive_frames", "ema_smoothed", "state_machine"] = "consecutive_frames",
                  **kwargs) -> tuple[bool, List[float]]:
        """
        Determine whether *audio* contains speech using the selected strategy.

        Steps:
            1. Normalise audio to float32.
            2. Compute per-frame probabilities under the model lock.
            3. Delegate to the chosen :class:`VADTriggerStrategies` method.

        Args:
            audio: 1-D numpy array (int16 or float32) at ``self.sample_rate`` Hz.
            strategy: Detection algorithm to apply.

                - ``"consecutive_frames"`` — N unbroken frames above threshold.
                - ``"ema_smoothed"``       — EMA-smoothed probability vs threshold.
                - ``"state_machine"``      — FSM with onset / offset hang-time.

            **kwargs: Extra parameters forwarded to the chosen strategy
                (e.g. ``min_speech_frames=4``, ``alpha=0.2``, ``onset_frames=3``).

        Returns:
            Tuple of (decision, probs): ``decision`` is ``True`` if the strategy
            declares speech; ``probs`` is the list of per-frame probabilities so
            callers can reuse them (e.g. for segment trimming) without a second
            inference pass. Returns ``(False, [])`` if the model failed to load.

        Raises:
            ValueError: If *strategy* is not one of the recognised values.
        """
        if self.model is None:
            return False, []

        # 1. Normalise raw PCM to float32 in [-1, 1].
        audio = self._to_float32(audio)

        # 2. Compute per-frame speech probabilities (lock serialises GRU state).
        with self._lock:
            probs = self._compute_frame_probs(audio)

        # 3. Delegate the binary speech/silence decision to the chosen strategy.
        if strategy == "consecutive_frames":
            decision = VADTriggerStrategies.detect_by_consecutive_frames(probs, self.threshold, **kwargs)
        elif strategy == "ema_smoothed":
            decision = VADTriggerStrategies.detect_by_ema_smoothed(probs, self.threshold, **kwargs)
        elif strategy == "state_machine":
            decision = VADTriggerStrategies.detect_by_state_machine(probs, self.threshold, **kwargs)
        else:
            raise ValueError(f"Unknown VAD strategy: {strategy}")

        return decision, probs

    def get_speech_probability(self, audio: np.ndarray) -> float:
        """
        Return the peak speech probability across all frames in *audio*.

        Useful for monitoring signal strength or setting adaptive thresholds
        without committing to a binary speech/silence decision.

        Args:
            audio: 1-D numpy array (int16 or float32) at ``self.sample_rate`` Hz.

        Returns:
            Maximum frame probability in [0.0, 1.0].
            Returns ``0.0`` if the model failed to load or the audio is too
            short to produce a complete frame.
        """
        if self.model is None:
            return 0.0

        # 1. Normalise raw PCM to float32 in [-1, 1].
        audio = self._to_float32(audio)

        # 2. Compute per-frame speech probabilities (lock serialises GRU state).
        with self._lock:
            probs = self._compute_frame_probs(audio)

        # 3. Return the peak probability across all frames.
        return max(probs) if probs else 0.0

    def segments_from_probs(self,
                            probs: List[float],
                            min_speech_duration_ms: int = 250,
                            min_silence_duration_ms: int = 100) -> list:
        """
        Locate speech segments from pre-computed per-frame probabilities.

        Prefer this over :meth:`detect_speech_segments` when frame probabilities
        are already available (e.g. returned by :meth:`is_speech`) to avoid a
        second ONNX inference pass on the same audio.

        Args:
            probs: Per-frame speech probabilities as returned by ``is_speech``.
            min_speech_duration_ms: Minimum duration (ms) for a segment to be
                included in the output.  Filters out very short noise bursts.
            min_silence_duration_ms: How long silence must last (ms) before the
                current segment is closed.  Higher values merge nearby words.

        Returns:
            List of ``(start_ms, end_ms)`` tuples, one per detected segment.
        """
        speech_segments = []
        current_speech_start = None  # sample index where the current segment opened
        silence_samples = 0

        for frame_idx, prob in enumerate(probs):
            sample_offset = frame_idx * self.window_size_samples

            if prob > self.threshold:
                # Speech frame — open a new segment if not already inside one.
                if current_speech_start is None:
                    current_speech_start = sample_offset
                silence_samples = 0  # reset silence counter on any speech frame
            else:
                if current_speech_start is not None:
                    # Silence frame inside an open segment — accumulate silence duration.
                    silence_samples += self.window_size_samples
                    silence_ms = (silence_samples / self.sample_rate) * 1000

                    if silence_ms >= min_silence_duration_ms:
                        # Silence long enough to close the segment.
                        end_sample = sample_offset
                        speech_ms = ((end_sample - current_speech_start) / self.sample_rate) * 1000

                        # Only emit segments above the minimum speech duration threshold.
                        if speech_ms >= min_speech_duration_ms:
                            start_ms = (current_speech_start / self.sample_rate) * 1000
                            end_ms = (end_sample / self.sample_rate) * 1000
                            speech_segments.append((start_ms, end_ms))

                        current_speech_start = None
                        silence_samples = 0

        # Flush any segment still open at the end of the audio (no trailing silence to close it).
        if current_speech_start is not None:
            total_samples = len(probs) * self.window_size_samples
            speech_ms = ((total_samples - current_speech_start) / self.sample_rate) * 1000
            if speech_ms >= min_speech_duration_ms:
                start_ms = (current_speech_start / self.sample_rate) * 1000
                end_ms = (total_samples / self.sample_rate) * 1000
                speech_segments.append((start_ms, end_ms))

        return speech_segments

    def detect_speech_segments(self,
                               audio: np.ndarray,
                               min_speech_duration_ms: int = 250,
                               min_silence_duration_ms: int = 100) -> list:
        """
        Locate all speech segments in *audio* and return their time boundaries.

        Runs a full ONNX inference pass internally. When frame probabilities are
        already available, use :meth:`segments_from_probs` instead to skip the
        redundant inference.

        Args:
            audio: 1-D numpy array (int16 or float32) at ``self.sample_rate`` Hz.
            min_speech_duration_ms: Minimum duration (ms) for a segment to be
                included in the output.  Filters out very short noise bursts.
            min_silence_duration_ms: How long silence must last (ms) before the
                current segment is closed.  Higher values merge nearby words.

        Returns:
            List of ``(start_ms, end_ms)`` tuples, one per detected segment.
            Returns an empty list if the model failed to load.
        """
        if self.model is None:
            return []

        # 1. Normalise raw PCM to float32 in [-1, 1].
        audio = self._to_float32(audio)

        # 2. Run ONNX inference to get per-frame speech probabilities.
        with self._lock:
            probs = self._compute_frame_probs(audio)

        # 3. Convert probabilities to (start_ms, end_ms) segments.
        return self.segments_from_probs(probs, min_speech_duration_ms, min_silence_duration_ms)
