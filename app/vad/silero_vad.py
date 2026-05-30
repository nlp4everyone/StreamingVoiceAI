from typing import List, Literal
from app.core.config import settings
from app.vad.trigger_strategies import VADTriggerStrategies
import  torch, threading
import numpy as np

class SileroVAD:
    """
    Wrapper around the Silero VAD PyTorch model for voice activity detection.

    Silero VAD is a lightweight GRU-based model that scores 512-sample frames
    (32 ms at 16 kHz) with a speech probability in [0, 1].  This class handles
    model loading, audio normalisation, frame-level inference, and delegates the
    final speech/silence decision to a pluggable :class:`VADTriggerStrategies`
    method.

    Threading:
        The underlying model is stateful (GRU hidden state).  A ``threading.Lock``
        serialises all inference calls so that concurrent WebSocket sessions cannot
        corrupt each other's state.  ``model.reset_states()`` is called at the
        start of every inference batch to ensure each audio clip is scored
        independently.

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
        self.model = None
        # Silero VAD is stateful (GRU hidden state); serialize concurrent calls
        # so internal state from one clip does not bleed into another.
        self._lock = threading.Lock()
        self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Download (first run) or load the Silero VAD model from torch.hub cache.

        Sets ``self.model`` to ``None`` on failure so callers can degrade
        gracefully instead of raising at inference time.
        """
        try:
            # 1. Fetch model weights from torch.hub (cached after first download).
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            # 2. Switch to inference mode — disables dropout and gradient tracking.
            self.model.eval()
        except Exception as e:
            self.model = None

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalise *audio* to float32 in [-1.0, 1.0] as required by Silero VAD.

        Args:
            audio: Input array — int16 (raw PCM) or any float dtype.

        Returns:
            float32 numpy array in the range [-1.0, 1.0].
        """
        if audio.dtype == np.int16:
            # Scale int16 range [-32768, 32767] → [-1.0, ~1.0]
            return audio.astype(np.float32) / 32768.0
        if audio.dtype != np.float32:
            return audio.astype(np.float32)
        return audio

    def _compute_frame_probs(self, audio_tensor: torch.Tensor) -> List[float]:
        """
        Run Silero VAD on *audio_tensor* and return per-frame speech probabilities.

        The tensor is split into non-overlapping windows of
        ``window_size_samples`` samples.  Incomplete trailing windows are
        discarded so every scored frame has the same length.

        .. warning::
            Must be called while holding ``self._lock``.  The model's GRU state
            is reset at entry, making each call independent of previous clips.

        Args:
            audio_tensor: 1-D float32 torch.Tensor at ``self.sample_rate`` Hz.

        Returns:
            List of speech probabilities, one per complete frame.
            Empty list if the audio is shorter than one window.
        """
        # Reset GRU hidden state so this clip is scored independently of the
        # previous call (important when the same instance serves multiple sessions).
        self.model.reset_states()

        probs = []
        with torch.no_grad():
            for i in range(0, len(audio_tensor), self.window_size_samples):
                chunk = audio_tensor[i:i + self.window_size_samples]
                # Skip the last partial window — Silero requires a fixed frame size.
                if len(chunk) < self.window_size_samples:
                    break
                probs.append(self.model(chunk, self.sample_rate).item())

        return probs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_speech(self,
                  audio: np.ndarray,
                  strategy: Literal["consecutive_frames", "ema_smoothed", "state_machine"] = "consecutive_frames",
                  **kwargs) -> bool:
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
            ``True`` if the strategy declares speech, ``False`` otherwise.
            Returns ``False`` immediately if the model failed to load.

        Raises:
            ValueError: If *strategy* is not one of the recognised values.
        """
        if self.model is None:
            return False

        # 1. Normalise raw PCM to float32 in [-1, 1].
        audio = self._to_float32(audio)
        # 2. Wrap in a torch tensor for model inference.
        audio_tensor = torch.from_numpy(audio).float()

        # 3. Compute per-frame speech probabilities (lock serialises GRU state).
        with self._lock:
            probs = self._compute_frame_probs(audio_tensor)

        # 4. Delegate the binary speech/silence decision to the chosen strategy.
        if strategy == "consecutive_frames":
            return VADTriggerStrategies.detect_by_consecutive_frames(probs, self.threshold, **kwargs)
        elif strategy == "ema_smoothed":
            return VADTriggerStrategies.detect_by_ema_smoothed(probs, self.threshold, **kwargs)
        elif strategy == "state_machine":
            return VADTriggerStrategies.detect_by_state_machine(probs, self.threshold, **kwargs)
        else:
            raise ValueError(f"Unknown VAD strategy: {strategy}")

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
        # 2. Wrap in a torch tensor for model inference.
        audio_tensor = torch.from_numpy(audio).float()

        # 3. Compute per-frame speech probabilities (lock serialises GRU state).
        with self._lock:
            probs = self._compute_frame_probs(audio_tensor)

        # 4. Return the peak probability across all frames.
        return max(probs) if probs else 0.0

    def detect_speech_segments(self,
                               audio: np.ndarray,
                               min_speech_duration_ms: int = 250,
                               min_silence_duration_ms: int = 100) -> list:
        """
        Locate all speech segments in *audio* and return their time boundaries.

        The algorithm is a simple threshold-based state machine:

        - A segment starts when a frame exceeds ``self.threshold``.
        - A segment ends after ``min_silence_duration_ms`` ms of continuous
          sub-threshold frames.
        - Segments shorter than ``min_speech_duration_ms`` ms are discarded.

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
        # 2. Wrap in a torch tensor for model inference.
        audio_tensor = torch.from_numpy(audio).float()

        # 3. Compute per-frame speech probabilities (lock serialises GRU state).
        with self._lock:
            probs = self._compute_frame_probs(audio_tensor)

        # 4. Walk frames with a threshold state machine to find segment boundaries.
        speech_segments = []
        current_speech_start = None  # sample index where the current segment began
        silence_samples = 0          # accumulated sub-threshold samples since last speech frame

        for frame_idx, prob in enumerate(probs):
            sample_offset = frame_idx * self.window_size_samples

            if prob > self.threshold:
                if current_speech_start is None:
                    # Speech onset detected — mark the start of a new segment.
                    current_speech_start = sample_offset
                # Reset silence accumulator; we are still in speech.
                silence_samples = 0
            else:
                if current_speech_start is not None:
                    # We are in a potential silence gap inside or after speech.
                    silence_samples += self.window_size_samples
                    silence_ms = (silence_samples / self.sample_rate) * 1000

                    if silence_ms >= min_silence_duration_ms:
                        # Silence has lasted long enough — close the segment.
                        end_sample = sample_offset
                        speech_ms = ((end_sample - current_speech_start) / self.sample_rate) * 1000

                        if speech_ms >= min_speech_duration_ms:
                            start_ms = (current_speech_start / self.sample_rate) * 1000
                            end_ms = (end_sample / self.sample_rate) * 1000
                            speech_segments.append((start_ms, end_ms))

                        # Reset for the next potential segment.
                        current_speech_start = None
                        silence_samples = 0

        # Handle a segment that reaches the end of the audio without a closing silence.
        if current_speech_start is not None:
            total_samples = len(probs) * self.window_size_samples
            speech_ms = ((total_samples - current_speech_start) / self.sample_rate) * 1000
            if speech_ms >= min_speech_duration_ms:
                start_ms = (current_speech_start / self.sample_rate) * 1000
                end_ms = (total_samples / self.sample_rate) * 1000
                speech_segments.append((start_ms, end_ms))

        return speech_segments
