from typing import List
from app.core.config import settings


class VADTriggerStrategies:
    """
    Collection of pluggable voice activity detection strategies.

    Each strategy receives a list of per-frame speech probabilities produced by
    Silero VAD (one probability per 512-sample / 32ms frame at 16 kHz) and
    returns True when speech is considered active.

    Choosing a strategy:
        - ``detect_by_consecutive_frames`` — strictest; requires N unbroken
          frames above threshold. Fast to compute, zero lag on silence.
        - ``detect_by_ema_smoothed``       — tolerant of brief noise spikes;
          smooths frame scores before comparing to threshold.
        - ``detect_by_state_machine``      — most realistic for conversation;
          models onset *and* offset so short pauses don't break an utterance.
    """

    @staticmethod
    def detect_by_consecutive_frames(probs: List[float],
                                     threshold: float,
                                     min_speech_frames: int = 3) -> bool:
        """
        Declare speech when at least `min_speech_frames` consecutive frames
        exceed `threshold`. Any sub-threshold frame resets the counter.

        Best for: clean audio where speech onset is sharp.

        Args:
            probs: Per-frame speech probabilities from Silero VAD.
            threshold: Minimum probability to count a frame as speech.
            min_speech_frames: Number of consecutive speech frames required.

        Returns:
            True if a qualifying consecutive run is found.
        """
        if not probs:
            return False

        # Scan frames, counting the current unbroken run of speech frames.
        consecutive = 0
        for prob in probs:
            if prob > threshold:
                consecutive += 1
                # Return as soon as the required run length is reached —
                # no need to scan the remaining frames.
                if consecutive >= min_speech_frames:
                    return True
            else:
                # Any gap resets the run; speech must be truly continuous.
                consecutive = 0

        return False

    @staticmethod
    def detect_by_ema_smoothed(probs: List[float],
                               threshold: float = 0.5,
                               alpha: float = 0.3) -> bool:
        """
        Smooth frame probabilities with an exponential moving average (EMA)
        and declare speech when the EMA crosses `threshold`.

        EMA formula: ema = alpha * prob + (1 - alpha) * ema

        Lower alpha → heavier smoothing, slower to react.
        Higher alpha → less smoothing, closer to raw frame scores.

        Best for: noisy environments where individual frame spikes should not
        trigger detection.

        Args:
            probs: Per-frame speech probabilities from Silero VAD.
            threshold: EMA value above which speech is declared.
            alpha: EMA smoothing factor (0 < alpha < 1).

        Returns:
            True if the smoothed probability exceeds `threshold` at any point.
        """
        if not probs:
            return False

        # 1. Seed the EMA with the first frame so the filter doesn't need a
        #    warm-up period to reach the actual signal level.
        ema = probs[0]
        if ema > threshold:
            return True

        # 2. Slide the EMA across remaining frames; declare speech on first crossing.
        for prob in probs[1:]:
            # Blend the new observation into the running average.
            # High alpha → reacts quickly; low alpha → heavy smoothing.
            ema = alpha * prob + (1.0 - alpha) * ema
            if ema > threshold:
                return True

        return False

    @staticmethod
    def detect_by_state_machine(probs: List[float],
                                threshold: float,
                                onset_frames: int = 2,
                                offset_frames: int = 3,
                                onset_threshold: float = settings.VAD_ONSET_THRESHOLD,
                                offset_threshold: float = settings.VAD_OFFSET_THRESHOLD) -> bool:
        """
        FSM with two states (silence / speech) governed by onset and offset
        frame counters.

        State transitions:
            silence → speech : `onset_frames` consecutive frames above onset_threshold
            speech  → silence: `offset_frames` consecutive frames at or below offset_threshold

        When onset_threshold > offset_threshold (hysteresis), a probability band
        [offset_threshold, onset_threshold] is neutral — frames in this band never
        trigger a transition, eliminating chattering near the decision boundary.

        A higher `offset_frames` adds hang-time so brief pauses inside an
        utterance don't prematurely end speech detection.

        Best for: conversational speech with natural inter-word gaps.

        Args:
            probs: Per-frame speech probabilities from Silero VAD.
            threshold: Fallback boundary used when onset_threshold / offset_threshold
                are not provided (preserves backward compatibility).
            onset_frames: Frames above onset_threshold needed to enter speech state.
            offset_frames: Frames at/below offset_threshold needed to leave speech state.
            onset_threshold: Probability floor for entering speech. Defaults to threshold.
            offset_threshold: Probability ceiling for leaving speech. Defaults to threshold.

        Returns:
            True if the FSM entered the speech state at any point.
        """
        if not probs:
            return False

        # 1. Initialise the FSM — starts in silence with all counters zeroed.
        #    was_speaking latches True on first onset so we can return True even
        #    if the FSM ends back in silence (utterance completed within the window).
        state = "silence"
        onset_count = 0   # consecutive above-threshold frames while silent
        offset_count = 0  # consecutive below-threshold frames while speaking
        was_speaking = False

        # 2. Drive the FSM one frame at a time.
        for prob in probs:
            if state == "silence":
                if prob > onset_threshold:
                    onset_count += 1
                    if onset_count >= onset_frames:
                        # Confirmed speech onset — transition to speech state.
                        state = "speech"
                        was_speaking = True
                        offset_count = 0  # reset for the upcoming offset measurement
                else:
                    # Sub-threshold frame breaks the onset run; start over.
                    onset_count = 0

            else:  # state == "speech"
                if prob <= offset_threshold:
                    offset_count += 1
                    if offset_count >= offset_frames:
                        # Sustained silence confirmed — transition back to silence.
                        state = "silence"
                        onset_count = 0
                        offset_count = 0
                else:
                    # Above-threshold frame: still speaking; discard accumulated
                    # offset so a brief pause doesn't end the utterance prematurely.
                    offset_count = 0

        # 3. Return whether speech was detected at any point in the window,
        #    regardless of the final FSM state (utterance may still be ongoing).
        return was_speaking
