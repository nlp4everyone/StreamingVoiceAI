import numpy as np
from app.core.config import settings


class RingAudioBuffer:
    """Fixed-capacity ring buffer for int16 PCM samples.

    Backed by a pre-allocated numpy int16 array instead of a deque of Python
    ints, reducing per-session memory ~14x (384 KB vs 5.4 MB for 12 s @ 16 kHz).
    """

    def __init__(self,
                 sample_rate: int = settings.SAMPLE_RATE,
                 buffer_seconds: int = settings.RING_BUFFER_SECONDS):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.max_samples = sample_rate * buffer_seconds
        self._buf = np.zeros(self.max_samples, dtype=np.int16)
        self._write_pos = 0   # index of the next write slot
        self._count = 0       # number of valid samples currently stored

    def append(self, audio_data: np.ndarray) -> None:
        """Append samples, evicting the oldest when the buffer is full."""
        samples = np.asarray(audio_data, dtype=np.int16).ravel()
        n = len(samples)
        if n == 0:
            return
        if n > self.max_samples:
            samples = samples[-self.max_samples:]
            n = self.max_samples

        space_to_end = self.max_samples - self._write_pos
        if n <= space_to_end:
            self._buf[self._write_pos:self._write_pos + n] = samples
        else:
            self._buf[self._write_pos:] = samples[:space_to_end]
            self._buf[:n - space_to_end] = samples[space_to_end:]

        self._write_pos = (self._write_pos + n) % self.max_samples
        self._count = min(self._count + n, self.max_samples)

    def _read_tail(self, n: int) -> np.ndarray:
        """Return the n most-recent samples in chronological order."""
        n = min(n, self._count)
        if n == 0:
            return np.array([], dtype=np.int16)
        end = self._write_pos
        start = (end - n) % self.max_samples
        if start < end:
            return self._buf[start:end].copy()
        return np.concatenate([self._buf[start:], self._buf[:end]])

    def get_latest(self, duration_seconds: float) -> np.ndarray:
        """Get the most-recent audio of the requested duration."""
        return self._read_tail(int(duration_seconds * self.sample_rate))

    def get_range(self, start_seconds: float, end_seconds: float) -> np.ndarray:
        """Get audio from end_seconds-ago to start_seconds-ago (chronological)."""
        start_samples = int(start_seconds * self.sample_rate)
        end_samples = int(end_seconds * self.sample_rate)

        if start_samples >= self._count:
            return np.array([], dtype=np.int16)

        end_samples = min(end_samples, self._count)
        start_samples = max(0, start_samples)
        if end_samples - start_samples <= 0:
            return np.array([], dtype=np.int16)

        tail = self._read_tail(end_samples)
        return tail[:-start_samples] if start_samples > 0 else tail

    def clear(self) -> None:
        """Reset the buffer without reallocating."""
        self._write_pos = 0
        self._count = 0

    def size(self) -> int:
        return self._count

    def size_seconds(self) -> float:
        return self._count / self.sample_rate

    def is_empty(self) -> bool:
        return self._count == 0
