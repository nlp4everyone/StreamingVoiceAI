import numpy as np

class AudioPreprocessor:
    """Audio preprocessing utilities."""
    
    @staticmethod
    def normalize(audio: np.ndarray,
                  target_level: float = 0.95) -> np.ndarray:
        """Normalize audio to a target peak level.

        Args:
            audio: Input audio samples as a NumPy array.
            target_level: Desired peak amplitude after normalization (default 0.95).

        Returns:
            Normalized audio array scaled so the peak absolute value equals
            ``target_level``. Returns the original array unchanged if it is
            empty or silent (all zeros).
        """
        if len(audio) == 0:
            return audio
        
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio
        
        return audio * (target_level / max_val)
    
    @staticmethod
    def convert_to_float32(audio: np.ndarray) -> np.ndarray:
        """Convert int16 audio to float32 in range [-1, 1].

        Args:
            audio: Input audio samples as a NumPy int16 array.

        Returns:
            Audio samples as a float32 array scaled to the range [-1, 1].
        """
        return audio.astype(np.float32) / 32768.0
    
    @staticmethod
    def convert_to_int16(audio: np.ndarray) -> np.ndarray:
        """Convert float32 audio in range [-1, 1] to int16.

        Args:
            audio: Input audio samples as a float32 array in the range [-1, 1].

        Returns:
            Audio samples as an int16 array scaled to the range [-32767, 32767].
        """
        return (audio * 32767.0).astype(np.int16)
    
    @staticmethod
    def apply_speech_padding(audio: np.ndarray,
                             padding_samples: int) -> np.ndarray:
        """Add zero padding to audio for speech context.

        Args:
            audio: Input audio samples as a NumPy array.
            padding_samples: Number of zero samples to prepend and append.

        Returns:
            Audio array with ``padding_samples`` zeros added to both ends.
        """
        return np.pad(audio, (padding_samples, padding_samples), mode='constant')
    
    @staticmethod
    def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio.

        Args:
            audio: Input audio samples as a NumPy array.

        Returns:
            Audio array with the mean subtracted so the signal is centered at
            zero. Returns the original array unchanged if it is empty.
        """
        if len(audio) == 0:
            return audio
        return audio - np.mean(audio)
