import numpy as np
import scipy.signal as signal

class AudioResampler:
    """Audio resampling utilities."""
    
    @staticmethod
    def resample(audio: np.ndarray,
                 original_sr: int,
                 target_sr: int) -> np.ndarray:
        """Resample audio from one sample rate to another.

        Args:
            audio: Input audio samples as a NumPy array.
            original_sr: Sample rate of the input audio in Hz.
            target_sr: Desired output sample rate in Hz.

        Returns:
            Resampled audio as an int16 NumPy array. Returns the original
            array unchanged if ``original_sr`` equals ``target_sr``.
        """
        if original_sr == target_sr:
            return audio
        
        number_of_samples = round(len(audio) * float(target_sr) / original_sr)
        resampled = signal.resample(audio, number_of_samples)
        return resampled.astype(np.int16)
    
    @staticmethod
    def validate_sample_rate(sample_rate: int) -> bool:
        """Check whether a sample rate is among the commonly supported rates.

        Args:
            sample_rate: Sample rate in Hz to validate.

        Returns:
            ``True`` if ``sample_rate`` is one of 8000, 16000, 22050, 44100,
            or 48000; ``False`` otherwise.
        """
        common_rates = [8000, 16000, 22050, 44100, 48000]
        return sample_rate in common_rates
