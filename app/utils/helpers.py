import numpy as np

def calculate_audio_duration(samples: int,
                             sample_rate: int) -> float:
    """
    Calculate audio duration in seconds.

    Args:
        samples: Number of audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Duration in seconds
    """
    return samples / sample_rate


def calculate_samples_from_duration(duration_seconds: float,
                                    sample_rate: int) -> int:
    """
    Calculate number of samples from duration.
    
    Args:
        duration_seconds: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Number of samples
    """
    return int(duration_seconds * sample_rate)


def normalize_audio_level(audio: np.ndarray,
                          target_dbfs: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dBFS level.
    
    Args:
        audio: Audio data
        target_dbfs: Target dBFS level
        
    Returns:
        Normalized audio
    """
    if len(audio) == 0:
        return audio
    
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    
    if rms == 0:
        return audio
    
    # Calculate target RMS
    target_rms = 10 ** (target_dbfs / 20)
    
    # Calculate scaling factor
    scale = target_rms / rms
    
    # Apply scaling
    normalized = audio * scale
    
    # Clip to prevent overflow
    normalized = np.clip(normalized, -32768, 32767)
    
    return normalized.astype(np.int16)


def detect_silence(audio: np.ndarray,
                   threshold: float = 0.01,
                   min_duration: float = 0.1) -> list:
    """
    Detect silence segments in audio.
    
    Args:
        audio: Audio data
        threshold: Silence threshold (0.0 to 1.0)
        min_duration: Minimum silence duration in seconds
        
    Returns:
        List of (start_sample, end_sample) tuples for silence segments
    """
    if len(audio) == 0:
        return []
    
    # Convert to float32 if needed
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32768.0
    else:
        audio_float = audio
    
    # Calculate energy
    energy = np.abs(audio_float)
    
    # Find silence regions
    is_silent = energy < threshold
    
    silence_segments = []
    in_silence = False
    start_idx = 0
    
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            start_idx = i
        elif not silent and in_silence:
            in_silence = False
            duration = (i - start_idx) / len(audio_float)
            if duration >= min_duration:
                silence_segments.append((start_idx, i))
    
    # Handle final silence segment
    if in_silence:
        duration = (len(audio_float) - start_idx) / len(audio_float)
        if duration >= min_duration:
            silence_segments.append((start_idx, len(audio_float)))
    
    return silence_segments
