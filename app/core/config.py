from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    AUDIO_PACKET_MS: int = 20
    RING_BUFFER_SECONDS: int = 12
    INFERENCE_INTERVAL_MS: int = 400
    INFERENCE_WINDOW_SECONDS: int = 6
    SILENCE_THRESHOLD_MS: int = 700
    SPEECH_PADDING_MS: int = 200
    
    # VAD settings
    VAD_THRESHOLD: float = 0.6
    VAD_SAMPLE_RATE: int = 16000
    VAD_WINDOW_SIZE_SAMPLES: int = 512
    VAD_TRIGGER_STRATEGY: str = "ema_smoothed"  # consecutive_frames | ema_smoothed | state_machine
    
    # STT settings
    STT_MODEL_PATH: Optional[str] = None
    STT_DEVICE: str = "cuda"  # or "cpu"
    STT_BATCH_SIZE: int = 1

    # NeMo ASR settings
    NEMO_API_URL: str = "http://172.17.0.1:8005/v1/audio/transcriptions"
    NEMO_MODEL: str = "nvidia/parakeet-ctc-0.6b-vi"
    
    # WebSocket settings
    WS_MAX_QUEUE_SIZE: int = 100
    WS_PING_INTERVAL: int = 20
    WS_PING_TIMEOUT: int = 20
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
