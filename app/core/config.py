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
    VAD_MODEL_PATH: str = "/app/models/silero_vad.onnx"
    VAD_USE_INT8: bool = False  # prefer INT8 quantized model; set False to force FP32
    # Hysteresis thresholds for state_machine strategy.
    # onset_threshold > offset_threshold creates a neutral band [offset, onset]
    # where no state transition occurs, eliminating chattering near the boundary.
    VAD_ONSET_THRESHOLD: float = 0.65   # prob must exceed this to START speaking
    VAD_OFFSET_THRESHOLD: float = 0.40  # prob must drop below this to STOP speaking
    
    # STT settings
    STT_MODEL_PATH: Optional[str] = None
    STT_DEVICE: str = "cuda"  # or "cpu"
    STT_BATCH_SIZE: int = 1

    # NeMo ASR settings
    NEMO_API_URL: str = "http://172.17.0.1:8005/v1/audio/transcriptions"
    NEMO_MODEL: str = "nvidia/parakeet-ctc-0.6b-vi"
    ASR_CONNECT_TIMEOUT: float = 2.0    # seconds to establish TCP connection
    ASR_REQUEST_TIMEOUT: float = 10.0   # seconds for full request (connect + transfer + response)
    ASR_SEMAPHORE_LIMIT: int = 8        # max concurrent NeMo HTTP requests across all sessions
    INFERENCE_QUEUE_MAXSIZE: int = 3    # per-session queue depth; excess windows are dropped
    VAD_POOL_SIZE: int = 8              # number of parallel VAD instances; match ASR_SEMAPHORE_LIMIT
    
    # Stabilizer settings
    STABILIZER_STRATEGY: str = "hard_length"  # frozen_prefix | hard_length | hard_then_frozen
    STABILIZER_MODE: str = "word_level"          # word_level | character_level
    STABILIZER_FREEZE_THRESHOLD: int = 3         # consecutive agreements before freezing (frozen_prefix only)

    # WebSocket settings
    WS_MAX_CONNECTIONS: int = 200   # hard cap on concurrent WebSocket sessions
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
