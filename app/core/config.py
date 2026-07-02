"""
Config loading — priority (highest → lowest):
  1. Environment variables   (runtime overrides, Docker -e flags)
  2. .env file               (local dev, not version-controlled)
  3. config/settings.yaml    (stable algorithm defaults, version-controlled)
  4. Field defaults below    (fallback so tests run without any files)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from pydantic.fields import FieldInfo

# Resolve yaml path relative to project root so it works from any cwd.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_YAML = _PROJECT_ROOT / "config" / "settings.yaml"


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Load settings from a YAML file as the lowest-priority source."""

    def __init__(self, settings_cls: Type[BaseSettings], yaml_path: Path) -> None:
        super().__init__(settings_cls)
        self._data: Dict[str, Any] = {}
        if yaml_path.is_file():
            with yaml_path.open() as f:
                self._data = yaml.safe_load(f) or {}

    def get_field_value(self, field: FieldInfo, field_name: str) -> Tuple[Any, str, bool]:
        value = self._data.get(field_name)
        return value, field_name, value is not None

    def field_is_complex(self, field: FieldInfo) -> bool:
        return False

    def __call__(self) -> Dict[str, Any]:
        return {k: v for k, v in self._data.items()}


class Settings(BaseSettings):
    # ═══════════════════════════════════════════════════════════════════════════
    # Algorithm settings — tune for quality/latency, stable across environments.
    # Defaults live in config/settings.yaml; override there, not in .env.
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Audio pipeline ───────────────────────────────────────────────────────
    SAMPLE_RATE: int = 16000             # Hz
    AUDIO_PACKET_MS: int = 20            # incoming WebSocket chunk size
    RING_BUFFER_SECONDS: int = 12        # rolling audio buffer length
    INFERENCE_INTERVAL_MS: int = 600     # min gap between ASR calls (used by chunker; runtime uses ONSET/STABLE below)
    ADAPTIVE_INTERVAL_ENABLED: bool = True  # adaptive interval: dynamically switch between ONSET/STABLE intervals
    ONSET_INTERVAL_MS: int = 400         # adaptive interval (onset): pacing right after speech begins — favors fast partials
    STABLE_INTERVAL_MS: int = 1200       # adaptive interval (stable): pacing when transcript stops changing — reduces redundant ASR calls
    RMS_SILENCE_THRESHOLD: int = 300     # int16 RMS below this skips VAD+ASR when not already speaking; frees VAD pool for active sessions
    INFERENCE_WINDOW_SECONDS: int = 6    # audio window sent to ASR
    SILENCE_THRESHOLD_MS: int = 700      # silence duration that ends an utterance
    TRAILING_SILENCE_MS: int = 1000      # trailing silence in the inference window that overrides is_speech=False
    SPEECH_PADDING_MS: int = 200         # extra audio around speech boundaries
    MIN_TRIMMED_AUDIO_MS: int = 500      # trimmed audio shorter than this skips ASR

    # ── VAD ──────────────────────────────────────────────────────────────────
    VAD_THRESHOLD: float = 0.6           # speech probability cutoff
    VAD_SAMPLE_RATE: int = 16000         # must match SAMPLE_RATE
    VAD_WINDOW_SIZE_SAMPLES: int = 512   # samples per frame (512 = 32 ms at 16 kHz)
    VAD_TRIGGER_STRATEGY: str = "ema_smoothed"  # consecutive_frames | ema_smoothed | state_machine
    VAD_ONSET_THRESHOLD: float = 0.65    # state_machine: threshold to enter speech
    VAD_OFFSET_THRESHOLD: float = 0.40   # state_machine: threshold to leave speech

    # ── Transcript stabilizer ────────────────────────────────────────────────
    INTRA_SILENCE_COMMIT_ENABLED: bool = True  # commit partial on mid-utterance pause
    INTRA_SILENCE_MS: int = 300                # pause to trigger intra-commit; < SILENCE_THRESHOLD_MS
    FINALIZE_RIGHT_PADDING_ENABLED: bool = True  # dedicated final ASR pass at utterance end
    FINALIZE_RIGHT_PADDING_MS: int = 200         # right padding for final pass; keep <= SPEECH_PADDING_MS
    STABILIZER_STRATEGY: str = "frozen_prefix"   # frozen_prefix | hard_length | edit_distance | n_consecutive | hard_then_frozen
    STABILIZER_MODE: str = "word_level"          # word_level | character_level
    STABILIZER_FREEZE_THRESHOLD: int = 3         # consecutive agreements before freezing a prefix
    STABILIZER_MAX_EDIT_DISTANCE: int = 2        # max word edits allowed vs last output
    STABILIZER_N_CONSECUTIVE: int = 3            # repetitions required to confirm a rollback

    # ═══════════════════════════════════════════════════════════════════════════
    # Deployment settings — tune per hardware/environment, set in .env.
    # ═══════════════════════════════════════════════════════════════════════════

    # ── NeMo ASR service ─────────────────────────────────────────────────────
    STT_MODEL_PATH: Optional[str] = None          # reserved for local model loading
    STT_DEVICE: str = "cuda"                       # cuda | cpu
    STT_BATCH_SIZE: int = 1
    NEMO_API_URL: str = "http://localhost:8005/v1/audio/transcriptions"
    NEMO_MODEL: str = "nvidia/parakeet-ctc-0.6b-vi"
    ASR_CONNECT_TIMEOUT: float = 2.0   # TCP connect timeout (s)
    ASR_REQUEST_TIMEOUT: float = 10.0  # full request timeout (s)

    # ── Concurrency — scale with available GPU/CPU ───────────────────────────
    ASR_SEMAPHORE_LIMIT: int = 8  # max concurrent NeMo requests; divide by WORKERS when scaling
    VAD_POOL_SIZE: int = 8        # parallel VAD instances; divide by WORKERS when scaling
    INFERENCE_QUEUE_MAXSIZE: int = 3  # per-session audio queue depth; excess dropped

    # ── VAD model ────────────────────────────────────────────────────────────
    VAD_MODEL_PATH: str = "/app/models/silero_vad.onnx"
    VAD_USE_INT8: bool = False    # prefer INT8 quantized model

    # ── WebSocket ────────────────────────────────────────────────────────────
    WS_MAX_CONNECTIONS: int = 200   # max concurrent sessions per worker
    WS_MAX_QUEUE_SIZE: int = 100    # outbound message queue depth
    WS_PING_INTERVAL: int = 20      # seconds between ping frames
    WS_PING_TIMEOUT: int = 20       # seconds to wait for pong

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Uvicorn worker count — each WebSocket connection is process-local, no shared state needed

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_TRANSCRIPT_CONTENT: bool = False  # log actual transcript text in StreamingHandler; keep False in production (PII)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = Path(os.environ.get("SETTINGS_YAML", str(_DEFAULT_YAML)))
        return (
            init_settings,                                  # highest — programmatic overrides
            env_settings,                                   # env vars (Docker, CI)
            dotenv_settings,                                # .env file
            YamlSettingsSource(settings_cls, yaml_path),   # settings.yaml
            # field defaults are the implicit lowest priority
        )


settings = Settings()
