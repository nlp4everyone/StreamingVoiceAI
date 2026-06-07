# Streaming Vietnamese Speech-to-Text

Production-ready multi-user streaming Speech-to-Text architecture using:

- Silero VAD (CPU) with pluggable detection strategies
- NVIDIA Parakeet Vietnamese STT (via NeMo HTTP inference server)
- FastAPI + WebSocket
- Ring buffer + sliding window chunking
- External GPU inference server (NeMo / Ray)

This project focuses on:
- realtime streaming
- low latency transcription
- stable partial results
- scalable multi-user processing

Authentication and security layers are intentionally minimized for simplicity, but the architecture allows easy integration later.

---

# Architecture

```text
Client (Browser / App)
        │  JSON over WebSocket
        │  {"type": "audio", "data": "<base64 PCM>"}
        ▼
FastAPI WebSocket Gateway  (/ws/stream)
        │
        ├── ConnectionManager  (per-session WS send helpers)
        ├── SessionManager     (session registry)
        └── StreamingHandler   (per-packet orchestration)
                │
                ▼
        StreamingSession  (per-session state)
                ├── RingAudioBuffer  (12s ring buffer, pre-allocated np.int16)
                ├── VADState         (speaking / silence tracking)
                ├── TranscriptState  (partial / final transcript)
                └── inference_queue  (asyncio.Queue, maxsize=INFERENCE_QUEUE_MAXSIZE)
                │
                │  handle_audio_packet() enqueues (audio_snapshot, now)
                │  _inference_worker() drains queue per session
                ▼
        StreamingHandler._inference_worker()  [background task per session]
                │  async with inference_semaphore  (ASR_SEMAPHORE_LIMIT global cap)
                ▼
        StreamingHandler._run_inference()
                │
                ├──▶ VAD pool (asyncio.Queue of VAD_POOL_SIZE SileroVAD instances)
                │       acquire instance → run_in_executor → release
                │       returns (decision, probs)  ← probs reused for trimming
                │         └── VADTriggerStrategies
                │               (consecutive_frames | ema_smoothed | state_machine)
                │
                │  if speech detected
                ▼
        StreamingHandler._trim_to_speech(audio_window)
          └── SileroVAD.detect_speech_segments()
                │  crop to speech region + SPEECH_PADDING_MS context
                ▼
        TranscriptionService.atranscribe(trimmed_audio)  [async]
          └── NvidiaNemoASREngine.atranscribe(audio)
                    │  HTTP POST multipart/form-data (aiohttp)
                    ▼
            NeMo Inference Server
            nvidia/parakeet-ctc-0.6b-vi
                    │
                    ▼
                raw transcript text
                │
                │  only if transcript changed
                ▼
        StabilizationService.stabilize(new_hypothesis, previous_partial)
          └── TranscriptStabilizer  (word-level LCP, default for Vietnamese)
                │
                ▼
        ConnectionManager.send_transcript()
                │
                ▼
Client  ← {"type": "transcript", "text": "...", "is_final": false|true}
```

---

# Main Flow

```text
① Client sends 20ms PCM packets  (base64 JSON, 16kHz int16)
    │
    ▼
② WebSocket route  (app/routers/websocket_router.py)
    │  base64 decode → np.frombuffer(dtype=np.int16)
    │  route by message type: "audio" | "control"
    │
    ▼
③ StreamingHandler.handle_audio_packet()
    │
    ├─ StreamingService.process_audio_packet()
    │       RingAudioBuffer.append(packet)   ← pre-allocated np.int16 ring buffer, auto-evict
    │       session.update_activity()
    │
    └─ StreamingService.should_run_inference()
            elapsed >= 400ms since last inference?
                NO  → return  (wait for next packet)
               YES  → snapshot audio window → enqueue to session.inference_queue
                        queue full?  → drop window, send backpressure (rate-limited 1/s)
    │
    ▼
④ StreamingHandler._inference_worker()  [background asyncio task per session]
    │  async with inference_semaphore  ← global cap (ASR_SEMAPHORE_LIMIT=8)
    ▼
    StreamingHandler._run_inference(audio_window)
    │
    ├─ audio_window = RingAudioBuffer.get_latest(6s)
    │
    ├─ acquire SileroVAD instance from vad_pool (asyncio.Queue, VAD_POOL_SIZE=8)
    │       run_in_executor → _compute_frame_probs()   ← GRU reset per call
    │       VADTriggerStrategies.ema_smoothed(probs, threshold=0.6, alpha=0.3)
    │       returns (decision: bool, probs: list[float])
    │       release instance back to pool
    │
    │   vad_pool empty?  → drop window, send backpressure (rate-limited 1/s)
    │
    │   VADState.update(decision, now)
    │       silence_duration >= 700ms  →  is_speaking = False
    │
    │   if NOT (decision OR vad_state.is_speaking):
    │       skip STT  ──────────────────────────────────────────────┐
    │                                                               │
    ├─ _trim_to_speech(audio_window, probs)                         │
    │       SileroVAD.segments_from_probs(probs)  ← reuse VAD probs │
    │       crop to [first_segment_start - padding,                 │
    │                last_segment_end   + padding]                  │
    │       removes leading/trailing silence before ASR             │
    │       falls back to full window if no segments found          │
    │                                                               │
    ├─ TranscriptionService.atranscribe(trimmed_audio)              │
    │       NvidiaNemoASREngine.atranscribe()                       │
    │           encode as in-memory WAV  (soundfile, PCM 16-bit)    │
    │           shared aiohttp.ClientSession POST                   │
    │           /v1/audio/transcriptions                            │
    │           timeout: ASR_CONNECT_TIMEOUT=2s / ASR_REQUEST_TIMEOUT=10s
    │           response["text"]                                    │
    │                                                               │
    ├─ if transcript changed:                                       │
    │       StabilizationService.stabilize(new, previous_partial)   │
    │           TranscriptStabilizer (word_level LCP)               │
    │           → stable prefix + updated suffix                    │
    │                                                               │
    │       TranscriptState.update_partial(stabilized)             │
    │       ConnectionManager.send_transcript(is_final=False)       │
    │                                               ◄───────────────┘
    └─ if NOT vad_state.is_speaking AND partial exists:
            TranscriptState.finalize()
            ConnectionManager.send_transcript(is_final=True)
    │
    ▼
⑤ Disconnect / cleanup
    WebSocketDisconnect  or  server error
    →  StreamingHandler.cleanup_session()
            flush pending partial as final (if any)
            SessionManager.remove_session()
            ConnectionManager.disconnect()
```

Inference windows overlap to preserve speech context across packets:

```text
t=0.0s  [0.0 → 4.0s]
t=0.4s  [0.4 → 4.4s]
t=0.8s  [0.8 → 4.8s]
```

---

# WebSocket Protocol

**Endpoint:** `ws://<host>/ws/stream`

## Client → Server

| Message | Format |
|---|---|
| Audio packet | `{"type": "audio", "data": "<base64 PCM int16>", "sample_rate": 16000}` |
| Control | `{"type": "control", "action": "start\|stop\|pause\|resume"}` |

## Server → Client

| Message | Format |
|---|---|
| Session info | `{"type": "session_info", "session_id": "...", "status": "connected"}` |
| Partial transcript | `{"type": "transcript", "text": "...", "is_final": false}` |
| Final transcript | `{"type": "transcript", "text": "...", "is_final": true}` |
| Backpressure | `{"type": "backpressure", "reason": "queue_full\|vad_pool_exhausted", "dropped_windows": N}` |
| Error | `{"type": "error", "message": "...", "code": "..."}` |

**Control actions:**
- `start` — reset session state (clears buffer, VAD, transcript)
- `stop` — flush any pending partial as a final transcript

---

# HTTP Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve static web client (`static/index.html`) |
| `GET` | `/static/*` | Static assets (CSS, JS) |
| `WS` | `/ws/stream` | Streaming audio endpoint |
| `GET` | `/api/health` | Health check (active sessions + connections) |

---

# Core Components

## WebSocket Router (`app/routers/websocket_router.py`)

Entry point for all WebSocket connections. Responsibilities:
- Accept connection, create session, send `session_info`
- Receive JSON text frames and dispatch by `type`
- `"audio"` → base64 decode → `np.int16` → `StreamingHandler.handle_audio_packet()`
- `"control"` → `StreamingHandler.handle_control_message()`
- On disconnect or error → `StreamingHandler.cleanup_session()`

## StreamingHandler (`app/websocket/handlers.py`)

Per-packet orchestration — the main processing pipeline:
- `handle_audio_packet()` — appends to buffer, snapshots audio window every 400ms and enqueues to the session's `inference_queue`; sends `backpressure` if queue is full
- `start_inference_worker()` — spawns a background `asyncio.Task` per session that drains `inference_queue` under the global `inference_semaphore`
- `_run_inference()` — acquires a VAD instance from the pool → VAD → trim → STT → stabilize → send; sends `backpressure` if VAD pool is exhausted
- `_trim_to_speech(audio_window, probs)` — crops the 6s inference window to the detected speech region + padding using frame probs already computed by `is_speech()`, avoiding a second ONNX pass
- `handle_control_message()` — `start` resets state; `stop` flushes pending partial as final
- `cleanup_session()` — stops inference worker, flushes pending partial, removes session

## ConnectionManager (`app/websocket/manager.py`)

Per-session WebSocket send helpers — `send_transcript()`, `send_error()`, `send_session_info()`, `connect()`, `disconnect()`.

## Schema (`app/schema/`)

Pydantic models for all message types:
- `websocket.py` — `ErrorMessage`, `ControlMessage`, `SessionInfoMessage`, `WebSocketMessage`
- `audio.py` — audio message schema
- `session.py` — session info schema
- `transcript.py` — transcript message schema
- `health.py` — health check response schema

## Session Management (`app/session/`)

- `state.py` — `StreamingSession`: owns `RingAudioBuffer`, `VADState`, `TranscriptState`; tracks `last_inference_time`, `inference_count`, `last_activity`
- `manager.py` — `SessionManager`: session registry (create / get / remove)
- `context.py` — session context helpers

## Ring Buffer (`app/audio/buffer.py`)

Each session holds up to 12 seconds of PCM samples in a pre-allocated `np.int16` array (a true ring buffer with a write-pointer and sample counter). This uses ~384 KB/session — 14× less than the previous `deque`-of-Python-ints approach (5.4 MB).

- `append(audio)` — push new samples using numpy slice writes; wraps around automatically, evicting oldest
- `get_latest(seconds)` — extract the most-recent N seconds as a contiguous int16 ndarray (O(N) copy, no Python loops)
- `get_range(start_s, end_s)` — extract a specific time slice
- `clear()` — reset write pointer and count without reallocating

## Silero VAD (`app/vad/`)

**`silero_vad.py` — `SileroVAD`**

CPU-based Voice Activity Detection running via **pure ONNX runtime** — no PyTorch dependency.
The model is driven directly via `ort.InferenceSession`; GRU hidden state is reset at the start
of each inference call to keep clips independent.

At app startup (`app/startup/__init__.py`), a **pool of `VAD_POOL_SIZE` (default 8) `SileroVAD`
instances** is created and placed in an `asyncio.Queue`. Each inference call acquires an instance
from the pool, runs ONNX inference via `run_in_executor` (dedicated thread), then releases the
instance back — eliminating the `threading.Lock` bottleneck and allowing up to `VAD_POOL_SIZE`
concurrent VAD inferences.

- `is_speech(audio, strategy=...)` — returns `(decision: bool, probs: list[float])`; callers reuse `probs` for speech trimming to avoid a redundant ONNX pass
- `get_speech_probability(audio)` — peak frame probability across the window
- `segments_from_probs(probs, ...)` — derive `(start_ms, end_ms)` speech segments directly from a pre-computed probability list (used by `_trim_to_speech`)
- `detect_speech_segments(audio)` — runs inference then calls `segments_from_probs`; use only when probs are not already available

**`trigger_strategies.py` — `VADTriggerStrategies`**

| Strategy | Mechanism |
|---|---|
| `consecutive_frames` | N consecutive frames above `VAD_THRESHOLD` (default `min_speech_frames=3`) |
| `ema_smoothed` | EMA of frame probs > `VAD_THRESHOLD` (`alpha=0.3`) — **default** |
| `state_machine` | FSM with dual-threshold hysteresis: `onset_frames=2` above `VAD_ONSET_THRESHOLD` (0.65) to enter speech; `offset_frames=3` below `VAD_OFFSET_THRESHOLD` (0.40) to exit — the neutral band [0.40, 0.65] prevents chattering |

## NvidiaNemoASREngine (`app/asr/nvidia_nemo/engine.py`)

HTTP client for an NVIDIA NeMo ASR inference server:

```python
POST http://localhost:8005/v1/audio/transcriptions
Content-Type: multipart/form-data
file: audio.wav  (PCM 16-bit, 16kHz, encoded in-memory via soundfile)
model: nvidia/parakeet-ctc-0.6b-vi
response_format: verbose_json
```

- `transcribe(audio)` — synchronous (requests)
- `atranscribe(audio)` — async, non-blocking; uses a **shared `aiohttp.ClientSession`** for connection pooling across all inference calls, with configurable timeouts (`ASR_CONNECT_TIMEOUT=2s`, `ASR_REQUEST_TIMEOUT=10s`)
- `aclose()` — closes the shared `ClientSession`; called from `startup.shutdown()` for clean teardown
- `is_ready()` — lightweight GET health probe
- No temp files; audio is encoded to `BytesIO` before upload

The inference server runs separately (NeMo / Ray).

## Transcript Stabilization (`app/stabilization/`)

`TranscriptStabilizer` smooths unstable streaming ASR output using longest common prefix (LCP).

**Two modes:**

| Mode | When to use |
|---|---|
| `word_level` (default) | Vietnamese and other space-delimited scripts |
| `character_level` | Latin-script languages needing finer precision |

```text
hypothesis 1:  xin chào
hypothesis 2:  xin chào m
hypothesis 3:  xin chào mọi
hypothesis 4:  xin chào một      ← correction
hypothesis 5:  xin chào mọi người
```

LCP anchors the stable prefix; the unstable suffix is replaced each cycle.
The transcript is only sent to the client when the stabilized text actually differs from the
previous partial — suppressing no-op updates.

`StabilizationService` (`app/services/stabilization_service.py`) wraps the stabilizer for use in handlers.

---

# Configuration (`app/core/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `AUDIO_PACKET_MS` | 20 | Expected client packet size |
| `RING_BUFFER_SECONDS` | 12 | Max audio retained per session (pre-allocated np.int16 ring buffer) |
| `INFERENCE_INTERVAL_MS` | 400 | How often VAD+STT runs |
| `INFERENCE_WINDOW_SECONDS` | 6 | Audio window fed to STT |
| `SILENCE_THRESHOLD_MS` | 700 | Silence before finalize |
| `SPEECH_PADDING_MS` | 200 | Context padding around speech region before ASR |
| `VAD_THRESHOLD` | 0.6 | Silero speech probability cutoff (`ema_smoothed` / `consecutive_frames`) |
| `VAD_ONSET_THRESHOLD` | 0.65 | Prob to **enter** speaking state (`state_machine` strategy) |
| `VAD_OFFSET_THRESHOLD` | 0.40 | Prob to **exit** speaking state — hysteresis band = [0.40, 0.65] |
| `VAD_SAMPLE_RATE` | 16000 | VAD expected sample rate |
| `VAD_WINDOW_SIZE_SAMPLES` | 512 | Frame size for VAD scoring (32ms at 16kHz) |
| `VAD_TRIGGER_STRATEGY` | `ema_smoothed` | Active VAD strategy |
| `VAD_POOL_SIZE` | 8 | Number of parallel VAD instances in the async pool |
| `VAD_MODEL_PATH` | `/app/models/silero_vad.onnx` | Path to the Silero VAD ONNX model |
| `NEMO_API_URL` | `http://172.17.0.1:8005/v1/audio/transcriptions` | NeMo server endpoint |
| `NEMO_MODEL` | `nvidia/parakeet-ctc-0.6b-vi` | Model identifier |
| `ASR_SEMAPHORE_LIMIT` | 8 | Max concurrent NeMo HTTP requests across all sessions |
| `INFERENCE_QUEUE_MAXSIZE` | 3 | Per-session queue depth; excess windows are dropped |
| `ASR_CONNECT_TIMEOUT` | 2.0 | Seconds to establish TCP connection to NeMo |
| `ASR_REQUEST_TIMEOUT` | 10.0 | Seconds for full NeMo request (connect + transfer + response) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | 8000 | Server port |
| `WORKERS` | 1 | Uvicorn worker count |

All values are overridable via environment variables or `.env`.

---

# Repository Structure

```text
app/
├── routers/
│   ├── websocket_router.py    # WS endpoint + message dispatch
│   └── health_router.py       # Health check endpoint
│
├── websocket/
│   ├── handlers.py            # StreamingHandler — per-packet orchestration
│   └── manager.py             # ConnectionManager
│
├── schema/
│   ├── websocket.py           # ErrorMessage, ControlMessage, SessionInfoMessage
│   ├── audio.py               # Audio message schema
│   ├── session.py             # Session info schema
│   ├── transcript.py          # Transcript message schema
│   └── health.py              # Health check schema
│
├── session/
│   ├── state.py               # StreamingSession, VADState, TranscriptState
│   ├── manager.py             # SessionManager
│   └── context.py
│
├── audio/
│   ├── buffer.py              # RingAudioBuffer (pre-allocated np.int16 ring buffer, 14× less memory)
│   ├── chunker.py             # SlidingWindowChunker
│   ├── preprocessing.py
│   └── resampler.py
│
├── vad/
│   ├── silero_vad.py          # SileroVAD — model load, frame probs, lock
│   └── trigger_strategies.py  # VADTriggerStrategies (3 detection modes)
│
├── asr/
│   ├── nvidia_nemo/
│   │   └── engine.py          # NvidiaNemoASREngine → HTTP API (sync + async)
│   └── pseudo/
│       └── engine.py          # Stub engine for local testing
│
├── services/
│   ├── streaming_service.py       # Buffer append + inference timing
│   ├── transcription_service.py   # NvidiaNemoASREngine wiring
│   ├── stabilization_service.py   # StabilizationService wrapper
│   └── session_service.py         # Session CRUD
│
├── stabilization/
│   ├── stabilizer.py              # TranscriptStabilizer (word/character LCP)
│   └── longest_common_prefix/
│       ├── word_level_lcp.py      # Word-level LCP (default, Vietnamese)
│       └── character_level_lcp.py # Character-level LCP
│
├── startup/
│   └── __init__.py            # App-level singletons (session_service, streaming_handler, …)
│
├── core/
│   └── config.py              # Settings (pydantic-settings)
│
└── utils/
    ├── logger.py
    └── helpers.py

static/
├── index.html   # Web client (markup)
├── index.css    # Web client (styles)
└── index.js     # Web client (WebSocket + recording logic)

docker/
├── Dockerfile.web
└── docker-compose.yml     # web + ray services
```

---
