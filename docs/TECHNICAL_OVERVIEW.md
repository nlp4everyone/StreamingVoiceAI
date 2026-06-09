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
App Startup  (lifespan)
        ├── _maybe_quantize_vad()         VAD_USE_INT8=true → quantize FP32 → INT8 on first run
        ├── SileroVAD × VAD_POOL_SIZE     load ONNX model into pool (asyncio.Queue)
        ├── ThreadPoolExecutor            one thread per VAD instance (true parallelism)
        ├── asyncio.Semaphore             global ASR cap (ASR_SEMAPHORE_LIMIT=8)
        └── asyncio.Task: idle-cleanup    runs every 60 s; closes sessions idle > 300 s

Client (Browser / App)
        │  JSON over WebSocket
        │  {"type": "audio", "data": "<base64 PCM>"}
        ▼
FastAPI WebSocket Gateway  (/ws/stream)
        │  connection_count >= WS_MAX_CONNECTIONS?
        │      YES → accept() + close(1013, "server_full") → return   ← no state allocated
        │      NO  ↓
        ├── ConnectionManager  (per-session WS send helpers; singleton)
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
                │  handle_audio_packet() enqueues audio_snapshot every 400 ms
                │  _inference_worker() drains queue per session
                ▼
        StreamingHandler._inference_worker()  [background asyncio.Task per session]
                │  async with inference_semaphore  (ASR_SEMAPHORE_LIMIT global cap)
                ▼
        StreamingHandler._run_inference()
                │
                ├──▶ VAD pool (asyncio.Queue of VAD_POOL_SIZE SileroVAD instances)
                │       asyncio.wait_for(pool.get(), timeout=5.0)
                │         timeout → drop window + backpressure (rate-limited 1/s)
                │       run_in_executor → vad.is_speech()   (dedicated thread per instance)
                │       returns (decision: bool, probs: list[float])
                │       release instance back to pool
                │         └── VADTriggerStrategies
                │               (consecutive_frames | ema_smoothed | state_machine)
                │
                │  if speech detected (decision OR vad_state.is_speaking)
                ▼
        StreamingHandler._trim_to_speech(audio_window, probs)
                │  SileroVAD.segments_from_probs(probs)   ← reuses VAD probs, no 2nd ONNX pass
                │  crop to [first_start − padding, last_end + padding]
                │  falls back to full window if no segments found
                ▼
        TranscriptionService.atranscribe(trimmed_audio)  [async]
          └── NvidiaNemoASREngine.atranscribe(audio)
                    │  encode as in-memory WAV (soundfile, PCM 16-bit)
                    │  shared aiohttp.ClientSession POST multipart/form-data
                    ▼
            NeMo Inference Server
            nvidia/parakeet-ctc-0.6b-vi
                    │
                    ▼
                raw transcript text
                │
                │  only if stabilized text differs from previous partial
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
⓪ App startup  (lifespan)
    │  _maybe_quantize_vad()    if VAD_USE_INT8=true AND _int8.onnx missing → quantize FP32 model
    │  SileroVAD × 8            load VAD_POOL_SIZE instances into asyncio.Queue (vad_pool)
    │  ThreadPoolExecutor       max_workers=VAD_POOL_SIZE, thread_name_prefix="vad"
    │  asyncio.Semaphore        inference_semaphore (ASR_SEMAPHORE_LIMIT=8)
    │  asyncio.Task             idle-cleanup loop (every 60 s, timeout 300 s)
    ▼
① Client sends 20ms PCM packets  (base64 JSON, 16kHz int16)
    │
    ▼
② WebSocket route  (app/routers/websocket_router.py)
    │
    │  capacity check:
    │      connection_count >= WS_MAX_CONNECTIONS (200)?
    │          YES → accept() + close(code=1013, reason="server_full") → return
    │          NO  → create session + accept connection + send session_info
    │
    │  start_inference_worker(session)   ← spawn background asyncio.Task per session
    │
    │  message loop:
    │      receive_text() → JSON parse → session.update_activity()  ← ALL message types
    │      "audio"   → base64 decode → np.frombuffer(dtype=np.int16) → handle_audio_packet()
    │      "control" → handle_control_message(action)
    │
    ▼
③ StreamingHandler.handle_audio_packet()
    │
    ├─ StreamingService.process_audio_packet()
    │       RingAudioBuffer.append(packet)   ← np.int16 ring buffer, auto-evict oldest
    │
    └─ StreamingService.should_run_inference()
            elapsed >= INFERENCE_INTERVAL_MS (400 ms) since last snapshot?
                NO  → return  (wait for next packet)
               YES  → get_inference_window()  → last INFERENCE_WINDOW_SECONDS (6 s) of audio
                        session.audio_queue.put_nowait(window)
                          QueueFull?  → dropped_windows++
                                        send backpressure (rate-limited 1/s): reason="queue_full"
    │
    ▼
④ StreamingHandler._inference_worker()  [background asyncio.Task per session]
    │  async with inference_semaphore  ← global cap (ASR_SEMAPHORE_LIMIT=8)
    ▼
    StreamingHandler._run_inference(audio_window)
    │
    ├─ _run_vad(session, audio_window)
    │       asyncio.wait_for(vad_pool.get(), timeout=5.0)
    │           TimeoutError?  → dropped_windows++
    │                            send backpressure: reason="vad_pool_exhausted"
    │                            return (False, [])
    │       loop.run_in_executor(vad_executor, vad.is_speech, audio_window, strategy)
    │           ← runs on dedicated VAD thread; event loop stays unblocked
    │           ← GRU hidden state reset per call (clips are independent)
    │       vad_pool.put_nowait(vad)   ← release immediately after inference
    │       returns (decision: bool, probs: list[float])
    │
    │   VADState.update(decision, now)
    │       silence_duration >= SILENCE_THRESHOLD_MS (700 ms)  →  is_speaking = False
    │
    │   if NOT (decision OR vad_state.is_speaking):
    │       skip STT  ──────────────────────────────────────────────────────┐
    │                                                                        │
    ├─ _trim_to_speech(audio_window, probs)                                  │
    │       self._vad_ref.segments_from_probs(probs)                         │
    │           ← pure Python; reuses existing probs — no 2nd ONNX pass     │
    │       start = max(0, first_segment_start_ms / 1000 × sr − padding)    │
    │       end   = min(len, last_segment_end_ms   / 1000 × sr + padding)   │
    │       padding = SPEECH_PADDING_MS (200 ms) = 3200 samples @ 16 kHz   │
    │       no segments found?  → use full audio_window as fallback          │
    │                                                                        │
    ├─ TranscriptionService.atranscribe(trimmed_audio)                       │
    │       NvidiaNemoASREngine.atranscribe()                                │
    │           soundfile → in-memory BytesIO WAV (PCM 16-bit, mono)        │
    │           shared aiohttp.ClientSession POST multipart/form-data        │
    │           → NEMO_API_URL /v1/audio/transcriptions                      │
    │           connect_timeout=ASR_CONNECT_TIMEOUT (2 s)                   │
    │           total_timeout=ASR_REQUEST_TIMEOUT (10 s)                     │
    │           response["text"]                                             │
    │                                                                        │
    ├─ if stabilized != previous_partial:                                    │
    │       StabilizationService.stabilize(new_hypothesis, prev_partial)     │
    │           TranscriptStabilizer.word_level_lcp()                        │
    │           → stable prefix + updated suffix (only changed text sent)   │
    │       TranscriptState.update_partial(stabilized)                      │
    │       ConnectionManager.send_transcript(is_final=False)                │
    │                                               ◄────────────────────────┘
    └─ if NOT vad_state.is_speaking AND partial_transcript exists:
            TranscriptState.finalize()
            ConnectionManager.send_transcript(is_final=True)
    │
    ▼
⑤ Disconnect / idle-timeout / cleanup
    WebSocketDisconnect (client drop or idle_timeout close by server)
    or unhandled server error
    →  StreamingHandler.cleanup_session()
            _stop_inference_worker()    cancel + await task
            flush pending partial as final (if any)
            SessionManager.remove_session()
            ConnectionManager.disconnect()
```

Inference windows overlap to preserve speech context across packet boundaries:

```text
t=0.0s  [0.0s → 6.0s]
t=0.4s  [0.4s → 6.4s]
t=0.8s  [0.8s → 6.8s]
```

---

# WebSocket Protocol

**Endpoint:** `ws://<host>/ws/stream`

## Client → Server

| Message | Format |
|---|---|
| Audio packet | `{"type": "audio", "data": "<base64 PCM int16>", "sample_rate": 16000}` |
| Control | `{"type": "control", "action": "start\|stop"}` |

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
| `VAD_USE_INT8` | `false` | Quantize FP32 → INT8 on first startup (`_int8.onnx` cached on disk) |
| `NEMO_API_URL` | `http://172.17.0.1:8005/v1/audio/transcriptions` | NeMo server endpoint |
| `NEMO_MODEL` | `nvidia/parakeet-ctc-0.6b-vi` | Model identifier |
| `ASR_SEMAPHORE_LIMIT` | 8 | Max concurrent NeMo HTTP requests across all sessions |
| `INFERENCE_QUEUE_MAXSIZE` | 3 | Per-session queue depth; excess windows are dropped |
| `ASR_CONNECT_TIMEOUT` | 2.0 | Seconds to establish TCP connection to NeMo |
| `ASR_REQUEST_TIMEOUT` | 10.0 | Seconds for full NeMo request (connect + transfer + response) |
| `WS_MAX_CONNECTIONS` | `200` | Hard cap on concurrent WebSocket sessions; excess closed with code 1013 |
| `WS_MAX_QUEUE_SIZE` | `100` | Per-connection send queue depth |
| `WS_PING_INTERVAL` | `20` | Keepalive ping interval (s) |
| `WS_PING_TIMEOUT` | `20` | Ping response timeout (s) |
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
│   └── __init__.py            # App-level singletons + VAD pool init + idle-cleanup task
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
