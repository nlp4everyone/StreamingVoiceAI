# 🎙️ Introduction:

A production-ready, multi-user streaming Speech-to-Text framework built with FastAPI and WebSocket for low-latency real-time audio transcription.

<br />

# 🧠 Key Features

### 🌊 Real-Time Streaming Transcription (FastAPI + WebSocket)

Low-latency audio transcription over WebSocket with per-session state management.

Clients stream 20ms PCM packets at 16kHz; the server emits partial and final transcript messages as speech is detected.

### 🎯 Voice Activity Detection (Silero VAD)

Intelligent speech detection using Silero VAD running on CPU, with pluggable trigger strategies.

Automatically identifies speech regions in the rolling audio buffer, enabling efficient processing by skipping silence before sending audio to the ASR engine.

Runs via **ONNX runtime** for faster load time and lower CPU overhead. The model is preloaded once at app startup and shared across all sessions.

**Resource comparison — ONNX runtime vs. PyTorch backend:**

| | SileroVAD (ONNX, with PyTorch) | ONNX Runtime SileroVAD (no PyTorch) | Reduction |
|---|---|---|---|
| Disk | 24 GB | 2.21 GB | ↓ 91% |
| RAM | 515 MB | 125 MB | ↓ 76% |
| Load time | 0.68 s | 0.05 s | ↓ 93% |

### 🔄 VAD + ASR Streaming Pipeline

Advanced pipeline architecture that chains VAD and ASR for optimal performance:

- **Ring buffer**: 12-second rolling ring buffer per session — pre-allocated numpy int16 array (~384 KB/session, 14× less memory than a deque)
- **Adaptive inference interval**: pacing dynamically switches between `ONSET_INTERVAL_MS` (400ms, fast partials) on new speech and `STABLE_INTERVAL_MS` (1200ms, reduced load) when the transcript stops changing; falls back to fixed `INFERENCE_INTERVAL_MS` when disabled
- **Sliding inference window**: 6-second window re-evaluated at the current adaptive interval for overlapping context
- **Speech trimming**: VAD crops the inference window to detected speech + padding before ASR; frame probabilities from VAD are reused for trimming to avoid a second ONNX pass
- **Non-blocking ASR**: audio windows are snapshot-enqueued into a per-session `asyncio.Queue`; a background worker drains the queue under a global semaphore (`ASR_SEMAPHORE_LIMIT`) so the WebSocket receive loop never blocks on ASR latency
- **Backpressure**: server sends a `backpressure` message to the client when the VAD pool or inference queue is saturated, rate-limited to once per second per session
- **Trailing-silence window correction**: overrides `is_speech=True` when the last speech segment in the inference window ended ≥ `TRAILING_SILENCE_MS` ago — prevents stale VAD decisions from triggering unnecessary ASR calls at end of utterance, reducing ASR calls by ~50%
- **Silence finalization**: 800ms of silence triggers a final transcript flush; the dedicated final ASR pass is run through the stabilizer to prevent raw ASR regressions overwriting committed text
- **Multi-user**: concurrent sessions are fully isolated with independent buffers and state

### 🧩 Transcript Stabilization (LCP)

Smooths unstable streaming ASR output using longest common prefix (LCP) algorithm.

Anchors the stable prefix across hypothesis updates; only changed text is sent to the client, suppressing no-op updates.

- **Word-level LCP** (default) — optimized for Vietnamese and space-delimited scripts
- **Character-level LCP** — available for Latin-script languages needing finer precision

### 🖥️ Built-in Web Client

Bundled browser UI served at `/` — one-click microphone recording, live transcript display, audio level meter, and keyboard shortcut (`Space` to toggle).

### 🔧 Configuration Management

Pydantic-settings–based configuration with a four-level priority chain (highest → lowest):

1. Environment variables (Docker `-e` flags, CI)
2. `.env` file (local dev, not version-controlled) — environment-specific values: URLs, paths, ports, concurrency limits
3. `config/settings.yaml` — stable algorithm parameters: inference intervals, VAD thresholds, stabilizer settings (version-controlled)
4. Field defaults in `app/core/config.py`

<br />

### 🔗 Installing:

Clone this project:
```
git clone https://github.com/nlp4everyone/StreamingVoiceAI.git
```
Go inside project:
```
cd StreamingVoiceAI/
```

Create .env file from sample:
```
cp .env.example .env
```

Configure the NeMo ASR server URL and model in `.env` (or leave defaults):
```
NEMO_API_URL=http://localhost:8005/v1/audio/transcriptions
NEMO_MODEL=nvidia/parakeet-ctc-0.6b-vi
PORT=8000
```

Stable algorithm parameters (inference intervals, VAD thresholds, stabilizer settings) live in `config/settings.yaml` and are version-controlled. Environment-specific values (URLs, paths, ports, concurrency limits) go in `.env`.

Run service with Docker Compose:
```
bash run_service.sh
```

Open the web client in your browser:
```
http://localhost:8000
```

Usage Example (Python WebSocket client):
```python
import asyncio
import base64
import json
import numpy as np
import websockets

async def stream_audio():
    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as ws:
        info = json.loads(await ws.recv())
        print("Session:", info["session_id"])

        # Send 20ms PCM int16 packets at 16kHz
        pcm = np.zeros(320, dtype=np.int16)   # replace with real mic audio
        payload = json.dumps({
            "type": "audio",
            "data": base64.b64encode(pcm.tobytes()).decode(),
            "sample_rate": 16000
        })
        await ws.send(payload)

        msg = json.loads(await ws.recv())
        print(f"[is_final={msg['is_final']}] {msg['text']}")

asyncio.run(stream_audio())
```

<br />

# 💴 Integrations:

- ⚙️ API Layer: FastAPI with WebSocket streaming endpoints

- 🌐 Web Client: Vanilla JS browser UI with MediaRecorder and AudioWorklet

- 💻 Runtime: Docker Compose (web service, port configurable via `PORT`)

- 🎯 VAD Model: Silero VAD ([snakers4/silero-vad](https://github.com/snakers4/silero-vad)) via ONNX runtime (no PyTorch dependency)

- 🤖 ASR Model: NVIDIA Parakeet CTC ([nvidia/parakeet-ctc-0.6b-vi](https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi)) via NeMo HTTP API

- 🔧 ASR Transport: aiohttp async multipart POST to NeMo inference server

- 📦 Audio I/O: soundfile (in-memory WAV encoding), scipy, onnxruntime

<br />

# 🧪 Testing:

### 🔹 Web Client

Navigate to `http://localhost:8000` and click the microphone button (or press `Space`) to begin streaming.

### 🔹 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve built-in web client |
| `GET` | `/static/*` | Static assets (CSS, JS) |
| `WS` | `/ws/stream` | Streaming audio endpoint |
| `GET` | `/health` | Health check (active sessions + connections) |

### 🔹 WebSocket Protocol

**Endpoint:** `ws://<host>/ws/stream`

**Client → Server:**

| Message | Format |
|---------|--------|
| Audio packet | `{"type": "audio", "data": "<base64 PCM int16>", "sample_rate": 16000}` |
| Control | `{"type": "control", "action": "start\|stop"}` |

**Server → Client:**

| Message | Format |
|---------|--------|
| Session info | `{"type": "session_info", "session_id": "...", "status": "connected"}` |
| Partial transcript | `{"type": "transcript", "text": "...", "is_final": false}` |
| Final transcript | `{"type": "transcript", "text": "...", "is_final": true}` |
| Backpressure | `{"type": "backpressure", "reason": "queue_full\|vad_pool_exhausted", "dropped_windows": N}` |
| Error | `{"type": "error", "message": "...", "code": "..."}` |

**Control actions:**
- `start` — reset session state (clears buffer, VAD, transcript)
- `stop` — flush any pending partial as a final transcript

### 🔹 Configuration Options

Key configuration parameters in `app/core/config.py` (all overridable via `.env`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `AUDIO_PACKET_MS` | 20 | Expected client packet size |
| `RING_BUFFER_SECONDS` | 12 | Max audio retained per session |
| `INFERENCE_INTERVAL_MS` | 600 | Fixed inference interval (ms) — used by chunker and as fallback when `ADAPTIVE_INTERVAL_ENABLED=false` |
| `ADAPTIVE_INTERVAL_ENABLED` | `true` | Dynamically switch pacing between `ONSET_INTERVAL_MS` and `STABLE_INTERVAL_MS` |
| `ONSET_INTERVAL_MS` | 400 | Adaptive interval (onset): pacing right after speech begins — favors fast partials |
| `STABLE_INTERVAL_MS` | 1200 | Adaptive interval (stable): pacing when transcript stops changing — reduces redundant ASR calls |
| `RMS_SILENCE_THRESHOLD` | 300 | int16 RMS energy gate — skips VAD+ASR on silent windows when not already speaking, freeing VAD pool for active sessions |
| `INFERENCE_WINDOW_SECONDS` | 6 | Audio window fed to STT |
| `SILENCE_THRESHOLD_MS` | 800 | Silence duration before finalize |
| `TRAILING_SILENCE_MS` | 1000 | Trailing silence in the inference window that overrides `is_speech=True` — prevents stale VAD detections; reduces ASR calls by ~50% at utterance end |
| `SPEECH_PADDING_MS` | 200 | Context padding around speech region before ASR |
| `VAD_THRESHOLD` | 0.6 | Silero speech probability cutoff (`ema_smoothed` / `consecutive_frames`) |
| `VAD_ONSET_THRESHOLD` | 0.65 | Prob to **enter** speaking state (`state_machine` strategy) |
| `VAD_OFFSET_THRESHOLD` | 0.40 | Prob to **exit** speaking state — hysteresis band prevents chattering |
| `VAD_TRIGGER_STRATEGY` | `state_machine` | Active VAD strategy (`consecutive_frames` \| `ema_smoothed` \| `state_machine`) |
| `VAD_POOL_SIZE` | 8 | Number of parallel VAD instances in the async pool |
| `VAD_MODEL_PATH` | `/app/models/silero_vad.onnx` | Path to the Silero VAD ONNX model file |
| `VAD_USE_INT8` | `false` | Quantize FP32 model to INT8 on first startup (`_int8.onnx` cached on disk) |
| `NEMO_API_URL` | `http://172.17.0.1:8005/v1/audio/transcriptions` | NeMo inference server endpoint |
| `NEMO_MODEL` | `nvidia/parakeet-ctc-0.6b-vi` | ASR model identifier |
| `ASR_SEMAPHORE_LIMIT` | 8 | Max concurrent NeMo HTTP requests across all sessions |
| `INFERENCE_QUEUE_MAXSIZE` | 3 | Per-session queue depth; excess windows are dropped and backpressure sent |
| `ASR_CONNECT_TIMEOUT` | 2.0 | Seconds to establish TCP connection to NeMo server |
| `ASR_REQUEST_TIMEOUT` | 10.0 | Seconds for full request (connect + transfer + response) |
| `WS_MAX_CONNECTIONS` | `200` | Hard cap on concurrent WebSocket sessions; excess connections closed with code 1013 |
| `WS_MAX_QUEUE_SIZE` | `100` | Per-connection send queue depth |
| `WS_PING_INTERVAL` | `20` | Keepalive ping interval (s) |
| `WS_PING_TIMEOUT` | `20` | Ping response timeout (s) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | 8000 | Server port |
| `WORKERS` | 1 | Uvicorn worker count |

# 📋 To-Do / Roadmap

### 🎯 Voice Activity Detection (VAD)
- [x] Implement Silero VAD with pluggable trigger strategies
- [x] Speech trimming to crop inference window before ASR
- [x] Trailing-silence window correction — overrides stale `is_speech=True`; ~50% fewer ASR calls at utterance end

### 🤖 ASR Integration
- [x] Async HTTP client for NVIDIA NeMo ASR inference server
- [x] In-memory WAV encoding (no temp files)

### 🔄 Transcript Stabilization
- [x] LCP stabilizer — word-level (Vietnamese) and character-level modes
- [x] Pluggable rollback suppression strategies (`frozen_prefix`, `hard_length`, `edit_distance`, `n_consecutive`, `hard_then_frozen`) with per-session state isolation
- [x] Intra-utterance silence commit and right-finalize padding for accurate segment boundaries
- [x] Stabilizer applied to final ASR pass — prevents raw ASR regressions overwriting committed text

### 🖥️ Web Client
- [x] Built-in browser UI with microphone recording and live transcripts

### 🔧 Refactor / Optimization
- [x] Pure ONNX runtime for SileroVAD — no PyTorch (↓ 91% disk / ↓ 76% RAM); model baked into Docker image layer
- [x] Pre-allocated np.int16 ring buffer (↓ 14× memory); VAD async pool + per-session inference queue (non-blocking receive loop)
- [x] Backpressure signaling, shared aiohttp ClientSession, structured logging
- [x] Adaptive inference interval — fast partials at onset (400ms), back off when stable (1200ms)
- [ ] Split configuration file

### 🛡️ Fault Tolerance
- [x] Process supervision — Docker `restart: unless-stopped` + `healthcheck` (detects hung processes, not just crashes)
- [x] Graceful shutdown — finalize all active sessions on SIGTERM before process exit
- [ ] Persist finalized transcripts — append-only log or SQLite so completed transcripts survive a crash
- [ ] Multi-worker + sticky sessions — multiple Uvicorn workers behind a load balancer with session-affinity routing to reduce blast radius per crash

# 📚 Model Citation

This project uses the **NVIDIA Parakeet CTC 0.6B Vietnamese model**:

➡️ **HuggingFace Model:** https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi

If you use this model, please consider citing the original authors.
