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

### 🔄 VAD + ASR Streaming Pipeline

Advanced pipeline architecture that chains VAD and ASR for optimal performance:

- **Ring buffer**: 12-second rolling deque per session, auto-evicting oldest samples
- **Sliding inference window**: 6-second window re-evaluated every 400ms for overlapping context
- **Speech trimming**: VAD crops the inference window to detected speech + padding before ASR
- **Silence finalization**: 700ms of silence triggers a final transcript flush
- **Multi-user**: concurrent sessions are fully isolated with independent buffers and state

### 🧩 Transcript Stabilization (LCP)

Smooths unstable streaming ASR output using longest common prefix (LCP) algorithm.

Anchors the stable prefix across hypothesis updates; only changed text is sent to the client, suppressing no-op updates.

- **Word-level LCP** (default) — optimized for Vietnamese and space-delimited scripts
- **Character-level LCP** — available for Latin-script languages needing finer precision

### 🖥️ Built-in Web Client

Bundled browser UI served at `/` — one-click microphone recording, live transcript display, audio level meter, and keyboard shortcut (`Space` to toggle).

### 🔧 Configuration Management

Pydantic-settings–based configuration for all audio, VAD, ASR, WebSocket, and server parameters.

All values are overridable via environment variables or a `.env` file.

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
cp .env.sample .env
```

Configure the NeMo ASR server URL and model in `.env` (or leave defaults):
```
NEMO_API_URL=http://localhost:8005/v1/audio/transcriptions
NEMO_MODEL=nvidia/parakeet-ctc-0.6b-vi
PORT=8000
```

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

- 🎯 VAD Model: Silero VAD ([snakers4/silero-vad](https://github.com/snakers4/silero-vad)) loaded via `torch.hub`

- 🤖 ASR Model: NVIDIA Parakeet CTC ([nvidia/parakeet-ctc-0.6b-vi](https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi)) via NeMo HTTP API

- 🔧 ASR Transport: aiohttp async multipart POST to NeMo inference server

- 📦 Audio I/O: soundfile (in-memory WAV encoding), scipy, torch, torchaudio

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
| `GET` | `/api/health` | Health check (active sessions + connections) |

### 🔹 WebSocket Protocol

**Endpoint:** `ws://<host>/ws/stream`

**Client → Server:**

| Message | Format |
|---------|--------|
| Audio packet | `{"type": "audio", "data": "<base64 PCM int16>", "sample_rate": 16000}` |
| Control | `{"type": "control", "action": "start\|stop\|pause\|resume"}` |

**Server → Client:**

| Message | Format |
|---------|--------|
| Session info | `{"type": "session_info", "session_id": "...", "status": "connected"}` |
| Partial transcript | `{"type": "transcript", "text": "...", "is_final": false}` |
| Final transcript | `{"type": "transcript", "text": "...", "is_final": true}` |
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
| `INFERENCE_INTERVAL_MS` | 400 | How often VAD+STT runs |
| `INFERENCE_WINDOW_SECONDS` | 6 | Audio window fed to STT |
| `SILENCE_THRESHOLD_MS` | 700 | Silence duration before finalize |
| `SPEECH_PADDING_MS` | 200 | Context padding around speech region before ASR |
| `VAD_THRESHOLD` | 0.6 | Silero speech probability cutoff |
| `VAD_TRIGGER_STRATEGY` | `ema_smoothed` | Active VAD strategy (`consecutive_frames` \| `ema_smoothed` \| `state_machine`) |
| `NEMO_API_URL` | `http://172.17.0.1:8005/v1/audio/transcriptions` | NeMo inference server endpoint |
| `NEMO_MODEL` | `nvidia/parakeet-ctc-0.6b-vi` | ASR model identifier |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | 8000 | Server port |
| `WORKERS` | 1 | Uvicorn worker count |

# 📋 To-Do / Roadmap

### 🎯 Voice Activity Detection (VAD)
- [x] Implement Silero VAD with pluggable trigger strategies
- [x] Speech trimming to crop inference window before ASR

### 🤖 ASR Integration
- [x] Async HTTP client for NVIDIA NeMo ASR inference server
- [x] In-memory WAV encoding (no temp files)

### 🔄 Transcript Stabilization
- [x] Word-level LCP stabilizer for Vietnamese
- [x] Character-level LCP stabilizer

### 🖥️ Web Client
- [x] Built-in browser UI with microphone recording and live transcripts

### 🔧 Refactor / Optimization
- [ ] Optimize SileroVAD with ONNX runtime
- [ ] Split configuration file
- [ ] Add comprehensive logging throughout the codebase

# 📚 Model Citation

This project uses the **NVIDIA Parakeet CTC 0.6B Vietnamese model**:

➡️ **HuggingFace Model:** https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi

If you use this model, please consider citing the original authors.
