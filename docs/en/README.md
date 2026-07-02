# 🎙️ StreamSpeak

A production-ready multi-user streaming Speech-to-Text framework built with FastAPI and WebSocket for low-latency real-time transcription.

<br />

## Key Features

- **Real-time streaming** — WebSocket endpoint, per-session state, partial + final transcripts
- **Voice Activity Detection** — Silero VAD via ONNX runtime (no PyTorch: ↓91% disk, ↓76% RAM); pluggable trigger strategies (`consecutive_frames`, `ema_smoothed`, `state_machine`)
- **Transcript Stabilization** — LCP-based stabilizer with rollback suppression; word-level (Vietnamese) and character-level modes
- **Adaptive inference pacing** — 400ms at onset, backs off to 1200ms when stable; ~50% fewer ASR calls at utterance end via trailing-silence correction
- **Non-blocking pipeline** — per-session inference queue + global ASR semaphore; backpressure signaling on queue or VAD pool saturation
- **Multi-user** — fully isolated sessions; pre-allocated np.int16 ring buffer (~384 KB/session)
- **Built-in web client** — one-click mic recording, live transcript, audio level meter (`Space` to toggle)

<br />

## Prerequisite: ASR Backend

StreamSpeak sends audio to an external NeMo ASR server for transcription — it does not run inference itself. Deploy [VoicePlatform](https://github.com/nlp4everyone/VoicePlatform) first:

```bash
git clone https://github.com/nlp4everyone/VoicePlatform.git
cd VoicePlatform/
git fetch && git checkout ray/nvidia_asr
cp .env.sample .env
# set HF_TOKEN in .env (required to download the Pyannote VAD model)
bash run_service.sh
```

This exposes the OpenAI-compatible ASR API at `http://localhost:8005/v1/audio/transcriptions`, which StreamSpeak's `NEMO_API_URL` points to below.

<br />

## Installation

```bash
git clone https://github.com/nlp4everyone/StreamSpeak.git
cd StreamSpeak/
cp .env.example .env
```

Configure the NeMo ASR server in `.env`:
```
NEMO_API_URL=http://localhost:8005/v1/audio/transcriptions
NEMO_MODEL=nvidia/parakeet-ctc-0.6b-vi
PORT=8000
```

> Algorithm parameters (VAD thresholds, inference intervals, stabilizer settings) live in `config/settings.yaml` and are version-controlled. Environment-specific values (URLs, ports, concurrency limits) go in `.env`.

Run with Docker Compose:
```bash
make up
```

Open `http://localhost:8000` in your browser.

<br />

## Quick Start (Python Client)

`scripts/stream_audio.py` streams a WAV file to the WebSocket endpoint as real-time-paced 20ms PCM packets and prints partial/final transcripts as they arrive:

```bash
python scripts/stream_audio.py [path/to/audio.wav]
```

Defaults to `resources/sample_vi.wav` if no path is given. The file is downmixed to mono and resampled to 16kHz if needed. Example output:

```
Session: session_5246de00-aacc-45d1-8b77-be810a5d488e
[is_final=False] Xin chào các bạn.
[is_final=False] Xin chào các bạn hôm nay chúng ta.
[is_final=True] Xin chào các bạn hôm nay chúng ta cùng tìm hiểu.
```

<br />

## Integrations

- **API**: FastAPI + WebSocket
- **Web client**: Vanilla JS (MediaRecorder + AudioWorklet)
- **Runtime**: Docker Compose
- **VAD**: [Silero VAD](https://github.com/snakers4/silero-vad) via ONNX runtime (no PyTorch)
- **ASR**: [NVIDIA Parakeet CTC 0.6B Vietnamese](https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi) via NeMo HTTP API
- **Audio I/O**: soundfile (in-memory WAV encoding), scipy, onnxruntime

<br />

## Documentation

- [Technical Overview](TECHNICAL_OVERVIEW.md) — architecture diagram, main flow, WebSocket protocol, component deep-dives, repository structure
- [Configuration Reference](../CONFIGURATION.md) — all config parameters with defaults and descriptions

<br />

## To-Do / Roadmap

### 🎯 Voice Activity Detection
- [x] Silero VAD with pluggable trigger strategies
- [x] Speech trimming to crop inference window before ASR
- [x] Trailing-silence window correction — ~50% fewer ASR calls at utterance end

### 🤖 ASR Integration
- [x] Async HTTP client for NVIDIA NeMo
- [x] In-memory WAV encoding (no temp files)

### 🔄 Transcript Stabilization
- [x] LCP stabilizer — word-level (Vietnamese) and character-level modes
- [x] Pluggable rollback suppression strategies
- [x] Intra-utterance silence commit and right-finalize padding
- [x] Stabilizer applied to final ASR pass

### 🖥️ Web Client
- [x] Built-in browser UI with live transcripts

### 🔧 Optimization
- [x] Pure ONNX runtime for SileroVAD (↓91% disk / ↓76% RAM)
- [x] Pre-allocated ring buffer (↓14× memory); non-blocking inference pipeline
- [x] Adaptive inference interval
- [ ] Split configuration file

### 🛡️ Fault Tolerance
- [x] Docker `restart: unless-stopped` + healthcheck
- [x] Graceful shutdown on SIGTERM
- [ ] Persist finalized transcripts
- [x] Multi-worker + sticky sessions

<br />

## Model Citation

This project uses the **NVIDIA Parakeet CTC 0.6B Vietnamese** model:
➡️ https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi
