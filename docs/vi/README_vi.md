# 🎙️ StreamSpeak

Framework nhận dạng giọng nói theo thời gian thực (Speech-to-Text), hỗ trợ nhiều người dùng đồng thời, xây dựng trên FastAPI và WebSocket với độ trễ thấp.

<br />

## Tính năng nổi bật

- **Streaming thời gian thực** — endpoint WebSocket, quản lý trạng thái độc lập mỗi session, trả về partial và final transcript liên tục
- **Voice Activity Detection (VAD)** — Silero VAD chạy qua ONNX runtime (không cần PyTorch: ↓91% dung lượng, ↓76% RAM); hỗ trợ nhiều chiến lược phát hiện (`consecutive_frames`, `ema_smoothed`, `state_machine`)
- **Transcript Stabilization** — ổn định kết quả bằng thuật toán LCP kết hợp rollback suppression; hỗ trợ word-level (tiếng Việt) và character-level
- **Adaptive inference pacing** — 400ms khi bắt đầu nói, tự động giãn ra 1200ms khi kết quả ổn định; giảm ~50% lượng ASR call nhờ trailing-silence correction
- **Non-blocking pipeline** — hàng đợi inference riêng mỗi session + ASR semaphore toàn cục; gửi tín hiệu backpressure khi quá tải
- **Đa người dùng** — mỗi session hoàn toàn độc lập; ring buffer pre-allocated (~384 KB/session)
- **Web client tích hợp** — giao diện ghi âm một chạm, hiển thị transcript trực tiếp, phím tắt `Space` để bật/tắt

<br />

## Điều kiện tiên quyết: ASR Backend

StreamSpeak gửi audio tới một NeMo ASR server bên ngoài để nhận dạng — bản thân nó không thực hiện inference. Cần deploy [VoicePlatform](https://github.com/nlp4everyone/VoicePlatform) trước:

```bash
git clone https://github.com/nlp4everyone/VoicePlatform.git
cd VoicePlatform/
git fetch && git checkout ray/nvidia_asr
cp .env.sample .env
# thiết lập HF_TOKEN trong .env (cần để tải model Pyannote VAD)
bash run_service.sh
```

Lệnh này sẽ mở API ASR tương thích OpenAI tại `http://localhost:8005/v1/audio/transcriptions` — chính là địa chỉ mà `NEMO_API_URL` của StreamSpeak sẽ trỏ tới ở bước dưới đây.

<br />

## Cài đặt

```bash
git clone https://github.com/nlp4everyone/StreamSpeak.git
cd StreamSpeak/
cp .env.example .env
```

Cấu hình NeMo ASR server trong `.env`:
```
NEMO_API_URL=http://localhost:8005/v1/audio/transcriptions
NEMO_MODEL=nvidia/parakeet-ctc-0.6b-vi
PORT=8000
```

> Các tham số thuật toán (VAD threshold, inference interval, stabilizer) được lưu trong `config/settings.yaml` và được version-control. Các giá trị phụ thuộc môi trường (URL, port, concurrency limit) đặt trong `.env`.

Chạy bằng Docker Compose:
```bash
make up
```

Mở trình duyệt tại `http://localhost:8000`.

<br />

## Ví dụ nhanh (Python Client)

`scripts/stream_audio.py` gửi một file WAV tới WebSocket endpoint dưới dạng các gói PCM 20ms theo nhịp thời gian thực, và in ra transcript partial/final ngay khi nhận được:

```bash
python scripts/stream_audio.py [path/to/audio.wav]
```

Mặc định dùng `resources/sample_vi.wav` nếu không truyền đường dẫn. File audio sẽ được downmix về mono và resample về 16kHz nếu cần. Ví dụ output:

```
Session: session_5246de00-aacc-45d1-8b77-be810a5d488e
[is_final=False] Xin chào các bạn.
[is_final=False] Xin chào các bạn hôm nay chúng ta.
[is_final=True] Xin chào các bạn hôm nay chúng ta cùng tìm hiểu.
```

<br />

## Tích hợp

- **API**: FastAPI + WebSocket
- **Web client**: Vanilla JS (MediaRecorder + AudioWorklet)
- **Runtime**: Docker Compose
- **VAD**: [Silero VAD](https://github.com/snakers4/silero-vad) qua ONNX runtime (không cần PyTorch)
- **ASR**: [NVIDIA Parakeet CTC 0.6B tiếng Việt](https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi) qua NeMo HTTP API
- **Audio I/O**: soundfile (WAV encoding trên RAM), scipy, onnxruntime

<br />

## Tài liệu

- [Tổng quan kỹ thuật](TECHNICAL_OVERVIEW_vi.md) — sơ đồ kiến trúc, pipeline xử lý, WebSocket protocol, mô tả component
- [Tham khảo cấu hình](../CONFIGURATION.md) — toàn bộ tham số cấu hình kèm giá trị mặc định

<br />

## To-Do / Roadmap

### 🎯 Voice Activity Detection
- [x] Silero VAD với các chiến lược phát hiện pluggable
- [x] Speech trimming — cắt bỏ khoảng lặng trước khi gửi ASR
- [x] Trailing-silence window correction — giảm ~50% ASR call ở cuối câu

### 🤖 ASR Integration
- [x] Async HTTP client cho NVIDIA NeMo
- [x] WAV encoding trực tiếp trên RAM (không tạo file tạm)

### 🔄 Transcript Stabilization
- [x] LCP stabilizer — word-level (tiếng Việt) và character-level
- [x] Các chiến lược rollback suppression pluggable
- [x] Intra-utterance silence commit và right-finalize padding
- [x] Stabilizer áp dụng cho final ASR pass

### 🖥️ Web Client
- [x] Giao diện trình duyệt tích hợp với transcript trực tiếp

### 🔧 Tối ưu hiệu năng
- [x] Pure ONNX runtime cho SileroVAD (↓91% dung lượng / ↓76% RAM)
- [x] Pre-allocated ring buffer (↓14× bộ nhớ); non-blocking inference pipeline
- [x] Adaptive inference interval
- [ ] Tách file cấu hình

### 🛡️ Khả năng chịu lỗi
- [x] Docker `restart: unless-stopped` + healthcheck
- [x] Graceful shutdown khi nhận SIGTERM
- [ ] Lưu transcript đã hoàn thành xuống storage
- [x] Multi-worker + sticky sessions

<br />

## Trích dẫn mô hình

Dự án sử dụng mô hình **NVIDIA Parakeet CTC 0.6B tiếng Việt**:
➡️ https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi
