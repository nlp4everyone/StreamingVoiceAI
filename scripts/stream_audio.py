"""Stream a WAV file to the StreamSpeak WebSocket endpoint and print live transcripts.

Usage:
    python scripts/stream_audio.py [path/to/audio.wav]

Defaults to resources/sample_en.wav. The file is resampled to mono 16kHz
int16 PCM and sent as 20ms packets, paced in real time, matching what a
live microphone stream would produce.
"""
import asyncio
import base64
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

URI = "ws://localhost:8000/ws/stream"
TARGET_SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_SAMPLES = TARGET_SAMPLE_RATE * CHUNK_MS // 1000  # 320 samples @ 16kHz
DEFAULT_AUDIO_PATH = Path(__file__).resolve().parent.parent / "resources" / "sample_vi.wav"


def load_pcm16(path: Path) -> np.ndarray:
    """Load a WAV file as mono int16 PCM resampled to TARGET_SAMPLE_RATE."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)  # downmix to mono

    if sample_rate != TARGET_SAMPLE_RATE:
        duration = len(mono) / sample_rate
        target_len = int(duration * TARGET_SAMPLE_RATE)
        src_x = np.linspace(0, duration, num=len(mono), endpoint=False)
        dst_x = np.linspace(0, duration, num=target_len, endpoint=False)
        mono = np.interp(dst_x, src_x, mono)

    pcm16 = np.clip(mono, -1.0, 1.0)
    return (pcm16 * 32767).astype(np.int16)


async def send_audio(ws, pcm: np.ndarray) -> None:
    """Send PCM audio as 20ms packets, paced in real time."""
    for start in range(0, len(pcm), CHUNK_SAMPLES):
        chunk = pcm[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        await ws.send(json.dumps({
            "type": "audio",
            "data": base64.b64encode(chunk.tobytes()).decode(),
            "sample_rate": TARGET_SAMPLE_RATE,
        }))
        await asyncio.sleep(CHUNK_MS / 1000)

    # Flush any trailing speech still buffered server-side
    await ws.send(json.dumps({"type": "control", "action": "stop"}))


async def receive_transcripts(ws) -> None:
    """Print messages from the server until the final transcript arrives."""
    async for raw in ws:
        msg = json.loads(raw)
        msg_type = msg.get("type")

        if msg_type == "transcript":
            print(f"[is_final={msg['is_final']}] {msg['text']}")
            if msg["is_final"]:
                return
        elif msg_type == "error":
            print(f"[error] {msg['message']}")
        elif msg_type == "session_info":
            print(f"[session_info] status={msg['status']}")
        elif msg_type == "backpressure":
            print(f"[backpressure] reason={msg['reason']} dropped={msg['dropped_windows']}")


async def stream_audio(audio_path: Path) -> None:
    pcm = load_pcm16(audio_path)

    async with websockets.connect(URI) as ws:
        info = json.loads(await ws.recv())
        print("Session:", info["session_id"])

        receiver = asyncio.create_task(receive_transcripts(ws))
        await send_audio(ws, pcm)
        await receiver


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO_PATH
    asyncio.run(stream_audio(path))
