import base64, json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict
from app import startup
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("Log")
websocket_router = APIRouter()

@websocket_router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio and receiving transcripts.

    Expected message format:
    - Audio: {"type": "audio", "data": "<base64_audio>", "sample_rate": 16000}
    - Control: {"type": "control", "action": "start|stop|pause|resume"}

    Response format:
    - Transcript: {"type": "transcript", "text": "...", "is_final": false}
    - Error: {"type": "error", "message": "..."}
    - SessionInfo: {"type": "session_info", "session_id": "...", "status": "..."}
    """
    # Reject early if the server is at capacity — before allocating any state.
    # accept() is required by the WebSocket protocol before sending a close frame.
    if startup.connection_manager.get_connection_count() >= settings.WS_MAX_CONNECTIONS:
        await websocket.accept()
        await websocket.close(code=1013, reason="server_full")
        return

    # Step 1: Create a new session and register the WebSocket connection
    session = startup.session_service.create_session()
    session_id = session.session_id

    await startup.connection_manager.connect(websocket, session_id)
    await startup.connection_manager.send_session_info(session_id, "connected")
    startup.streaming_handler.start_inference_worker(session)

    try:
        # Step 2: Main message loop — receive and dispatch client messages
        while True:
            data = await websocket.receive_text()
            session.update_activity()

            try:
                # Step 3: Parse JSON and route by message type
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "audio":
                    # Step 4a: Forward raw audio chunks for transcription
                    await _handle_audio_message(session_id, message)

                elif message_type == "control":
                    # Step 4b: Handle session lifecycle commands (start/stop/pause/resume)
                    action = message.get("action")
                    await startup.streaming_handler.handle_control_message(session_id, action)

                else:
                    await startup.connection_manager.send_error(
                        session_id,
                        f"Unknown message type: {message_type}",
                        "UNKNOWN_MESSAGE_TYPE",
                    )

            except json.JSONDecodeError:
                await startup.connection_manager.send_error(
                    session_id,
                    "Invalid JSON format",
                    "JSON_DECODE_ERROR",
                )

            except Exception as e:
                await startup.connection_manager.send_error(
                    session_id,
                    f"Error processing message: {str(e)}",
                    "PROCESSING_ERROR",
                )

    except WebSocketDisconnect:
        # Step 5a: Client disconnected — clean up streaming state and session
        await startup.streaming_handler.cleanup_session(session_id)

    except Exception as e:
        # Step 5b: Unexpected server-side error — log and tear down the session
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await startup.streaming_handler.cleanup_session(session_id)


async def _handle_audio_message(session_id: str, message: Dict) -> None:
    try:
        # Step 1: Validate that audio payload is present
        audio_base64 = message.get("data")
        if not audio_base64:
            await startup.connection_manager.send_error(
                session_id,
                "No audio data provided",
                "NO_AUDIO_DATA",
            )
            return

        # Step 2: Decode base64 → raw bytes → int16 PCM samples
        audio_bytes = base64.b64decode(audio_base64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        # Step 3: Pass PCM samples to the streaming handler for transcription
        await startup.streaming_handler.handle_audio_packet(session_id, audio_data)

    except Exception as e:
        await startup.connection_manager.send_error(
            session_id,
            f"Error processing audio: {str(e)}",
            "AUDIO_PROCESSING_ERROR",
        )