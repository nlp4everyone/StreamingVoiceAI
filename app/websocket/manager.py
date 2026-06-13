import logging
from fastapi import WebSocket
from typing import Dict, Optional
from app.schema import TranscriptMessage, ErrorMessage, SessionInfoMessage, BackpressureMessage
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for streaming sessions."""

    _instance: Optional["ConnectionManager"] = None

    def __new__(cls) -> "ConnectionManager":
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the connection registry once; no-op on subsequent calls."""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self,
                      websocket: WebSocket,
                      session_id: str) -> None:
        """Accept a WebSocket connection and store it."""
        # Complete the WebSocket handshake
        await websocket.accept()
        # Register the connection under the session ID
        self.active_connections[session_id] = websocket

    def disconnect(self,
                   session_id: str) -> None:
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_transcript(self,
                              session_id: str,
                              text: str,
                              is_final: bool = False,
                              confidence: float = None) -> bool:
        """Send transcript message to a specific session."""
        if session_id not in self.active_connections:
            return False

        websocket = self.active_connections[session_id]
        # Build the transcript message payload
        message = TranscriptMessage(
            type="transcript",
            text=text,
            is_final=is_final,
            confidence=confidence,
            timestamp=datetime.now()
        )

        # Send and disconnect the session on failure
        try:
            await websocket.send_json(message.model_dump(mode='json'))
            return True
        except Exception as e:
            logger.warning(f"Error sending transcript to {session_id}: {e}")
            self.disconnect(session_id)
            return False

    async def send_error(self,
                         session_id: str,
                         message: str,
                         code: str = None) -> bool:
        """Send error message to a specific session."""
        if session_id not in self.active_connections:
            return False

        websocket = self.active_connections[session_id]
        # Build the error message payload
        error_message = ErrorMessage(
            type="error",
            message=message,
            code=code,
            timestamp=datetime.now()
        )

        # Send and disconnect the session on failure
        try:
            await websocket.send_json(error_message.model_dump(mode='json'))
            return True
        except Exception as e:
            logger.warning(f"Error sending error to {session_id}: {e}")
            self.disconnect(session_id)
            return False

    async def send_session_info(self,
                                session_id: str,
                                status: str) -> bool:
        """Send session info message to a specific session."""
        if session_id not in self.active_connections:
            return False

        websocket = self.active_connections[session_id]
        # Build the session info payload
        message = SessionInfoMessage(
            type="session_info",
            session_id=session_id,
            status=status,
            timestamp=datetime.now()
        )

        # Send and disconnect the session on failure
        try:
            await websocket.send_json(message.model_dump(mode='json'))
            return True
        except Exception as e:
            logger.warning(f"Error sending session info to {session_id}: {e}")
            self.disconnect(session_id)
            return False

    async def send_backpressure(self,
                               session_id: str,
                               reason: str,
                               dropped_windows: int) -> bool:
        """Notify the client that the server is dropping inference windows."""
        if session_id not in self.active_connections:
            return False

        websocket = self.active_connections[session_id]
        # Build the backpressure notification payload.
        message = BackpressureMessage(
            reason=reason,
            dropped_windows=dropped_windows,
            timestamp=datetime.now()
        )

        # Send and disconnect the session on failure.
        try:
            await websocket.send_json(message.model_dump(mode='json'))
            return True
        except Exception as e:
            logger.warning(f"Error sending backpressure to {session_id}: {e}")
            self.disconnect(session_id)
            return False

    async def broadcast(self,
                        message: dict) -> None:
        """Broadcast a message to all active connections."""
        disconnected = []

        # 1. Attempt delivery to every active connection; collect failures.
        #    Failures are deferred — modifying active_connections mid-iteration would break the loop.
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting to {session_id}: {e}")
                disconnected.append(session_id)

        # 2. Clean up sessions that failed during broadcast.
        for session_id in disconnected:
            self.disconnect(session_id)
    
    async def close_idle_session(self, session_id: str) -> None:
        """Send a close frame to an idle session; triggers WebSocketDisconnect in the router."""
        websocket = self.active_connections.get(session_id)
        if websocket is None:
            return
        try:
            # code=1000 (normal closure) so the client knows this was intentional, not a crash.
            # The close frame triggers WebSocketDisconnect in the router, which runs cleanup.
            await websocket.close(code=1000, reason="idle_timeout")
            logger.info(f"[{session_id}] Closed idle WebSocket (idle_timeout)")
        except Exception as e:
            # Socket may already be gone if the client disconnected concurrently.
            logger.debug(f"[{session_id}] close_idle_session error (already gone?): {e}")

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def is_connected(self,
                     session_id: str) -> bool:
        """Check if a session is connected."""
        return session_id in self.active_connections
