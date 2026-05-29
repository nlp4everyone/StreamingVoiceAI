from fastapi import WebSocket
from typing import Dict, Optional
from app.schema import TranscriptMessage, ErrorMessage, SessionInfoMessage
from datetime import datetime

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
            print(f"Error sending transcript to {session_id}: {e}")
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
            print(f"Error sending error to {session_id}: {e}")
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
            print(f"Error sending session info to {session_id}: {e}")
            self.disconnect(session_id)
            return False

    async def broadcast(self,
                        message: dict) -> None:
        """Broadcast a message to all active connections."""
        disconnected = []

        # Attempt delivery to every active connection
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to {session_id}: {e}")
                disconnected.append(session_id)

        # Clean up sessions that failed during broadcast
        for session_id in disconnected:
            self.disconnect(session_id)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def is_connected(self,
                     session_id: str) -> bool:
        """Check if a session is connected."""
        return session_id in self.active_connections
