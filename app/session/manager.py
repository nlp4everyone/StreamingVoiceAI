from typing import Dict, Optional
from app.session.state import StreamingSession
import uuid


class SessionManager:
    """Registry that owns all active StreamingSession objects for the process lifetime."""

    _instance: Optional["SessionManager"] = None

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.sessions: Dict[str, StreamingSession] = {}
    
    def create_session(self) -> StreamingSession:
        """Create a new streaming session."""
        session_id = str(uuid.uuid4())
        session = StreamingSession(session_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self,
                    session_id: str) -> Optional[StreamingSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def remove_session(self,
                       session_id: str) -> bool:
        """Remove a session by ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_inactive_sessions(self,
                                  timeout_seconds: int = 300) -> int:
        """Remove inactive sessions and return count of removed sessions."""
        inactive_ids = [
            session_id for session_id, session in self.sessions.items()
            if not session.is_active(timeout_seconds)
        ]
        
        for session_id in inactive_ids:
            del self.sessions[session_id]
        
        return len(inactive_ids)
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self.sessions)
    
    def get_all_session_ids(self) -> list:
        """Get all session IDs."""
        return list(self.sessions.keys())
