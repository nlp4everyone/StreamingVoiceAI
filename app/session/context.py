"""Request-scoped facade that binds a single active session to the lifetime of one WebSocket handler."""

from typing import Optional
from app.session.state import StreamingSession
from app.session.manager import SessionManager


class SessionContext:
    """Context manager for session operations."""
    
    def __init__(self,
                 session_manager: SessionManager):
        self.session_manager = session_manager
        self.current_session: Optional[StreamingSession] = None
    
    def create_session(self) -> StreamingSession:
        """Create a new session and set it as current."""
        session = self.session_manager.create_session()
        self.current_session = session
        return session
    
    def set_session(self,
                    session_id: str) -> Optional[StreamingSession]:
        """Set current session by ID."""
        session = self.session_manager.get_session(session_id)
        if session:
            self.current_session = session
        return session
    
    def get_current_session(self) -> Optional[StreamingSession]:
        """Get current session."""
        return self.current_session
    
    def close_session(self) -> bool:
        """Close current session."""
        if self.current_session:
            session_id = self.current_session.session_id
            result = self.session_manager.remove_session(session_id)
            self.current_session = None
            return result
        return False
