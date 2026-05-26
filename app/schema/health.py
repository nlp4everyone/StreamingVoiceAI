from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for the GET /health endpoint."""

    status: str = Field(description="Service health status (e.g. 'healthy').")
    active_sessions: int = Field(ge=0, description="Number of currently active sessions.")
    active_connections: int = Field(ge=0, description="Number of open WebSocket connections.")