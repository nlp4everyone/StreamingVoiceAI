"""
Health check router.

Exposes GET /health so load-balancers and monitoring tools can confirm
the service is up and report live session/connection counts.
"""

from fastapi import APIRouter
from app.schema.health import HealthResponse
from app import startup

health_router = APIRouter(tags=["health"])


@health_router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check() -> HealthResponse:
    """
    Return the current health status of the service together with live
    metrics for active sessions and open WebSocket connections.
    """
    return HealthResponse(
        status="healthy",
        active_sessions=startup.session_manager.get_active_session_count(),
        active_connections=startup.connection_manager.get_connection_count(),
    )