from contextlib import asynccontextmanager
# FastAPI component
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# Router
from app.routers import websocket_router, health_router
from app import startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup.startup()
    yield
    await startup.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Streaming Speech-to-Text Service ",
        description="Production-ready multi-user streaming Speech-to-Text architecture using Silero VAD",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(websocket_router)
    app.include_router(health_router)

app = create_app()
