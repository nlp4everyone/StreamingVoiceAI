from contextlib import asynccontextmanager
# FastAPI component
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# Router
from app.routers import websocket_router, health_router, web_router
from app import startup
import os

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

    # Serve static files (CSS, JS)
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Include routers
    app.include_router(websocket_router)
    app.include_router(health_router)
    app.include_router(web_router)

    return app


app = create_app()
