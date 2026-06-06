import asyncio
from app.websocket.manager import ConnectionManager
from app.session.manager import SessionManager
from app.services.session_service import SessionService
from app.websocket.handlers import StreamingHandler
from app.vad.silero_vad import SileroVAD
from app.core.config import settings
from app.utils.logger import setup_logger
import time

logger = setup_logger("Startup")

connection_manager: ConnectionManager = None  # type: ignore[assignment]
session_manager: SessionManager = None  # type: ignore[assignment]
session_service: SessionService = None  # type: ignore[assignment]
streaming_handler: StreamingHandler = None  # type: ignore[assignment]
vad: SileroVAD = None  # type: ignore[assignment]
inference_semaphore: asyncio.Semaphore = None  # type: ignore[assignment]


async def startup() -> None:
    global connection_manager, session_manager, session_service, streaming_handler, vad, inference_semaphore

    logger.info("Initializing services...")

    logger.info("Loading SileroVAD model...")
    t0 = time.monotonic()
    vad = SileroVAD()
    logger.info(f"SileroVAD model loaded in {time.monotonic() - t0:.2f}s")

    inference_semaphore = asyncio.Semaphore(settings.ASR_SEMAPHORE_LIMIT)
    connection_manager = ConnectionManager()
    session_manager = SessionManager()
    session_service = SessionService(session_manager, connection_manager)
    streaming_handler = StreamingHandler(connection_manager, session_manager, vad=vad,
                                         inference_semaphore=inference_semaphore)

    logger.info("Services initialized.")


async def shutdown() -> None:
    logger.info("Shutting down services...")
    if streaming_handler:
        await streaming_handler.transcription_service.aclose()
    logger.info("Services shut down.")