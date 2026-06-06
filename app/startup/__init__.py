import asyncio
import concurrent.futures
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
inference_semaphore: asyncio.Semaphore = None  # type: ignore[assignment]
vad_pool: asyncio.Queue = None  # type: ignore[assignment]
vad_executor: concurrent.futures.ThreadPoolExecutor = None  # type: ignore[assignment]


async def startup() -> None:
    global connection_manager, session_manager, session_service, streaming_handler
    global inference_semaphore, vad_pool, vad_executor

    logger.info("Initializing services...")

    # Load VAD_POOL_SIZE independent instances — each has its own GRU state
    # and lock so they can run truly in parallel on separate threads.
    logger.info(f"Loading {settings.VAD_POOL_SIZE} VAD instances...")
    t0 = time.monotonic()
    vad_instances = [SileroVAD() for _ in range(settings.VAD_POOL_SIZE)]
    logger.info(f"VAD pool ready in {time.monotonic() - t0:.2f}s")

    vad_pool = asyncio.Queue()
    for v in vad_instances:
        vad_pool.put_nowait(v)

    # Dedicated thread pool: one thread per VAD instance for true parallelism.
    vad_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=settings.VAD_POOL_SIZE,
        thread_name_prefix="vad",
    )

    inference_semaphore = asyncio.Semaphore(settings.ASR_SEMAPHORE_LIMIT)
    connection_manager = ConnectionManager()
    session_manager = SessionManager()
    session_service = SessionService(session_manager, connection_manager)
    streaming_handler = StreamingHandler(
        connection_manager,
        session_manager,
        vad_pool=vad_pool,
        vad_executor=vad_executor,
        inference_semaphore=inference_semaphore,
    )

    logger.info("Services initialized.")


async def shutdown() -> None:
    logger.info("Shutting down services...")
    if streaming_handler:
        await streaming_handler.transcription_service.aclose()
    if vad_executor:
        vad_executor.shutdown(wait=False)
    logger.info("Services shut down.")
