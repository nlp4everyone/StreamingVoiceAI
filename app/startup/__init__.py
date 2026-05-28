from app.websocket.manager import ConnectionManager
from app.session.manager import SessionManager
from app.services.session_service import SessionService
from app.websocket.handlers import StreamingHandler
from app.utils.logger import setup_logger

logger = setup_logger("Startup")

connection_manager: ConnectionManager = None  # type: ignore[assignment]
session_manager: SessionManager = None  # type: ignore[assignment]
session_service: SessionService = None  # type: ignore[assignment]
streaming_handler: StreamingHandler = None  # type: ignore[assignment]


async def startup() -> None:
    global connection_manager, session_manager, session_service, streaming_handler

    logger.info("Initializing services...")

    connection_manager = ConnectionManager()
    session_manager = SessionManager()
    session_service = SessionService(session_manager, connection_manager)
    streaming_handler = StreamingHandler(connection_manager, session_manager)

    logger.info("Services initialized.")


async def shutdown() -> None:
    logger.info("Shutting down services...")
    # Future resource cleanup (DB connections, thread pools, etc.)
    logger.info("Services shut down.")