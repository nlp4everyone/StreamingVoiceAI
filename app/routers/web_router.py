from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

web_router = APIRouter(tags=["static"])

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")

@web_router.get("/", summary="Web client")
async def web_client():
    """Serve the static HTML page that hosts the WebSocket client UI."""
    # Resolve absolute path to index.html inside the static directory
    html_path = os.path.join(_STATIC_DIR, "index.html")
    # Return the file if it exists, otherwise surface a clear error
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "HTML client not found"}
