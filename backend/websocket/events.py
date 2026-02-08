"""
WebSocket endpoint for system events (captures, training progress, alerts).

Supports a simple ping/pong keep-alive protocol. System events are pushed
from the server side via the ConnectionManager.broadcast_event method.
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.websocket.manager import connection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/events")
async def events_websocket(websocket: WebSocket) -> None:
    """
    System events WebSocket endpoint.

    Protocol:
        Client sends: {"action": "ping"}
        Server sends: {"type": "pong"}
        Server pushes: {"type": "<event_type>", ...data}
    """
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception:
        logger.exception("Error in events WebSocket")
        connection_manager.disconnect(websocket)
