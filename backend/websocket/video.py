"""
WebSocket endpoint for live video frame streaming.

Clients connect and send subscribe/unsubscribe messages to select
which feed they want to watch. Frames are pushed from the server
side via the ConnectionManager.
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.websocket.manager import connection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/video")
async def video_websocket(websocket: WebSocket) -> None:
    """
    Video streaming WebSocket endpoint.

    Protocol:
        Client sends: {"action": "subscribe", "feed_id": "<id>"}
        Client sends: {"action": "unsubscribe"}
        Server sends: {"type": "frame", "feed_id": "...", "data": "<base64>", ...}
    """
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "subscribe":
                feed_id = data.get("feed_id")
                if feed_id:
                    await connection_manager.subscribe_to_feed(websocket, feed_id)
                    await websocket.send_json({
                        "type": "subscribed",
                        "feed_id": feed_id,
                    })

            elif action == "unsubscribe":
                feed_id = data.get("feed_id")
                await connection_manager.unsubscribe_from_feed(websocket, feed_id)
                await websocket.send_json({"type": "unsubscribed"})

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception:
        logger.exception("Error in video WebSocket")
        connection_manager.disconnect(websocket)
