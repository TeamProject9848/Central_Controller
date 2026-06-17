import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FlutterBridge:

    def __init__(self):
        self.clients = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(
        self,
        loop: asyncio.AbstractEventLoop,
    ):
        self.loop = loop

        logger.info(
            "FlutterBridge event loop registered"
        )

    async def register(self, websocket):

        self.clients.add(websocket)

        logger.info(
            f"Flutter client connected "
            f"({len(self.clients)} total)"
        )

    async def unregister(self, websocket):

        self.clients.discard(websocket)

        logger.info(
            f"Flutter client disconnected "
            f"({len(self.clients)} total)"
        )

    async def broadcast(
        self,
        payload: dict,
    ):

        logger.debug(
            f"Broadcasting Flutter payload: {payload}"
        )

        if not self.clients:
            return

        message = json.dumps(payload)

        dead = []

        for client in list(self.clients):

            try:

                await client.send(message)

            except Exception as e:

                logger.error(
                    f"Broadcast failed: {e}"
                )

                dead.append(client)

        for client in dead:

            self.clients.discard(client)

    def _schedule_broadcast(
        self,
        payload: dict,
    ):

        if self.loop is None:

            logger.error(
                f"No event loop registered. "
                f"Dropping payload: {payload}"
            )

            return

        asyncio.run_coroutine_threadsafe(
            self.broadcast(payload),
            self.loop,
        )

    def send_alert(
        self,
        key: str,
        person_id: Optional[str] = None,
    ):

        payload = {
            "type": "alert",
            "key": key,
        }
        if person_id is not None:
            payload["person_id"] = person_id

        logger.info(
            f"Sending Flutter alert: {payload}"
        )

        self._schedule_broadcast(payload)

    def send_person_left(
        self,
        person_id: str,
    ):

        payload = {
            "type": "person_left",
            "person_id": person_id,
        }

        logger.info(
            f"Sending person left event: {payload}"
        )

        self._schedule_broadcast(payload)

    def send_audio(
        self,
        url: str,
        priority: int = 2,
    ):

        payload = {
            "type": "audio",
            "url": url,
            "priority": priority,
        }

        self._schedule_broadcast(payload)

    def stop_audio(self):

        payload = {
            "type": "audio_stop",
        }

        self._schedule_broadcast(payload)

    def send_sign_translation(
        self,
        text: str,
    ):

        payload = {
            "type": "sign_translation",
            "text": text,
        }

        self._schedule_broadcast(payload)

    def send_status(
        self,
        state: str,
        frame_age: float = 0.0,
    ):
        """Broadcast current system state to Flutter clients."""

        payload = {
            "type": "status",
            "state": state,
            "frame_age": round(frame_age, 1),
        }

        self._schedule_broadcast(payload)

    def send_face_result(
        self,
        person: str,
        confidence: float,
    ):

        payload = {
            "type": "face_recognition",
            "person": person,
            "confidence": confidence,
        }

        self._schedule_broadcast(payload)


    def send_face_event(
        self,
        event_type: str,
        message_key: str,
        session_id: str = None,
        metadata: dict = None,
        text: str = None,
    ):
        """Broadcast face event (prompt, result, status) to Flutter clients."""
        payload = {
            "type": "face_event",
            "event_type": event_type,
            "message_key": message_key,
            "session_id": session_id,
            "text": text,
            "metadata": metadata or {},
        }
        logger.info(f"Sending Flutter face_event: {event_type}/{message_key}")
        self._schedule_broadcast(payload)