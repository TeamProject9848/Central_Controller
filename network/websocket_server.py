import json
import logging
import asyncio

import websockets

from network.flutter_bridge import FlutterBridge
from network.webrtc_signaling import WebRTCSignaling
from core.event_bus import IntentEvent, IntentEventType

logger = logging.getLogger(__name__)


class CompanionWebSocketServer:

    def __init__(self, controller):

        self.controller = controller

        self.bridge = FlutterBridge()

        self.webrtc = WebRTCSignaling(
            controller._frame_buffer
        )

    async def handler(self, websocket):

        await self.bridge.register(websocket)

        try:

            async for message in websocket:

                try:
                    payload = json.loads(message)

                except Exception as e:
                    logger.error(
                        f"Invalid websocket message: {e}"
                    )
                    continue

                msg_type = payload.get("type")

                logger.debug(
                    f"Received websocket message: {msg_type}"
                )

                if msg_type == "webrtc_offer":

                    logger.info(
                        "Received WebRTC offer"
                    )

                    response = await self.webrtc.handle_offer(
                        payload["sdp"]
                    )

                    await websocket.send(
                        json.dumps(response)
                    )

                elif msg_type == "set_mode":

                    mode = payload.get(
                        "mode",
                        "danger"
                    )

                    logger.info(
                        f"App mode switched to {mode}"
                    )

                    self.controller.set_app_mode(
                        mode
                    )

                elif msg_type == "face_intent":
                    if self.controller.current_mode.value != "face":
                        logger.warning(f"Ignored face intent '{payload.get('intent')}' because current app mode is {self.controller.current_mode.value}")
                        continue
                    intent = payload.get("intent", "")
                    logger.info(f"Face intent received: {intent}")
                    
                    intent_map = {
                        "start_registration": IntentEventType.START_FACE_REGISTRATION,
                        "cancel_registration": IntentEventType.CANCEL_FACE_REGISTRATION,
                        "identify": IntentEventType.IDENTIFY_FACE,
                    }
                    event_type = intent_map.get(intent, IntentEventType.UNKNOWN)
                    
                    # Extract metadata (person_id, etc.) from payload
                    metadata = payload.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {k: v for k, v in payload.items() 
                                    if k not in ("type", "intent")}
                    
                    self.controller._event_bus.post(
                        IntentEvent(event_type=event_type, raw_input=intent, metadata=metadata)
                    )
                    logger.debug(f"Posted IntentEvent: {event_type.name} with metadata {metadata}")

        except Exception as e:

            logger.error(
                f"Websocket handler error: {e}",
                exc_info=True
            )

        finally:

            await self.bridge.unregister(
                websocket
            )

    async def start(
        self,
        host="0.0.0.0",
        port=8765
    ):

        logger.info(
            f"Flutter websocket server running on {host}:{port}"
        )

        # Capture and store the running event loop so other threads
        # can schedule coroutines safely via FlutterBridge.
        loop = asyncio.get_running_loop()
        try:
            self.bridge.set_loop(loop)
        except Exception as e:
            logger.error(f"Failed to set FlutterBridge event loop: {e}")

        return await websockets.serve(
            self.handler,
            host,
            port
        )
