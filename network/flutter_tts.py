import logging
from core.state_machine import SystemState

logger = logging.getLogger(__name__)


def send_caption_tts(controller, text: str):
    """
    Sends the image captioning result to the Flutter client via WebSocket.
    Strictly restricted to delivering captioning results only.
    Checks the current controller state to ensure it is not in ALERT.
    """
    if not text:
        return

    state = controller.state
    if state == SystemState.ALERT:
        logger.warning(
            f"Blocking flutter_tts delivery: system is in ALERT state. Text: '{text}'"
        )
        return

    if controller._flutter_server:
        payload = {
            "type": "flutter_tts",
            "text": text,
            "state": state.name,
        }
        logger.info(f"Sending caption result to Flutter TTS: '{text}'")
        controller._flutter_server._schedule_broadcast(payload)
    else:
        logger.warning(
            "Flutter server not registered — cannot send caption to Flutter TTS."
        )
