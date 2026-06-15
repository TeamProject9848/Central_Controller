import time
import logging

from aiortc import VideoStreamTrack
from camera.buffer import TimestampedFrame

logger = logging.getLogger(__name__)


class WebRTCVideoReceiver:

    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer

    async def handle_track(self, track: VideoStreamTrack):

        logger.info("Receiving WebRTC video track")

        while True:

            frame = await track.recv()

            image = frame.to_ndarray(format="rgb24")

            self.frame_buffer.push(image)