import logging

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

logger = logging.getLogger(__name__)


class WebRTCSignaling:

    def __init__(self, frame_buffer):
        self.peer_connection = None
        self.frame_buffer = frame_buffer

    async def handle_offer(self, offer_sdp: str):

        logger.info("Creating RTCPeerConnection")

        self.peer_connection = RTCPeerConnection()

        @self.peer_connection.on("track")
        async def on_track(track):

            logger.info(
                f"Received track: {track.kind}"
            )

            @track.on("ended")
            def _on_ended():
                logger.warning("Remote track ended")

            if track.kind != "video":
                logger.info(
                    f"Ignoring non-video track: {track.kind}"
                )
                return

            logger.info(
                "Receiving WebRTC video stream"
            )

            frame_count = 0

            while True:

                try:
                    frame = await track.recv()

                    frame_count += 1

                    if frame_count % 30 == 0:
                        logger.debug(
                            f"Received {frame_count} video frames"
                        )

                    image = frame.to_ndarray(
                        format="bgr24"
                    )

                    self.frame_buffer.push(
                        image
                    )

                    if frame_count % 30 == 0:
                        logger.debug(
                            f"Pushed frame {image.shape}"
                        )

                except MediaStreamError:
                    logger.warning("WebRTC video receiver: remote stream ended or unavailable")
                    break

                except Exception as e:
                    logger.error(
                        f"WebRTC video receiver stopped: {e}",
                        exc_info=True
                    )
                    break

        logger.info(
            "Applying remote SDP offer"
        )

        offer = RTCSessionDescription(
            sdp=offer_sdp,
            type="offer"
        )

        await self.peer_connection.setRemoteDescription(
            offer
        )

        logger.info(
            "Remote description set"
        )

        answer = await self.peer_connection.createAnswer()

        await self.peer_connection.setLocalDescription(
            answer
        )

        logger.info(
            "WebRTC answer generated"
        )

        return {
            "type": "webrtc_answer",
            "sdp": self.peer_connection.localDescription.sdp,
            "sdpType": self.peer_connection.localDescription.type,
        }

    async def close(self):

        if self.peer_connection:

            await self.peer_connection.close()

            self.peer_connection = None