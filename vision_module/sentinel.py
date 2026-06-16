

import cv2
import numpy as np
import logging
from typing import Callable, Optional

from config import VisionConfig

logger = logging.getLogger(__name__)

NONE_EMIT_EVERY = 10 


class Sentinel:
    """
    Frame-by-frame motion detector.

    Compares each incoming frame against the previous one.
    Reports MOTION when pixel change exceeds configured thresholds.
    Reports NONE when the scene is stable.

    When MOTION is detected, optionally notifies the safety pipeline
    via on_motion_frame callback — so YOLO can be triggered reactively
    in states where it isn't running continuously (e.g. IDLE, SEMANTIC).

    Thread safety:
        process_frame() is called from the vision worker thread.
        Both callbacks are set once at init and never changed.
        _prev_gray is only accessed from the vision worker thread.
    """

    def __init__(self,
                 emit_callback: Callable,
                 on_motion_frame: Optional[Callable] = None):
        """
        Args:
            emit_callback:    Called with a VisionEvent (MOTION or NONE).
                              This is self._emit() from VisionManager.
            on_motion_frame:  Called with the raw frame when motion is detected.
                              Safety pipeline registers this to trigger reactive
                              YOLO checks in IDLE/SEMANTIC states.
                              None = no reactive YOLO (continuous mode handles it).
        """
        self._emit            = emit_callback
        self._on_motion_frame = on_motion_frame


        self._prev_gray: Optional[np.ndarray] = None

 
        self._pixel_threshold = VisionConfig.SENTINEL_MOTION_THRESHOLD
        self._area_ratio_min  = VisionConfig.SENTINEL_MOTION_AREA_RATIO

        # Stats for health monitoring
        self._frames_processed = 0
        self._motion_count     = 0
        self._none_skip_count  = 0

        logger.debug("Sentinel initialized")


    def process_frame(self, frame: np.ndarray):
        """
        Analyse one frame for motion. Emits MOTION or NONE.

        When MOTION is detected and on_motion_frame is registered,
        passes the frame to the safety pipeline for reactive YOLO check.

        Called by VisionManager on the vision worker thread for every
        frame received from the controller, regardless of system state.

        Args:
            frame: BGR numpy array from FrameBuffer.
        """
        self._frames_processed += 1

  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if (
            self._prev_gray is None
            or self._prev_gray.shape != gray.shape
        ):
            self._prev_gray = gray
            return

        delta = cv2.absdiff(
            self._prev_gray,
            gray
        )

        _, thresh = cv2.threshold(
            delta,
            self._pixel_threshold,
            255,
            cv2.THRESH_BINARY
        )

        changed_pixels = np.count_nonzero(thresh)
        total_pixels   = gray.shape[0] * gray.shape[1]
        area_ratio     = changed_pixels / total_pixels if total_pixels > 0 else 0.0

        self._prev_gray = gray

        if area_ratio >= self._area_ratio_min:
            self._motion_count += 1
            self._none_skip_count = 0
            self._emit_motion(confidence=area_ratio)

            if self._on_motion_frame is not None:
                try:
                    self._on_motion_frame(frame)
                except Exception as e:
                    logger.error(f"on_motion_frame callback raised: {e}",
                                 exc_info=True)
        else:
            self._none_skip_count += 1
            if self._none_skip_count >= NONE_EMIT_EVERY:
                self._none_skip_count = 0
                self._emit_none()

    def reset(self):
        """
        Clear the previous frame reference.

        Call this when the video stream reconnects or resumes after
        suspension — otherwise the first post-resume frame will be
        compared against a stale pre-suspend frame, producing a false
        motion spike.
        """
        self._prev_gray = None
        self._none_skip_count = 0
        logger.debug("Sentinel reset — previous frame cleared")

    def set_on_motion_frame(self, callback: Optional[Callable]):
        """
        Update the motion frame callback after construction.

        VisionManager calls this when switching between reactive mode
        (IDLE/SEMANTIC — callback set) and continuous mode
        (NAVIGATION/OVERRIDE — callback cleared, YOLO runs independently).

        Args:
            callback: Frame callback for safety pipeline, or None to disable.
        """
        self._on_motion_frame = callback
        state = "enabled" if callback else "disabled"
        logger.debug(f"Sentinel on_motion_frame callback {state}")

    def get_stats(self) -> dict:
        """Health snapshot for VisionManager debug reporting."""
        return {
            "frames_processed": self._frames_processed,
            "motion_events":    self._motion_count,
            "motion_rate":      round(
                self._motion_count / self._frames_processed, 3
            ) if self._frames_processed > 0 else 0.0,
        }



    def _emit_motion(self, confidence: float):
        """Emit a MOTION event with area_ratio as confidence score."""
        from core.event_bus import VisionEvent, VisionEventType
        try:
            self._emit(VisionEvent(
                event_type=VisionEventType.MOTION,
                confidence=round(min(confidence, 1.0), 4),
                source="sentinel",
            ))
        except Exception as e:
            logger.error(f"Sentinel failed to emit MOTION: {e}", exc_info=True)

    def _emit_none(self):
        """Emit a NONE event — scene is stable."""
        from core.event_bus import VisionEvent, VisionEventType
        try:
            self._emit(VisionEvent(
                event_type=VisionEventType.NONE,
                confidence=0.0,
                source="sentinel",
            ))
        except Exception as e:
            logger.error(f"Sentinel failed to emit NONE: {e}", exc_info=True)



if __name__ == "__main__":
    import sys, types

    config = types.ModuleType('config')
    class VisionConfig:
        SENTINEL_MOTION_THRESHOLD = 25
        SENTINEL_MOTION_AREA_RATIO = 0.02
    config.VisionConfig = VisionConfig
    sys.modules['config'] = config

    core = types.ModuleType('core')
    event_bus = types.ModuleType('core.event_bus')
    from enum import Enum, auto
    from dataclasses import dataclass, field
    from time import time as _time

    class VisionEventType(Enum):
        NONE = auto(); MOTION = auto(); RISK = auto()

    @dataclass
    class VisionEvent:
        event_type: VisionEventType
        confidence: float = 0.0
        hazard_class: str = None
        depth_zone: str = None
        timestamp: float = field(default_factory=_time)

    event_bus.VisionEvent = VisionEvent
    event_bus.VisionEventType = VisionEventType
    sys.modules['core'] = core
    sys.modules['core.event_bus'] = event_bus

    print("Running Sentinel tests...\n")
    events        = []
    motion_frames = []

    def mock_emit(event):       events.append(event)
    def mock_motion_cb(frame):  motion_frames.append(frame)

    sentinel = Sentinel(emit_callback=mock_emit,
                        on_motion_frame=mock_motion_cb)

    frame_black  = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_bright = np.ones((480, 640, 3), dtype=np.uint8) * 200

    sentinel.process_frame(frame_black)
    assert len(events) == 0
    print("PASS  Test 1: First frame produces no event")

    for _ in range(NONE_EMIT_EVERY):
        sentinel.process_frame(frame_black)
    assert events[-1].event_type == VisionEventType.NONE
    assert len(motion_frames) == 0
    print("PASS  Test 2: Identical frames → NONE (rate-limited), no motion callback")

    sentinel.process_frame(frame_bright)
    assert events[-1].event_type == VisionEventType.MOTION
    assert events[-1].confidence > 0.0
    assert len(motion_frames) == 1
    print(f"PASS  Test 3: Large change → MOTION + callback fired "
          f"(confidence={events[-1].confidence})")

    sentinel.reset()
    assert sentinel._prev_gray is None
    sentinel.process_frame(frame_black)
    assert events[-1].event_type == VisionEventType.MOTION
    print("PASS  Test 4: reset() clears state")

    sentinel.reset()
    sentinel.set_on_motion_frame(None)
    before = len(motion_frames)
    sentinel.process_frame(frame_black)   
    sentinel.process_frame(frame_bright)  # motion
    assert len(motion_frames) == before
    print("PASS  Test 5: set_on_motion_frame(None) disables callback")

    sentinel.set_on_motion_frame(mock_motion_cb)
    sentinel.process_frame(frame_black)
    assert len(motion_frames) == before + 1
    print("PASS  Test 6: Re-enabling callback works")


    sentinel._emit_motion(confidence=5.0)
    assert events[-1].confidence <= 1.0
    print("PASS  Test 7: Confidence clamped to 1.0")


    stats = sentinel.get_stats()
    assert stats["frames_processed"] > 0
    assert "motion_rate" in stats
    print(f"PASS  Test 8: Stats tracked — {stats}")

    print("\nAll Sentinel tests passed.")