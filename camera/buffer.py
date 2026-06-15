import threading
import logging
from dataclasses import dataclass, field
from collections import deque
from time import time
from typing import Optional
import numpy as np
from config import CameraConfig, SystemConfig
logger = logging.getLogger(__name__)

@dataclass
class TimestampedFrame:
    frame: np.ndarray
    received_at: float = field(default_factory=time)
    frame_id: int = 0

    @property
    def age_ms(self) -> float:
        return (time() - self.received_at) * 1000

    @property
    def is_stale(self) -> bool:
        return self.age_ms > SystemConfig.MAX_FRAME_AGE_MS

class FrameBuffer:
    def latest(self) -> Optional[TimestampedFrame]:
        with self._lock:
            return self._last_frame

    def __init__(self, maxsize: int=CameraConfig.FRAME_BUFFER_SIZE):
        self._maxsize = maxsize
        self._buffer: deque = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._frame_id_counter = 0
        self._push_count = 0
        self._drop_count = 0
        self._pull_count = 0
        self._pull_miss = 0
        self._last_frame = None
        logger.debug(f'FrameBuffer initialized | maxsize={maxsize}')

    def push(self, frame: np.ndarray):
        with self._lock:
            self._frame_id_counter += 1
            current_id = self._frame_id_counter
            was_full = len(self._buffer) >= self._maxsize
            wrapped = TimestampedFrame(frame=frame, frame_id=current_id)
            self._last_frame = wrapped
            self._buffer.append(wrapped)
            self._push_count += 1
            if was_full:
                self._drop_count += 1
                logger.debug(f'Frame dropped (buffer full) — total drops: {self._drop_count}')

    def pull(self) -> Optional[TimestampedFrame]:
        with self._lock:
            if not self._buffer:
                self._pull_miss += 1
                return None
            frame = self._buffer[-1]
            self._buffer.clear()
            self._pull_count += 1
            if frame.is_stale:
                logger.warning(f'Pulled stale frame | age={frame.age_ms:.0f}ms | frame_id={frame.frame_id} | threshold={SystemConfig.MAX_FRAME_AGE_MS}ms')
            return frame

    def peek(self) -> Optional[TimestampedFrame]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def clear(self):
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            if count > 0:
                logger.debug(f'FrameBuffer cleared — {count} frames discarded')

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_stats(self) -> dict:
        with self._lock:
            drop_rate = self._drop_count / self._push_count if self._push_count > 0 else 0.0
            return {'push_count': self._push_count, 'drop_count': self._drop_count, 'drop_rate': round(drop_rate, 3), 'pull_count': self._pull_count, 'pull_miss': self._pull_miss, 'buffer_size': len(self._buffer), 'maxsize': self._maxsize}