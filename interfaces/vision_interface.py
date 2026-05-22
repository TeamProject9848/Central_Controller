import logging
from abc import abstractmethod
from typing import Optional
import numpy as np
from interfaces.base import BaseModule
from camera.buffer import TimestampedFrame
logger = logging.getLogger(__name__)

class VisionInterface(BaseModule):

    def __init__(self):
        super().__init__('VisionManager')
        self._vision_level = 'sentinel_only'

    @abstractmethod
    def on_frame(self, timestamped_frame: TimestampedFrame):
        ...

    def set_vision_level(self, level: str, state_context: str = ""):
        if level == self._vision_level:
            return
        state_str = f" [{state_context}]" if state_context else ""
        logger.info(f'[VisionManager] Vision level: {self._vision_level} → {level}{state_str}')
        self._vision_level = level
        self._apply_vision_level(level, state_context)

    @abstractmethod
    def _apply_vision_level(self, level: str, state_context: str = ""):
        ...

    @abstractmethod
    def request_caption(self):
        ...

    @abstractmethod
    def request_ocr(self):
        ...

    @abstractmethod
    def cancel_semantic_task(self):
        ...

    @property
    def vision_level(self) -> str:
        return self._vision_level