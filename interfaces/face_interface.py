import logging
from abc import abstractmethod
from typing import Optional, Any, Dict
from interfaces.base import BaseModule

logger = logging.getLogger(__name__)


class FaceInterface(BaseModule):
    """Contract for face recognition/liveness modules."""

    def __init__(self):
        super().__init__('FaceModule')

    @abstractmethod
    def on_frame(self, frame) -> None:
        """Receive controller-owned frame stream."""
        ...

    @abstractmethod
    def start_registration(self, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        ...

    @abstractmethod
    def cancel_registration(self, session_id: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def request_identification(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        ...

    @abstractmethod
    def set_mode(self, mode: str) -> None:
        """Allow controller to set mode (e.g., identify/registration/idle)."""
        ...
