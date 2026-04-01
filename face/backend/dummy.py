import logging
from typing import Any, Dict, Optional, Tuple
from face.backend.base import FaceBackend

logger = logging.getLogger(__name__)


class DummyFaceBackend(FaceBackend):
    """Placeholder backend that performs no real inference.

    Keeps contract satisfaction while allowing integration testing of control flow.
    """

    def detect_and_align(self, frame) -> Optional[Any]:
        # No detection performed in dummy backend.
        return None

    def extract_embedding(self, aligned_face) -> Optional[Any]:
        return None

    def evaluate_liveness(self, frame, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        return False, {'reason': 'not_implemented'}

    def close(self) -> None:
        return
