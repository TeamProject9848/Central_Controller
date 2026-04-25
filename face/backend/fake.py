import hashlib
import logging
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from face.backend.base import FaceBackend

logger = logging.getLogger(__name__)


class FakeFaceBackend(FaceBackend):
    """Lightweight backend for integration testing without real models."""

    def __init__(self, embed_dim: int = 64):
        self._embed_dim = embed_dim

    def detect_and_align(self, frame) -> Optional[Any]:
        # Accept any non-empty frame as "detected"
        return frame

    def extract_embedding(self, aligned_face) -> Optional[List[float]]:
        try:
            data = np.asarray(aligned_face).tobytes()
        except Exception:
            data = bytes(str(aligned_face), encoding='utf-8')
        h = hashlib.sha256(data).digest()
        # Deterministic pseudo-embedding
        return [b / 255.0 for b in h[: self._embed_dim]]

    def evaluate_liveness(self, frame, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        return True, {'method': 'fake', 'score': 1.0}

    def close(self) -> None:
        return
