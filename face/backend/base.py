from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class FaceBackend(ABC):
    """Backend adapter contract so models can be swapped without changing business logic."""

    @abstractmethod
    def detect_and_align(self, frame) -> Optional[Any]:
        """Return aligned face crop or None if not found/usable."""
        ...

    @abstractmethod
    def extract_embedding(self, aligned_face) -> Optional[Any]:
        """Return embedding vector or None on failure/low quality."""
        ...

    @abstractmethod
    def evaluate_liveness(self, frame, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """Return (is_live, details) for liveness challenge evaluation."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...
