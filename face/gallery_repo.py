from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple


class FaceGalleryRepo(ABC):
    """Abstract repository for face embeddings/templates."""

    @abstractmethod
    def add_or_update(self, person_id: str, embeddings: List[Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        ...

    @abstractmethod
    def search(self, embedding: Any, top_k: int = 1) -> List[Tuple[str, float]]:
        """Return list of (person_id, distance) sorted ascending by distance."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class InMemoryFaceGallery(FaceGalleryRepo):
    """Simple in-memory gallery for testing/integration without persistence."""

    def __init__(self):
        self._store: Dict[str, List[Any]] = {}
 
    def add_or_update(self, person_id: str, embeddings: List[Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        self._store[person_id] = embeddings
        return True

    def search(self, embedding: Any, top_k: int = 1) -> List[Tuple[str, float]]:
        results = []
        for pid, embs in self._store.items():
            for emb in embs:
                dist = self._distance(embedding, emb)
                results.append((pid, dist))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def close(self) -> None:
        return

    @staticmethod
    def _distance(a, b) -> float:
        try:
            import numpy as np
            va = np.array(a, dtype=float)
            vb = np.array(b, dtype=float)
            return float(np.linalg.norm(va - vb))
        except Exception:
            return 1.0
