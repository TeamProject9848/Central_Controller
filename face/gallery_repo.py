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
