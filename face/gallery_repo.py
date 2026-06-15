import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


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


class PickleFaceGallery(FaceGalleryRepo):
    """Persistent gallery matching the standalone PoC shape."""

    def __init__(self, path: str):
        self._path = path
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load()

    def add_or_update(self, person_id: str, embeddings: List[Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            import numpy as np

            existing = self._store.get(person_id, {'embeddings': []})
            combined = list(existing.get('embeddings', [])) + list(embeddings)
            if not combined:
                return False
            emb_arr = np.stack([np.asarray(e, dtype=float) for e in combined], axis=0)
            self._store[person_id] = {
                'prototype': np.mean(emb_arr, axis=0),
                'embeddings': combined,
                'metadata': metadata or existing.get('metadata'),
            }
            self._save()
            return True
        except Exception as exc:
            logger.error(f'Failed to update face gallery: {exc}', exc_info=True)
            return False

    def search(self, embedding: Any, top_k: int = 1) -> List[Tuple[str, float]]:
        results = []
        for person_id, record in self._store.items():
            prototype = record.get('prototype')
            if prototype is None:
                continue
            results.append((person_id, self._distance(embedding, prototype)))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def close(self) -> None:
        self._save()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            self._store = {}
            return
        try:
            with open(self._path, 'rb') as fh:
                loaded = pickle.load(fh)
            self._store = loaded if isinstance(loaded, dict) else {}
        except Exception as exc:
            logger.warning(f'Could not load face gallery {self._path}: {exc}')
            self._store = {}

    def _save(self) -> None:
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self._path, 'wb') as fh:
            pickle.dump(self._store, fh)

    @staticmethod
    def _distance(a, b) -> float:
        try:
            import numpy as np
            va = np.asarray(a, dtype=float)
            vb = np.asarray(b, dtype=float)
            return float(np.linalg.norm(va - vb))
        except Exception:
            return 1.0
