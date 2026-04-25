from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List
from time import time


class FaceMode(Enum):
    IDLE = auto()
    IDENTIFY = auto()
    REGISTRATION = auto()


class RegistrationState(Enum):
    IDLE = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class RegistrationSession:
    session_id: str
    required_poses: List[str]
    current_index: int = 0
    state: RegistrationState = RegistrationState.IN_PROGRESS
    started_at: float = field(default_factory=time)
    metadata: Optional[Dict[str, Any]] = None
    last_prompt_key: Optional[str] = None
    embeddings: List[Any] = field(default_factory=list)
