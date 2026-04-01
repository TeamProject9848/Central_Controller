import logging
import uuid
from typing import Optional, Dict, Any, List
from interfaces.face_interface import FaceInterface
from face.backend.base import FaceBackend
from face.backend.dummy import DummyFaceBackend
from face.gallery_repo import FaceGalleryRepo
from face.models import FaceMode, RegistrationSession, RegistrationState
from config import FaceConfig
from core.event_bus import FaceEventType

logger = logging.getLogger(__name__)


class FaceModule(FaceInterface):
    """Incremental face module scaffold.

    - Keeps controller-owned policy: emits FaceEvents only.
    - Uses dummy backend by default to avoid impacting runtime until real backend is provided.
    - Maintains registration session state without marking completion automatically.
    """

    def __init__(self, backend: Optional[FaceBackend] = None, gallery_repo: Optional[FaceGalleryRepo] = None):
        super().__init__()
        self._backend = backend or DummyFaceBackend()
        self._gallery_repo = gallery_repo
        self._mode = FaceMode.IDLE
        self._registration: Optional[RegistrationSession] = None

    # BaseModule hooks
    def _on_start(self):
        self._mode = FaceMode.IDLE

    def _on_stop(self):
        try:
            self._backend.close()
        except Exception:
            pass
        try:
            if self._gallery_repo:
                self._gallery_repo.close()
        except Exception:
            pass
        self._mode = FaceMode.IDLE
        self._registration = None

    def _on_suspend(self):
        # No-op for now; inference is already lightweight/dummy.
        return

    def _on_resume(self):
        return

    # FaceInterface contract
    def on_frame(self, frame) -> None:
        # Placeholder: during registration, re-emit current pose prompt once if not sent.
        if self._mode == FaceMode.REGISTRATION and self._registration and self._registration.state == RegistrationState.IN_PROGRESS:
            if self._registration.last_prompt_key is None:
                self._emit_current_pose_prompt()

    def start_registration(self, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.state.name == 'ERROR':
            logger.warning('[FaceModule] Cannot start registration while in ERROR state')
            return
        sid = session_id or str(uuid.uuid4())
        required_poses: List[str] = getattr(FaceConfig, 'REQUIRED_POSES', ['front', 'left', 'right'])
        self._registration = RegistrationSession(session_id=sid, required_poses=list(required_poses), metadata=metadata)
        self._mode = FaceMode.REGISTRATION
        self._emit_current_pose_prompt()

    def cancel_registration(self, session_id: Optional[str] = None) -> None:
        if self._registration and (session_id is None or session_id == self._registration.session_id):
            self._registration.state = RegistrationState.CANCELLED
            self._emit_prompt('registration_failed', self._registration.session_id, FaceConfig.PRIORITY_CRITICAL)
            self._registration.last_prompt_key = None
        self._mode = FaceMode.IDLE
        self._registration = None

    def request_identification(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        # For now, just emit a placeholder prompt; real match logic will come with backend wiring.
        self._mode = FaceMode.IDENTIFY
        self._emit_prompt('identify_unknown', None, FaceConfig.PRIORITY_RESULT)

    def set_mode(self, mode: str) -> None:
        try:
            self._mode = FaceMode[mode.upper()]
        except Exception:
            logger.warning(f'[FaceModule] Unknown mode requested: {mode}')

    # Internal helpers
    def advance_registration_step(self, success: bool = True) -> None:
        """Advance to next pose or emit retry; placeholder until backend drives captures."""
        if not self._registration or self._registration.state != RegistrationState.IN_PROGRESS:
            return
        if not success:
            self._emit_prompt('registration_retry', self._registration.session_id, FaceConfig.PRIORITY_GUIDANCE)
            return
        from core.event_bus import FaceEventType  # local import to avoid circular dep
        self._registration.current_index += 1
        if self._registration.current_index >= len(self._registration.required_poses):
            self._registration.state = RegistrationState.COMPLETE
            self._emit_event(FaceEventType.REGISTRATION_COMPLETE, message_key='registration_complete', priority=FaceConfig.PRIORITY_RESULT)
            self._mode = FaceMode.IDLE
            return
        self._registration.last_prompt_key = None
        self._emit_current_pose_prompt()

    def _emit_current_pose_prompt(self) -> None:
        if not self._registration or self._registration.state != RegistrationState.IN_PROGRESS:
            return
        pose = self._registration.required_poses[self._registration.current_index]
        key_map = {'front': 'registration_start', 'left': 'registration_left', 'right': 'registration_right'}
        message_key = key_map.get(pose, f'registration_{pose}')
        self._registration.last_prompt_key = message_key
        self._emit_prompt(message_key, self._registration.session_id, FaceConfig.PRIORITY_GUIDANCE)

    def _emit_prompt(self, message_key: str, session_id: Optional[str], priority: int) -> None:
        self._emit_event(FaceEventType.PROMPT, message_key=message_key, session_id=session_id, priority=priority)

    def _emit_event(self, event_type, message_key: Optional[str], session_id: Optional[str], priority: int) -> None:
        from core.event_bus import FaceEvent, FaceEventType  # local import to avoid circular dep

        event = FaceEvent(
            event_type=event_type,
            message_key=message_key,
            session_id=session_id,
            metadata={'registration_state': self._registration.state.name if self._registration else None},
            priority=priority,
        )
        self._emit(event)
