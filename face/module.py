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
        # Registration flow: collect embeddings per pose, then mark complete.
        if self._mode == FaceMode.REGISTRATION and self._registration and self._registration.state == RegistrationState.IN_PROGRESS:
            if self._registration.last_prompt_key is None:
                self._emit_current_pose_prompt()
            self._process_registration_frame(frame)
        elif self._mode == FaceMode.IDENTIFY:
            self._process_identify_frame(frame)

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
        from core.event_bus import FaceEventType
        if self._registration and (session_id is None or session_id == self._registration.session_id):
            self._registration.state = RegistrationState.CANCELLED
            self._emit_event(
                FaceEventType.REGISTRATION_FAILED,
                message_key='registration_failed',
                session_id=self._registration.session_id,
                priority=FaceConfig.PRIORITY_CRITICAL,
            )
            self._registration.last_prompt_key = None
        self._mode = FaceMode.IDLE
        self._registration = None

    def request_identification(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._mode = FaceMode.IDENTIFY

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
            self._complete_registration()
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

    def _emit_event(self, event_type, message_key: Optional[str], session_id: Optional[str] = None, priority: int = FaceConfig.PRIORITY_RESULT, metadata: Optional[Dict[str, Any]] = None) -> None:
        from core.event_bus import FaceEvent, FaceEventType  # local import to avoid circular dep

        event_metadata = {
            'registration_state': self._registration.state.name if self._registration else None,
        }
        if metadata:
            event_metadata.update(metadata)
        event = FaceEvent(
            event_type=event_type,
            message_key=message_key,
            session_id=session_id,
            metadata=event_metadata,
            priority=priority,
        )
        self._emit(event)

    def _process_registration_frame(self, frame) -> None:
        aligned = self._backend.detect_and_align(getattr(frame, 'frame', frame))
        if aligned is None:
            return
        is_live, live_meta = self._backend.evaluate_liveness(aligned, metadata=self._registration.metadata if self._registration else None)
        if FaceConfig.LIVENESS_REQUIRED and not is_live:
            self._emit_prompt('registration_retry', self._registration.session_id if self._registration else None, FaceConfig.PRIORITY_GUIDANCE)
            return
        embedding = self._backend.extract_embedding(aligned)
        if embedding is None or self._registration is None:
            self._emit_prompt('registration_retry', self._registration.session_id if self._registration else None, FaceConfig.PRIORITY_GUIDANCE)
            return
        self._registration.embeddings.append(embedding)
        self.advance_registration_step(success=True)

    def _process_identify_frame(self, frame) -> None:
        aligned = self._backend.detect_and_align(getattr(frame, 'frame', frame))
        if aligned is None:
            return
        embedding = self._backend.extract_embedding(aligned)
        if embedding is None:
            return
        message_key = 'identify_unknown'
        priority = FaceConfig.PRIORITY_RESULT
        if self._gallery_repo:
            results = self._gallery_repo.search(embedding, top_k=1)
            if results:
                person_id, distance = results[0]
                metadata = {'person_id': person_id, 'distance': distance}
                if distance <= FaceConfig.MATCH_THRESHOLD_STRONG:
                    message_key = 'identify_success'
                self._emit_event(FaceEventType.IDENTIFIED, message_key=message_key, session_id=None, priority=priority, metadata=metadata)
                self._mode = FaceMode.IDLE
                return
        self._emit_event(FaceEventType.IDENTIFIED, message_key=message_key, session_id=None, priority=priority)
        self._mode = FaceMode.IDLE

    def _complete_registration(self) -> None:
        from core.event_bus import FaceEventType  # local import to avoid circular dep

        if not self._registration:
            return
        session = self._registration
        person_id = session.metadata.get('person_id') if session.metadata else None
        if isinstance(person_id, str):
            person_id = person_id.strip().lower()
            if session.metadata:
                session.metadata['person_id'] = person_id
        person_id = person_id or session.session_id
        if self._gallery_repo:
            wrote = self._gallery_repo.add_or_update(person_id, session.embeddings, metadata=session.metadata)
            if not wrote:
                session.state = RegistrationState.FAILED
                self._emit_event(
                    FaceEventType.REGISTRATION_FAILED,
                    message_key='registration_failed',
                    session_id=session.session_id,
                    priority=FaceConfig.PRIORITY_CRITICAL,
                    metadata={'person_id': person_id},
                )
                self._mode = FaceMode.IDLE
                return
        session.state = RegistrationState.COMPLETE
        self._emit_event(
            FaceEventType.REGISTRATION_COMPLETE,
            message_key='registration_complete',
            session_id=session.session_id,
            priority=FaceConfig.PRIORITY_RESULT,
            metadata={'person_id': person_id},
        )
        self._mode = FaceMode.IDLE
