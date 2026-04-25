import types
from core.controller import CentralController
from core.event_bus import FaceEvent, FaceEventType
from config import FaceConfig


def test_face_prompt_dedup_same_session():
    controller = CentralController()
    recorded = []
    controller._post_audio = types.MethodType(lambda self, ct, text=None, alert_key=None, priority=None: recorded.append((text, priority)), controller)

    event = FaceEvent(event_type=FaceEventType.PROMPT, message_key='registration_start', session_id='s1')
    controller._on_face_event(event)
    controller._on_face_event(event)  # duplicate should be skipped

    assert len(recorded) == 1
    assert recorded[0][0] == FaceConfig.PROMPTS['registration_start']


def test_face_priority_mapping():
    controller = CentralController()
    assert controller._face_priority_for_event(FaceEventType.PROMPT) == FaceConfig.PRIORITY_GUIDANCE
    assert controller._face_priority_for_event(FaceEventType.REGISTRATION_PROGRESS) == FaceConfig.PRIORITY_GUIDANCE
    assert controller._face_priority_for_event(FaceEventType.IDENTIFIED) == FaceConfig.PRIORITY_RESULT
    assert controller._face_priority_for_event(FaceEventType.REGISTRATION_COMPLETE) == FaceConfig.PRIORITY_RESULT
    assert controller._face_priority_for_event(FaceEventType.REGISTRATION_FAILED) == FaceConfig.PRIORITY_CRITICAL


def test_face_prompt_replayed_after_alert_exit():
    controller = CentralController()
    recorded = []
    controller._post_audio = types.MethodType(lambda self, ct, text=None, alert_key=None, priority=None: recorded.append((text, priority)), controller)

    controller._pending_face_prompt = ('registration_start', FaceConfig.PRIORITY_GUIDANCE, 's1')
    controller._on_exit_alert()

    assert recorded[0][0] == FaceConfig.PROMPTS['registration_start']
