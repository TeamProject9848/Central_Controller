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


def test_person_near_alert_sends_tracker_id():
    from core.event_bus import VisionEvent, VisionEventType

    controller = CentralController()
    alerts_sent = []

    class MockFlutterServer:
        def send_alert(self, key, person_id=None):
            alerts_sent.append((key, person_id))

    controller.set_flutter_server(MockFlutterServer())

    event = VisionEvent(
        event_type=VisionEventType.RISK,
        confidence=0.8,
        hazard_class='person',
        depth_zone='NEAR',
        tracker_id=42
    )

    controller._on_risk_event(event)

    assert len(alerts_sent) == 1
    assert alerts_sent[0][0] == 'PERSON_NEAR'
    assert alerts_sent[0][1] == '42'


def test_person_left_event_sends_person_left_to_flutter():
    from core.event_bus import VisionEvent, VisionEventType

    controller = CentralController()
    left_events_sent = []

    class MockFlutterServer:
        def send_person_left(self, person_id):
            left_events_sent.append(person_id)

    controller.set_flutter_server(MockFlutterServer())

    event = VisionEvent(
        event_type=VisionEventType.PERSON_LEFT,
        tracker_id=42
    )

    controller._on_vision_event(event)

    assert len(left_events_sent) == 1
    assert left_events_sent[0] == '42'


def test_face_module_frame_distribution_gating():
    from core.controller import AppMode
    from core.state_machine import SystemState
    from unittest.mock import MagicMock

    controller = CentralController()
    mock_face = MagicMock()
    mock_face.is_running = True
    controller.register_face_module(mock_face)

    # By default, app mode is AppMode.DANGER
    assert controller.current_mode == AppMode.DANGER

    frame = MagicMock()
    controller._distribute_frame(frame)
    # Since current_mode is DANGER, on_frame should not be called on face module
    mock_face.on_frame.assert_not_called()

    # Change app mode to face
    controller.set_app_mode("face")
    assert controller.current_mode == AppMode.FACE

    controller._distribute_frame(frame)
    mock_face.on_frame.assert_called_once_with(frame)

    # Reset mock and enter ALERT state
    mock_face.reset_mock()
    controller._state_machine.transition(SystemState.ALERT)
    controller._distribute_frame(frame)
    # Under ALERT state, on_frame should not be called
    mock_face.on_frame.assert_not_called()


def test_set_app_mode_resets_face_module():
    from unittest.mock import MagicMock
    controller = CentralController()
    mock_face = MagicMock()
    controller.register_face_module(mock_face)

    controller.set_app_mode("face")
    mock_face.cancel_registration.assert_not_called()

    # Switch away from face mode
    controller.set_app_mode("danger")
    mock_face.cancel_registration.assert_called_once()
    mock_face.set_mode.assert_called_once_with('idle')

