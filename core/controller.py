import threading
import logging
import time
import sys
from typing import Optional
from config import CameraConfig, SystemConfig, AudioConfig, FaceConfig
from core.event_bus import (
    EventBus,
    VisionEvent,
    VisionEventType,
    StreamEvent,
    StreamEventType,
    IntentEvent,
    IntentEventType,
    AudioCommand,
    AudioCommandType,
    FaceEvent,
    FaceEventType,
)
import asyncio
from core.state_machine import StateMachine, SystemState
from camera.source import CameraSource
from camera.buffer import FrameBuffer, TimestampedFrame
from audio.queue import AudioQueue
from interfaces.base import BaseModule
from interfaces.face_interface import FaceInterface
from enum import Enum

class AppMode(Enum):
    DANGER = "danger"
    FACE = "face"
    SIGN = "sign"
    CAPTION = "caption"
logger = logging.getLogger(__name__)

class CentralController:

    def __init__(self):
        self.current_mode = AppMode.DANGER
        self._flutter_server = None
        self._event_bus = EventBus()
        self._state_machine = StateMachine(initial_state=SystemState.IDLE)
        self._frame_buffer = FrameBuffer()
        self._camera_source = CameraSource(self._event_bus, self._frame_buffer)
        self._audio_queue = AudioQueue()
        self._vision_module = None
        self._audio_module = None
        self._input_module = None
        self._face_module: Optional[FaceInterface] = None
        self._sign_module = None
        self._last_face_prompt: Optional[tuple[str, Optional[str]]] = None  # (session_id, message_key)
        self._pending_face_prompt: Optional[tuple[str, int, Optional[str]]] = None  # (message_key, priority, session_id)
        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        self._alert_safe_since: Optional[float] = None   # When safety pipeline first said NONE during ALERT
        self._last_risk_time: float = 0.0
        self._pre_alert_state: Optional[SystemState] = None
        self._current_semantic_task = None
        self._register_event_handlers()
        self._register_state_hooks()
        logger.info('CentralController initialized')


    def register_vision_module(self, module):
        self._vision_module = module
        module.set_event_callback(self._event_bus.post)
        
        # Route semantic results (caption/OCR) appropriately
        module.set_semantic_result_callback(self._on_semantic_result)
        
        logger.info(f"Vision module registered: {module.module_name}")

    def register_audio_module(self, module):
        self._audio_module = module
        logger.info(f'Audio module registered: {module.module_name}')

    def register_input_module(self, module):
        self._input_module = module
        module.set_event_callback(self._event_bus.post)
        logger.info(f'Input module registered: {module.module_name}')

    def register_face_module(self, module: FaceInterface):
        self._face_module = module
        module.set_event_callback(self._event_bus.post)
        logger.info(f'Face module registered: {module.module_name}')

    def register_sign_module(self, module):
        self._sign_module = module
        module.on_sentence_callback = self.send_sign_translation
        module.is_active_callback = lambda: self.current_mode == AppMode.SIGN
        logger.info("Sign module registered")

    def start(self, blocking: bool=True):
        if self._running:
            logger.warning('Controller already running')
            return
        logger.info('=' * 60)
        logger.info('CENTRAL CONTROLLER STARTING')
        logger.info('=' * 60)
        self._running = True
        self._audio_queue.start()
        self._start_guest_modules()
        logger.info(
            "Waiting for Flutter WebRTC camera stream..."
        )
        connected = True
        self._apply_vision_level()
        self._post_audio(AudioCommandType.SPEAK, text='System ready.', priority=AudioConfig.PRIORITY_NAVIGATION_STATUS)
        logger.info('Controller startup complete — entering main loop')
        if blocking:
            self._main_loop()
        else:
            self._main_thread = threading.Thread(target=self._main_loop, name='ControllerMainLoop', daemon=False)
            self._main_thread.start()

    def stop(self):
        if not self._running:
            return
        logger.info('Controller stopping...')
        self._running = False
        self._camera_source.stop()
        self._stop_guest_modules()
        self._audio_queue.stop()
        logger.info('Controller stopped cleanly')

    def _main_loop(self):
        logger.info('Main loop started')
        tick_count = 0
        while self._running:
            tick_start = time.time()
            tick_count += 1
            try:
                frame = self._frame_buffer.pull()
                if frame is not None:
                    self._distribute_frame(frame)
                self._event_bus.process_events(max_events=10)
                if tick_count % 100 == 0:
                    self._health_check()
                if SystemConfig.DEBUG_DISPLAY:
                    self._update_debug_display(frame)
            except Exception as e:
                logger.error(f'Main loop error (tick {tick_count}): {e}', exc_info=True)
            elapsed = time.time() - tick_start
            # --- CHECK ALERT TIMER ---
            if self._state_machine.state == SystemState.ALERT and self._alert_safe_since is not None:
                safe_duration = time.time() - self._alert_safe_since
                if safe_duration >= AudioConfig.ALERT_RESOLVE_SEC:
                    logger.info(f'Alert safe for {safe_duration:.1f}s — resolving')
                    self._resolve_alert()
            sleep_time = SystemConfig.CONTROLLER_TICK_SEC - (time.time() - tick_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > SystemConfig.CONTROLLER_TICK_SEC * 2:
                logger.debug(f'Tick overrun: {elapsed * 1000:.1f}ms (target={SystemConfig.CONTROLLER_TICK_SEC * 1000:.0f}ms)')
        logger.info(f'Main loop exited after {tick_count} ticks')

    def _distribute_frame(self, frame: TimestampedFrame):
        current_state = self._state_machine.state
        if self._vision_module and self._vision_module.is_running:
            try:
                self._vision_module.on_frame(frame)
            except Exception as e:
                logger.error(f'Vision module on_frame raised: {e}', exc_info=True)
        if self._sign_module:
            try:
                self._sign_module.on_frame(frame)
            except Exception as e:
                logger.error(f'Sign module on_frame raised: {e}', exc_info=True)
        if (
            self.current_mode == AppMode.FACE
            and self._face_module
            and self._face_module.is_running
            and (current_state not in (SystemState.ALERT, SystemState.ACTIVE_WALK_OVERRIDE))
        ):
            try:
                self._face_module.on_frame(frame)
            except Exception as e:
                logger.error(f'Face module on_frame raised: {e}', exc_info=True)
        if self._input_module and self._input_module.is_running and (current_state not in (SystemState.ALERT, SystemState.ACTIVE_WALK_OVERRIDE)):
            try:
                if hasattr(self._input_module, 'on_frame'):
                    self._input_module.on_frame(frame.frame)
            except Exception as e:
                logger.error(f'Input module on_frame raised: {e}', exc_info=True)

    def _register_event_handlers(self):
        self._event_bus.register_risk_callback(self._on_risk_event)
        self._event_bus.register_handler(VisionEvent, self._on_vision_event)
        self._event_bus.register_handler(StreamEvent, self._on_stream_event)
        self._event_bus.register_handler(IntentEvent, self._on_intent_event)
        self._event_bus.register_handler(FaceEvent, self._on_face_event)
        logger.debug('Event handlers registered')

    def _on_risk_event(self, event: VisionEvent):
        now = time.time()
        if now - self._last_risk_time < AudioConfig.ALERT_COOLDOWN_SEC:
            logger.debug('RISK event within cooldown window — suppressing duplicate')
            return
        self._last_risk_time = now
        self._alert_safe_since = None   # Reset safety timer — hazard confirmed
        logger.warning(f'RISK INTERRUPT | class={event.hazard_class} | confidence={event.confidence:.2f} | depth={event.depth_zone}')
        current = self._state_machine.state
        if current != SystemState.ALERT:
            self._pre_alert_state = current
        if self._state_machine.can_transition_to(SystemState.ALERT):
            self._state_machine.transition(SystemState.ALERT, reason=f'RISK: {event.hazard_class} @ {event.confidence:.2f}')
        alert_key = self._select_alert_key(event)

        if self._flutter_server:
            person_id = str(event.tracker_id) if (event.hazard_class == 'person' and event.tracker_id is not None) else None
            self._flutter_server.send_alert(alert_key, person_id=person_id)
    
        self._speak_alert(alert_key)

    def _on_vision_event(self, event: VisionEvent):
        if event.event_type == VisionEventType.MOTION:
            logger.debug(f'MOTION detected | confidence={event.confidence:.2f}')
            # NOTE: We do NOT reset _alert_safe_since here.
            # Sentinel MOTION means "pixels changed" — not "hazard present".
            # Only actual RISK events (via _on_risk_event) reset the timer.
        elif event.event_type == VisionEventType.NONE:
            if self._state_machine.state == SystemState.ALERT and event.source == "safety":
                # Safety pipeline confirmed scene is clear — start or continue the timer
                # Ignore sentinel NONE (just means "no pixel motion", not "no hazard")
                if self._alert_safe_since is None:
                    self._alert_safe_since = time.time()
                    logger.debug(f'Alert safety timer started — will resolve in {AudioConfig.ALERT_RESOLVE_SEC}s')
        elif event.event_type == VisionEventType.PERSON_LEFT:
            logger.info(f"PERSON LEFT event received for tracker_id={event.tracker_id}")
            if self._flutter_server:
                self._flutter_server.send_person_left(str(event.tracker_id))

    def _on_stream_event(self, event: StreamEvent):
        if event.event_type == StreamEventType.LOST:
            logger.error(f'Stream LOST: {event.reason}')
            self._speak_alert('STREAM_LOST')
            current = self._state_machine.state
            if current in (SystemState.NAVIGATION, SystemState.ACTIVE_WALK_OVERRIDE, SystemState.ALERT):
                if self._state_machine.can_transition_to(SystemState.IDLE):
                    self._state_machine.transition(SystemState.IDLE, reason='Stream lost — cannot navigate safely')
        elif event.event_type == StreamEventType.RECONNECTING:
            logger.info(f'Stream reconnecting: {event.reason}')
        elif event.event_type == StreamEventType.CONNECTED:
            logger.info('Stream reconnected')
            self._speak_alert('STREAM_RECONNECTED')

    def _on_intent_event(self, event: IntentEvent):
        current_state = self._state_machine.state
        if event.event_type == IntentEventType.TOGGLE_OVERRIDE:
            self._handle_override_toggle()
            return
        if current_state == SystemState.ALERT:
            logger.debug(f'Intent {event.event_type.name} ignored — in ALERT state')
            return
        if current_state == SystemState.ACTIVE_WALK_OVERRIDE:
            if event.event_type not in (IntentEventType.STOP_NAVIGATION,):
                logger.debug(f'Intent {event.event_type.name} ignored — in OVERRIDE mode')
                return
        if event.event_type in (IntentEventType.START_FACE_REGISTRATION, IntentEventType.CANCEL_FACE_REGISTRATION, IntentEventType.IDENTIFY_FACE):
            if self.current_mode != AppMode.FACE:
                logger.warning(f"Ignored face intent event {event.event_type.name} because current mode is {self.current_mode.name}")
                return
        handlers = {IntentEventType.START_NAVIGATION: lambda e: self._handle_start_navigation(), IntentEventType.STOP_NAVIGATION: lambda e: self._handle_stop_navigation(), IntentEventType.REQUEST_CAPTION: lambda e: self._handle_request_caption(), IntentEventType.REQUEST_OCR: lambda e: self._handle_request_ocr(), IntentEventType.START_FACE_REGISTRATION: self._handle_start_face_registration, IntentEventType.CANCEL_FACE_REGISTRATION: self._handle_cancel_face_registration, IntentEventType.IDENTIFY_FACE: self._handle_identify_face, IntentEventType.UNKNOWN: lambda e: self._handle_unknown_intent()}
        handler = handlers.get(event.event_type)
        if handler:
            try:
                handler(event)
            except Exception as e:
                logger.error(f'Intent handler for {event.event_type.name} raised: {e}', exc_info=True)
        else:
            logger.warning(f'No handler for intent: {event.event_type.name}')

    def _on_face_event(self, event: FaceEvent):
        message_key = event.message_key or ''
        text = FaceConfig.PROMPTS.get(message_key, message_key)
        if not text:
            logger.debug('FaceEvent ignored - no message_key/prompt text provided')
            return

        # Announce the identified person's name on success
        if message_key == 'identify_success' and event.metadata:
            person_id = event.metadata.get('person_id') or event.metadata.get('person_name')
            if person_id:
                text = f"{text} {person_id}"

        priority = event.priority if event.priority is not None else self._face_priority_for_event(event.event_type)
        if self._state_machine.state == SystemState.ALERT:
            self._pending_face_prompt = (message_key, priority, event.session_id, text)
            logger.debug('Face prompt deferred due to ALERT state')
            # Forward face event to Flutter even when deferred
            if self._flutter_server:
                self._flutter_server.send_face_event(
                    event_type=event.event_type.name,
                    message_key=message_key,
                    session_id=event.session_id,
                    metadata=event.metadata,
                    text=text,
                )
            return
        last_key = self._last_face_prompt
        if last_key and last_key[0] == event.session_id and last_key[1] == message_key:
            logger.debug('Duplicate face prompt skipped')
            return
        self._last_face_prompt = (event.session_id, message_key)
        self._post_audio(AudioCommandType.SPEAK, text=text, priority=priority)
        
        # Forward face event to Flutter UI
        if self._flutter_server:
            self._flutter_server.send_face_event(
                event_type=event.event_type.name,
                message_key=message_key,
                session_id=event.session_id,
                metadata=event.metadata,
                text=text,
            )

    def _handle_start_navigation(self):
        if self._state_machine.can_transition_to(SystemState.NAVIGATION):
            self._state_machine.transition(SystemState.NAVIGATION, reason='User started navigation')
            self._speak_alert('NAVIGATION_START')
        else:
            logger.debug('START_NAVIGATION intent ignored — transition not allowed')

    def _handle_stop_navigation(self):
        current = self._state_machine.state
        if current in (SystemState.NAVIGATION, SystemState.ACTIVE_WALK_OVERRIDE):
            if self._state_machine.can_transition_to(SystemState.IDLE):
                self._state_machine.transition(SystemState.IDLE, reason='User stopped navigation')
                self._speak_alert('NAVIGATION_STOP')

    def _handle_request_caption(self):
        if self._state_machine.can_transition_to(SystemState.SEMANTIC):
            if self._vision_module:
                self._current_semantic_task = "caption"
                self._state_machine.transition(SystemState.SEMANTIC, reason='User requested caption')
                self._vision_module.request_caption()
            else:
                logger.warning('Caption requested but no vision module registered')
        else:
            logger.debug('Caption request ignored — cannot enter SEMANTIC state now')

    def _handle_request_ocr(self):
        if self._state_machine.can_transition_to(SystemState.SEMANTIC):
            if self._vision_module:
                self._current_semantic_task = "ocr"
                self._state_machine.transition(SystemState.SEMANTIC, reason='User requested OCR')
                self._vision_module.request_ocr()
            else:
                logger.warning('OCR requested but no vision module registered')
        else:
            logger.debug('OCR request ignored - cannot enter SEMANTIC state now')

    def _on_semantic_result(self, text: str):
        is_caption = self._current_semantic_task == "caption" or text.startswith("I see: ")
        
        if is_caption:
            from network.flutter_tts import send_caption_tts
            send_caption_tts(self, text)
        else:
            self._post_audio(
                AudioCommandType.SPEAK,
                text=text,
                priority=AudioConfig.PRIORITY_RESPONSE
            )
        
        # Clean up caption state
        self._current_semantic_task = None

        if self._state_machine.state == SystemState.SEMANTIC:
            self._state_machine.transition(SystemState.IDLE, reason='Semantic task completed')

    def _handle_start_face_registration(self, event: IntentEvent):
        if not self._face_module:
            logger.warning('Face registration requested but no face module registered')
            return
        metadata = event.metadata or {}
        self._face_module.start_registration(metadata=metadata)
        logger.info(f'Face registration started with metadata: {metadata}')

    def _handle_cancel_face_registration(self, event: IntentEvent):
        if not self._face_module:
            logger.warning('Cancel face registration requested but no face module registered')
            return
        metadata = event.metadata or {}
        self._face_module.cancel_registration(metadata=metadata)

    def _handle_identify_face(self, event: IntentEvent):
        if not self._face_module:
            logger.warning('Identify face requested but no face module registered')
            return
        metadata = event.metadata or {}
        self._face_module.request_identification(metadata=metadata)

    def _handle_override_toggle(self):
        current = self._state_machine.state
        if current == SystemState.ACTIVE_WALK_OVERRIDE:
            if self._state_machine.can_transition_to(SystemState.NAVIGATION):
                self._state_machine.transition(SystemState.NAVIGATION, reason='User deactivated override')
                self._speak_alert('OVERRIDE_OFF')
        elif current == SystemState.NAVIGATION:
            if self._state_machine.can_transition_to(SystemState.ACTIVE_WALK_OVERRIDE):
                self._state_machine.transition(SystemState.ACTIVE_WALK_OVERRIDE, reason='User activated override')
                self._speak_alert('OVERRIDE_ON')
        else:
            logger.debug(f'Override toggle ignored — current state is {current.name}')

    def _handle_unknown_intent(self):
        logger.debug('Unknown intent detected — ignoring')

    def _resolve_alert(self):
        self._alert_safe_since = None
        target = self._pre_alert_state or SystemState.NAVIGATION
        logger.info(f'Alert resolved — returning to {target.name}')
        if self._state_machine.can_transition_to(target):
            self._state_machine.transition(target, reason=f'Hazard cleared after {AudioConfig.ALERT_RESOLVE_SEC}s of safety')
        elif self._state_machine.can_transition_to(SystemState.IDLE):
            self._state_machine.transition(SystemState.IDLE, reason='Alert resolved — fallback to IDLE')
        self._pre_alert_state = None

    def _select_alert_key(self, event: VisionEvent) -> str:
        hazard = event.hazard_class or ''
        depth = event.depth_zone or ''
        if hazard == 'person' and depth == 'NEAR':
            return 'PERSON_NEAR'
        elif hazard in ('car', 'truck', 'bus', 'motorcycle') and depth in ('NEAR', 'MID'):
            return 'VEHICLE_NEAR'
        elif depth == 'NEAR':
            return 'OBSTACLE_NEAR'
        elif depth == 'MID':
            return 'OBSTACLE_MID'
        else:
            return 'OBSTACLE_NEAR'

    def _face_priority_for_event(self, event_type: FaceEventType) -> int:
        if event_type == FaceEventType.PROMPT or event_type == FaceEventType.REGISTRATION_PROGRESS:
            return FaceConfig.PRIORITY_GUIDANCE
        if event_type == FaceEventType.IDENTIFIED or event_type == FaceEventType.REGISTRATION_COMPLETE:
            return FaceConfig.PRIORITY_RESULT
        if event_type == FaceEventType.REGISTRATION_FAILED:
            return FaceConfig.PRIORITY_CRITICAL
        return AudioConfig.PRIORITY_RESPONSE

    def _register_state_hooks(self):
        sm = self._state_machine
        sm.on_enter(SystemState.IDLE, self._on_enter_idle)
        sm.on_exit(SystemState.IDLE, self._on_exit_idle)
        sm.on_enter(SystemState.NAVIGATION, self._on_enter_navigation)
        sm.on_exit(SystemState.NAVIGATION, self._on_exit_navigation)
        sm.on_enter(SystemState.ALERT, self._on_enter_alert)
        sm.on_exit(SystemState.ALERT, self._on_exit_alert)
        sm.on_enter(SystemState.ACTIVE_WALK_OVERRIDE, self._on_enter_override)
        sm.on_exit(SystemState.ACTIVE_WALK_OVERRIDE, self._on_exit_override)
        sm.on_enter(SystemState.SEMANTIC, self._on_enter_semantic)
        sm.on_exit(SystemState.SEMANTIC, self._on_exit_semantic)
        logger.debug('State hooks registered')

    def _on_enter_idle(self):
        self._apply_vision_level()
        self._audio_queue.unlock_audio()
        self._send_flutter_status()

    def _on_exit_idle(self):
        pass

    def _on_enter_navigation(self):
        self._apply_vision_level()
        self._audio_queue.unlock_audio()
        if self._input_module and self._input_module.is_suspended:
            self._input_module.resume()
        self._send_flutter_status()

    def _on_exit_navigation(self):
        pass

    def _on_enter_alert(self):
        self._apply_vision_level()
        self._audio_queue.lock_to_alerts()
        self._current_semantic_task = None
        if self._input_module and self._input_module.is_running:
            self._input_module.suspend()
        if self._vision_module and hasattr(self._vision_module, 'cancel_semantic_task'):
            self._vision_module.cancel_semantic_task()
        self._event_bus.clear()
        self._frame_buffer.clear()
        self._send_flutter_status()

    def _on_exit_alert(self):
        self._audio_queue.unlock_audio()
        self._alert_safe_since = None  # Reset safety timer
        if self._input_module and self._input_module.is_suspended:
            self._input_module.resume()
        if self._pending_face_prompt:
            pending = self._pending_face_prompt
            message_key = pending[0]
            priority = pending[1]
            session_id = pending[2]
            text = pending[3] if len(pending) > 3 else FaceConfig.PROMPTS.get(message_key, message_key)
            if text:
                self._last_face_prompt = (session_id, message_key)
                self._post_audio(AudioCommandType.SPEAK, text=text, priority=priority)
                if self._flutter_server:
                    self._flutter_server.send_face_event(
                        event_type='PROMPT',
                        message_key=message_key,
                        session_id=session_id,
                        text=text,
                    )
            self._pending_face_prompt = None

    def _on_enter_override(self):
        self._apply_vision_level()
        self._audio_queue.lock_to_alerts()
        if self._input_module and self._input_module.is_running:
            self._input_module.suspend()
        if self._input_module and hasattr(self._input_module, 'clear_buffer'):
            self._input_module.clear_buffer()
        self._event_bus.clear()
        self._send_flutter_status()

    def _on_exit_override(self):
        self._audio_queue.unlock_audio()
        if self._input_module and self._input_module.is_suspended:
            self._input_module.resume()

    def _on_enter_semantic(self):
        self._apply_vision_level()
        self._send_flutter_status()

    def _on_exit_semantic(self):
        self._current_semantic_task = None
        if self._vision_module and hasattr(self._vision_module, 'cancel_semantic_task'):
            self._vision_module.cancel_semantic_task()

    def _apply_vision_level(self):
        if self._vision_module is None:
            return
        level = self._state_machine.vision_level
        state_name = self._state_machine.state.name  # <-- Get the current state
        try:
            # Pass the state_name to show up in the brackets
            self._vision_module.set_vision_level(level, state_name)
        except Exception as e:
            logger.error(f"Failed to set vision level '{level}': {e}", exc_info=True)

    def _send_flutter_status(self):
        """Push the current system state to all connected Flutter clients."""
        if not self._flutter_server:
            return
        label = self._state_machine.state_label
        frame = self._frame_buffer.latest()
        age = frame.age_ms if frame else 0.0
        self._flutter_server.send_status(label, age)

    def _speak_alert(self, alert_key: str):
        self._audio_queue.post(AudioCommand(command_type=AudioCommandType.ALERT, alert_key=alert_key, priority=AudioConfig.PRIORITY_ALERT))

    def _post_audio(self, command_type: AudioCommandType, text: str=None, alert_key: str=None, priority: int=AudioConfig.PRIORITY_RESPONSE):
        self._audio_queue.post(AudioCommand(command_type=command_type, text=text, alert_key=alert_key, priority=priority))

    def _start_guest_modules(self):
        for module in self._guest_modules():
            if module is not None:
                success = module.start()
                if not success:
                    logger.error(f'Failed to start module: {module.module_name}')
        if self._sign_module is not None:
            self._sign_module.start()

    def _stop_guest_modules(self):
        for module in self._guest_modules():
            if module is not None:
                module.stop()
        if self._sign_module is not None:
            self._sign_module.stop()

    def _guest_modules(self):
        return [self._vision_module, self._audio_module, self._input_module, self._face_module]

    def _health_check(self):
        frame = self._frame_buffer.latest()
        age = frame.age_ms if frame else 999999
        if age > CameraConfig.STREAM_TIMEOUT_SEC * 1000:
            logger.warning(f'No frame received for {age:.0f}ms — stream may be lost')
        for module in self._guest_modules():
            if module is not None and (not module.is_healthy):
                logger.warning(f'Module {module.module_name} is not healthy: state={module.state.name}')
        summary = self._state_machine.summary()
        logger.debug(f"Health | state={summary['state']} | vision={summary['vision_level']} | in_state={summary['time_in_state']}s | audio_q={self._audio_queue.get_stats()['queue_size']} | events_pending={self._event_bus.pending_count()} | frame_age={age:.0f}ms")

    def _update_debug_display(self, frame: Optional[TimestampedFrame]):
        if not SystemConfig.DEBUG_DISPLAY:
            return
        try:
            import cv2
            import numpy as np
            display_frame = None
            if self._vision_module and hasattr(self._vision_module, 'get_debug_frame'):
                display_frame = self._vision_module.get_debug_frame()

            if display_frame is not None:
                # Vision module uses RGB, OpenCV uses BGR
                display = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            elif frame is not None:
                display = cv2.cvtColor(frame.frame, cv2.COLOR_RGB2BGR)
            else:
                display = np.zeros((CameraConfig.FRAME_HEIGHT, CameraConfig.FRAME_WIDTH, 3), dtype='uint8')

            state = self._state_machine.state.name
            vision = self._state_machine.vision_level
            age_ms = frame.age_ms if frame else 0
            locked = self._audio_queue.get_stats()['locked']
            cv2.rectangle(display, (0, 0), (400, 80), (0, 0, 0), -1)
            cv2.putText(display, f'STATE: {state}', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display, f'VISION: {vision}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, f'FRAME AGE: {age_ms:.0f}ms | AUDIO LOCKED: {locked}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            
            cv2.imshow(SystemConfig.DEBUG_WINDOW_NAME, display)
            if cv2.waitKey(1) & 255 == ord('q'):
                logger.info('Quit key pressed — stopping controller')
                self.stop()
        except Exception as e:
            logger.debug(f'Debug display error: {e}')

    @property
    def state(self) -> SystemState:
        return self._state_machine.state

    @property
    def is_running(self) -> bool:
        return self._running
    
    def send_flutter_alert(self, key: str):
        if self._flutter_server:
            self._flutter_server.send_alert(key)

    def send_sign_translation(self, sentence: str):
        logger.info(f"Broadcasting sign translation: '{sentence}'")
        if self._flutter_server:
            self._flutter_server.send_sign_translation(sentence)

    def set_flutter_server(self, flutter_server):
        self._flutter_server = flutter_server

    def set_app_mode(self, mode: str):
        mode = mode.lower()
        logger.info(
            f"Feature request received: {mode}"
        )

        # Reset sign language module state on mode changes to ensure we start fresh
        if self._sign_module and hasattr(self._sign_module, 'reset'):
            try:
                self._sign_module.reset()
            except Exception as e:
                logger.error(f"Failed to reset sign language module: {e}", exc_info=True)

        # Reset face module state when switching away from face mode
        if mode != "face" and self._face_module:
            try:
                self._face_module.cancel_registration()
                self._face_module.set_mode('idle')
            except Exception as e:
                logger.error(f"Failed to reset face module: {e}", exc_info=True)

        if mode == "danger":
            self.current_mode = AppMode.DANGER

        elif mode == "face":
            self.current_mode = AppMode.FACE

        elif mode == "sign":
            self.current_mode = AppMode.SIGN

        elif mode == "caption":
            self.current_mode = AppMode.CAPTION
            if self._state_machine.state == SystemState.IDLE:
                self._handle_request_caption()
            else:
                logger.info("Caption mode selected but state is not IDLE — ignoring caption request")

        logger.info(f"App mode changed to {self.current_mode.value}")
    