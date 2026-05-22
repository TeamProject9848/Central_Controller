import threading
import logging
import time
import sys
from typing import Optional
from config import CameraConfig, SystemConfig, AudioConfig
from core.event_bus import EventBus, VisionEvent, VisionEventType, StreamEvent, StreamEventType, IntentEvent, IntentEventType, AudioCommand, AudioCommandType
from core.state_machine import StateMachine, SystemState
from camera.source import CameraSource
from camera.buffer import FrameBuffer, TimestampedFrame
from audio.queue import AudioQueue
from interfaces.base import BaseModule
logger = logging.getLogger(__name__)

class CentralController:

    def __init__(self):
        self._event_bus = EventBus()
        self._state_machine = StateMachine(initial_state=SystemState.IDLE)
        self._frame_buffer = FrameBuffer()
        self._camera_source = CameraSource(self._event_bus, self._frame_buffer)
        self._audio_queue = AudioQueue()
        self._vision_module = None
        self._audio_module = None
        self._input_module = None
        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        self._clear_frame_count = 0
        self._required_clear_frames = 10
        self._last_risk_time: float = 0.0
        self._pre_alert_state: Optional[SystemState] = None
        self._register_event_handlers()
        self._register_state_hooks()
        logger.info('CentralController initialized')

    def register_vision_module(self, module):
        self._vision_module = module
        module.set_event_callback(self._event_bus.post)
        
        # Route semantic results (caption/OCR) to AudioQueue as SPEAK commands
        module.set_semantic_result_callback(
            lambda text: self._post_audio(
                AudioCommandType.SPEAK,
                text=text,
                priority=AudioConfig.PRIORITY_RESPONSE
            )
        )
        
        logger.info(f"Vision module registered: {module.module_name}")

    def register_audio_module(self, module):
        self._audio_module = module
        logger.info(f'Audio module registered: {module.module_name}')

    def register_input_module(self, module):
        self._input_module = module
        module.set_event_callback(self._event_bus.post)
        logger.info(f'Input module registered: {module.module_name}')

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
        connected = self._camera_source.start()
        if not connected:
            logger.warning('Camera stream not available at startup — will retry automatically')
            self._speak_alert('STREAM_LOST')
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
            sleep_time = SystemConfig.CONTROLLER_TICK_SEC - elapsed
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
        logger.debug('Event handlers registered')

    def _on_risk_event(self, event: VisionEvent):
        now = time.time()
        if now - self._last_risk_time < AudioConfig.ALERT_COOLDOWN_SEC:
            logger.debug('RISK event within cooldown window — suppressing duplicate')
            return
        self._last_risk_time = now
        self._clear_frame_count = 0
        logger.warning(f'RISK INTERRUPT | class={event.hazard_class} | confidence={event.confidence:.2f} | depth={event.depth_zone}')
        current = self._state_machine.state
        if current != SystemState.ALERT:
            self._pre_alert_state = current
        if self._state_machine.can_transition_to(SystemState.ALERT):
            self._state_machine.transition(SystemState.ALERT, reason=f'RISK: {event.hazard_class} @ {event.confidence:.2f}')
        alert_key = self._select_alert_key(event)
        self._speak_alert(alert_key)

    def _on_vision_event(self, event: VisionEvent):
        if event.event_type == VisionEventType.MOTION:
            logger.debug(f'MOTION detected | confidence={event.confidence:.2f}')
        elif event.event_type == VisionEventType.NONE:
            if self._state_machine.state == SystemState.ALERT:
                self._clear_frame_count += 1
                logger.debug(f'Clear frame {self._clear_frame_count}/{self._required_clear_frames}')
                if self._clear_frame_count >= self._required_clear_frames:
                    self._resolve_alert()

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
        handlers = {IntentEventType.START_NAVIGATION: self._handle_start_navigation, IntentEventType.STOP_NAVIGATION: self._handle_stop_navigation, IntentEventType.REQUEST_CAPTION: self._handle_request_caption, IntentEventType.REQUEST_OCR: self._handle_request_ocr, IntentEventType.UNKNOWN: self._handle_unknown_intent}
        handler = handlers.get(event.event_type)
        if handler:
            try:
                handler()
            except Exception as e:
                logger.error(f'Intent handler for {event.event_type.name} raised: {e}', exc_info=True)
        else:
            logger.warning(f'No handler for intent: {event.event_type.name}')

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
                self._state_machine.transition(SystemState.SEMANTIC, reason='User requested caption')
                self._vision_module.request_caption()
            else:
                logger.warning('Caption requested but no vision module registered')
        else:
            logger.debug('Caption request ignored — cannot enter SEMANTIC state now')

    def _handle_request_ocr(self):
        if self._state_machine.can_transition_to(SystemState.SEMANTIC):
            if self._vision_module:
                self._state_machine.transition(SystemState.SEMANTIC, reason='User requested OCR')
                self._vision_module.request_ocr()
            else:
                logger.warning('OCR requested but no vision module registered')
        else:
            logger.debug('OCR request ignored — cannot enter SEMANTIC state now')

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
        self._clear_frame_count = 0
        target = self._pre_alert_state or SystemState.NAVIGATION
        logger.info(f'Alert resolved — returning to {target.name}')
        if self._state_machine.can_transition_to(target):
            self._state_machine.transition(target, reason=f'Hazard cleared after {self._required_clear_frames} clear frames')
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

    def _on_exit_idle(self):
        pass

    def _on_enter_navigation(self):
        self._apply_vision_level()
        self._audio_queue.unlock_audio()
        if self._input_module and self._input_module.is_suspended:
            self._input_module.resume()

    def _on_exit_navigation(self):
        pass

    def _on_enter_alert(self):
        self._apply_vision_level()
        self._audio_queue.lock_to_alerts()
        if self._input_module and self._input_module.is_running:
            self._input_module.suspend()
        if self._vision_module and hasattr(self._vision_module, 'cancel_semantic_task'):
            self._vision_module.cancel_semantic_task()
        self._event_bus.clear()
        self._frame_buffer.clear()

    def _on_exit_alert(self):
        self._audio_queue.unlock_audio()
        if self._input_module and self._input_module.is_suspended:
            self._input_module.resume()

    def _on_enter_override(self):
        self._apply_vision_level()
        self._audio_queue.lock_to_alerts()
        if self._input_module and self._input_module.is_running:
            self._input_module.suspend()
        if self._input_module and hasattr(self._input_module, 'clear_buffer'):
            self._input_module.clear_buffer()
        self._event_bus.clear()

    def _on_exit_override(self):
        self._audio_queue.unlock_audio()
        if self._input_module and self._input_module.is_suspended:
            self._input_module.resume()

    def _on_enter_semantic(self):
        self._apply_vision_level()

    def _on_exit_semantic(self):
        if self._vision_module and hasattr(self._vision_module, 'cancel_semantic_task'):
            self._vision_module.cancel_semantic_task()

    def _apply_vision_level(self):
        if self._vision_module is None:
            return
        level = self._state_machine.vision_level
        try:
            self._vision_module.set_vision_level(level)
        except Exception as e:
            logger.error(f"Failed to set vision level '{level}': {e}", exc_info=True)

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

    def _stop_guest_modules(self):
        for module in self._guest_modules():
            if module is not None:
                module.stop()

    def _guest_modules(self):
        return [self._vision_module, self._audio_module, self._input_module]

    def _health_check(self):
        age = self._camera_source.last_frame_age_ms
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