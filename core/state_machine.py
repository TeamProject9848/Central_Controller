import logging
import threading
from enum import Enum, auto
from typing import Callable, Optional
from time import time
logger = logging.getLogger(__name__)

class SystemState(Enum):
    IDLE = auto()
    NAVIGATION = auto()
    ALERT = auto()
    ACTIVE_WALK_OVERRIDE = auto()
    SEMANTIC = auto()
ALLOWED_TRANSITIONS: dict[SystemState, set[SystemState]] = {SystemState.IDLE: {SystemState.NAVIGATION, SystemState.SEMANTIC, SystemState.ALERT}, SystemState.NAVIGATION: {SystemState.IDLE, SystemState.ALERT, SystemState.ACTIVE_WALK_OVERRIDE}, SystemState.ALERT: {SystemState.NAVIGATION, SystemState.IDLE, SystemState.ACTIVE_WALK_OVERRIDE}, SystemState.ACTIVE_WALK_OVERRIDE: {SystemState.NAVIGATION, SystemState.ALERT}, SystemState.SEMANTIC: {SystemState.IDLE, SystemState.ALERT}}
STATE_METADATA: dict[SystemState, dict] = {SystemState.IDLE: {'label': 'Idle', 'description': 'System ready. Sentinel monitoring active.', 'vision_level': 'sentinel_only', 'audio_allowed': ['alert', 'response', 'description', 'status'], 'is_safety_state': False}, SystemState.NAVIGATION: {'label': 'Navigation', 'description': 'Walking mode. Safety vision active.', 'vision_level': 'sentinel_and_safety', 'audio_allowed': ['alert', 'status'], 'is_safety_state': True}, SystemState.ALERT: {'label': 'ALERT', 'description': 'Hazard detected. All attention on safety.', 'vision_level': 'sentinel_and_safety', 'audio_allowed': ['alert'], 'is_safety_state': True}, SystemState.ACTIVE_WALK_OVERRIDE: {'label': 'Walk Override', 'description': 'Maximum safety mode active.', 'vision_level': 'sentinel_and_safety_max', 'audio_allowed': ['alert'], 'is_safety_state': True}, SystemState.SEMANTIC: {'label': 'Semantic Task', 'description': 'Processing visual task. Sentinel monitoring.', 'vision_level': 'sentinel_and_semantic', 'audio_allowed': ['alert', 'response'], 'is_safety_state': False}}

class StateMachine:

    def __init__(self, initial_state: SystemState=SystemState.IDLE):
        self._state: SystemState = initial_state
        self._lock = threading.RLock()
        self._state_entered_at: float = time()
        self._history: list[tuple] = []
        self._on_enter_callbacks: dict[SystemState, list[Callable]] = {s: [] for s in SystemState}
        self._on_exit_callbacks: dict[SystemState, list[Callable]] = {s: [] for s in SystemState}
        logger.info(f'StateMachine initialized in state: {initial_state.name}')

    @property
    def state(self) -> SystemState:
        with self._lock:
            return self._state

    @property
    def state_label(self) -> str:
        return STATE_METADATA[self.state]['label']

    @property
    def is_safety_state(self) -> bool:
        return STATE_METADATA[self.state]['is_safety_state']

    @property
    def vision_level(self) -> str:
        return STATE_METADATA[self.state]['vision_level']

    @property
    def allowed_audio(self) -> list[str]:
        return STATE_METADATA[self.state]['audio_allowed']

    @property
    def time_in_state(self) -> float:
        return time() - self._state_entered_at

    def on_enter(self, state: SystemState, callback: Callable[[], None]):
        self._on_enter_callbacks[state].append(callback)
        logger.debug(f'on_enter registered for {state.name}: {callback.__name__}')

    def on_exit(self, state: SystemState, callback: Callable[[], None]):
        self._on_exit_callbacks[state].append(callback)
        logger.debug(f'on_exit registered for {state.name}: {callback.__name__}')

    def transition(self, new_state: SystemState, reason: str='') -> bool:
        with self._lock:
            current = self._state
            if new_state == current:
                logger.debug(f'Transition to same state {new_state.name} — ignored')
                return True
            allowed = ALLOWED_TRANSITIONS.get(current, set())
            if new_state not in allowed:
                logger.error(f"ILLEGAL TRANSITION: {current.name} → {new_state.name} | reason='{reason}' | allowed={[s.name for s in allowed]}")
                raise ValueError(f'Illegal state transition: {current.name} → {new_state.name}. Allowed from {current.name}: {[s.name for s in allowed]}')
            self._fire_callbacks(self._on_exit_callbacks[current], f'on_exit:{current.name}')
            prev_state = self._state
            self._state = new_state
            self._state_entered_at = time()
            self._history.append((prev_state, new_state, time(), reason))
            logger.info(f'STATE: {prev_state.name} → {new_state.name}')
            logger.info(f"STATE: {prev_state.name} → {new_state.name} | reason='{reason}' | vision={self.vision_level}")
            self._fire_callbacks(self._on_enter_callbacks[new_state], f'on_enter:{new_state.name}')
        return True

    def _fire_callbacks(self, callbacks: list[Callable], context: str):
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                logger.error(f'Callback crash in {context} → {cb.__name__}: {e}', exc_info=True)

    def can_transition_to(self, target: SystemState) -> bool:
        with self._lock:
            return target in ALLOWED_TRANSITIONS.get(self._state, set())

    def is_in(self, *states: SystemState) -> bool:
        return self.state in states

    def get_history(self, last_n: int=10) -> list[tuple]:
        return self._history[-last_n:]

    def summary(self) -> dict:
        return {'state': self.state.name, 'label': self.state_label, 'vision_level': self.vision_level, 'allowed_audio': self.allowed_audio, 'is_safety': self.is_safety_state, 'time_in_state': round(self.time_in_state, 2), 'transitions': len(self._history)}