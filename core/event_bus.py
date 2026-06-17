import queue
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Dict, Any
from time import time
from config import SystemConfig
logger = logging.getLogger(__name__)

class VisionEventType(Enum):
    NONE = auto()
    MOTION = auto()
    RISK = auto()
    PERSON_LEFT = auto()

class StreamEventType(Enum):
    CONNECTED = auto()
    LOST = auto()
    RECONNECTING = auto()

class IntentEventType(Enum):
    START_NAVIGATION = auto()
    STOP_NAVIGATION = auto()
    REQUEST_CAPTION = auto()
    REQUEST_OCR = auto()
    TOGGLE_OVERRIDE = auto()
    START_FACE_REGISTRATION = auto()
    CANCEL_FACE_REGISTRATION = auto()
    IDENTIFY_FACE = auto()
    UNKNOWN = auto()

class AudioCommandType(Enum):
    SPEAK = auto()
    ALERT = auto()
    STOP = auto()
    CLEAR = auto()


class FaceEventType(Enum):
    PROMPT = auto()
    IDENTIFIED = auto()
    REGISTRATION_PROGRESS = auto()
    REGISTRATION_COMPLETE = auto()
    REGISTRATION_FAILED = auto()

@dataclass
class VisionEvent:
    event_type: VisionEventType
    confidence: float = 0.0
    hazard_class: Optional[str] = None
    depth_zone: Optional[str] = None
    tracker_id: Optional[int] = None
    source: Optional[str] = None        # "sentinel" or "safety" — for controller to distinguish
    timestamp: float = field(default_factory=time)

    def is_stale(self, max_age_ms: float) -> bool:
        age_ms = (time() - self.timestamp) * 1000
        return age_ms > max_age_ms

@dataclass
class StreamEvent:
    event_type: StreamEventType
    reason: Optional[str] = None
    timestamp: float = field(default_factory=time)

@dataclass
class IntentEvent:
    event_type: IntentEventType
    raw_input: Optional[str] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time)

@dataclass
class AudioCommand:
    command_type: AudioCommandType
    text: Optional[str] = None
    alert_key: Optional[str] = None
    priority: int = 2
    timestamp: float = field(default_factory=time)


@dataclass
class FaceEvent:
    event_type: FaceEventType
    message_key: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    priority: int = 5
    timestamp: float = field(default_factory=time)


EVENT_PRIORITY = {
    VisionEventType.RISK: 0,
    StreamEventType.LOST: 1,
    VisionEventType.MOTION: 2,
    VisionEventType.PERSON_LEFT: 2,
    IntentEventType.TOGGLE_OVERRIDE: 3,
    IntentEventType.START_NAVIGATION: 4,
    IntentEventType.STOP_NAVIGATION: 4,
    StreamEventType.RECONNECTING: 5,
    StreamEventType.CONNECTED: 5,
    IntentEventType.START_FACE_REGISTRATION: 6,
    IntentEventType.CANCEL_FACE_REGISTRATION: 6,
    IntentEventType.IDENTIFY_FACE: 6,
    FaceEventType.REGISTRATION_FAILED: 5,
    FaceEventType.PROMPT: 6,
    FaceEventType.IDENTIFIED: 6,
    FaceEventType.REGISTRATION_PROGRESS: 6,
    FaceEventType.REGISTRATION_COMPLETE: 6,
    IntentEventType.REQUEST_CAPTION: 6,
    IntentEventType.REQUEST_OCR: 6,
    IntentEventType.UNKNOWN: 7,
    VisionEventType.NONE: 8,
}

class EventBus:

    def __init__(self):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._sequence: int = 0
        self._sequence_lock = threading.Lock()
        self._risk_callback: Optional[Callable[[VisionEvent], None]] = None
        self._handlers: dict[type, Callable] = {}
        logger.debug('EventBus initialized')

    def register_risk_callback(self, callback: Callable[[VisionEvent], None]):
        self._risk_callback = callback
        logger.debug('Risk callback registered')

    def register_handler(self, event_type: type, handler: Callable):
        self._handlers[event_type] = handler
        logger.debug(f'Handler registered for {event_type.__name__}')

    def post(self, event):
        if isinstance(event, VisionEvent):
            if event.event_type != VisionEventType.RISK:
                zone = event.depth_zone if event.depth_zone is not None else 'N/A'
                # Moved from logger.info to logger.debug
                logger.debug(f'VisionEvent: {event.event_type.name} (confidence={event.confidence:.2f}, zone={zone})')
        
        # Keep RISK events highly visible
        if isinstance(event, VisionEvent) and event.event_type == VisionEventType.RISK:
            logger.warning(f'RISK EVENT — direct callback firing | class={event.hazard_class} | confidence={event.confidence:.2f} | depth={event.depth_zone}')
            if self._risk_callback:
                self._risk_callback(event)
            else:
                logger.error('RISK event posted but no risk callback registered! Falling back to queue. Register callback before starting.')
                self._enqueue(event, priority=0)
            return
            
        event_subtype = getattr(event, 'event_type', None)
        priority = EVENT_PRIORITY.get(event_subtype, 5)
        self._enqueue(event, priority)

    def _enqueue(self, event, priority: int):
        with self._sequence_lock:
            seq = self._sequence
            self._sequence += 1
        self._queue.put((priority, seq, event))
        logger.debug(f'Queued {type(event).__name__}({event.event_type}) priority={priority} seq={seq}')

    def process_events(self, max_events: int=10):
        processed = 0
        while processed < max_events:
            try:
                priority, seq, event = self._queue.get_nowait()
            except queue.Empty:
                break
            if hasattr(event, 'is_stale'):
                if event.is_stale(SystemConfig.MAX_FRAME_AGE_MS):
                    logger.warning(f'Discarding stale event: {type(event).__name__} ({event.event_type})')
                    processed += 1
                    continue
            handler = self._handlers.get(type(event))
            if handler:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f'Handler for {type(event).__name__} raised: {e}', exc_info=True)
            else:
                logger.warning(f'No handler for event type: {type(event).__name__} ({event.event_type}) — register one in controller')
            processed += 1
        return processed

    def clear(self):
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        logger.info(f'EventBus cleared — {cleared} events discarded')

    def pending_count(self) -> int:
        return self._queue.qsize()