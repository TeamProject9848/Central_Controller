import cv2
import threading
import logging
import time
from typing import Optional, Callable
from config import CameraConfig, SystemConfig
from core.event_bus import EventBus, StreamEvent, StreamEventType
logger = logging.getLogger(__name__)

class CameraSource:

    def __init__(self, event_bus: EventBus, frame_buffer):
        self._event_bus = event_bus
        self._frame_buffer = frame_buffer
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0.0
        self._fail_counter = 0
        self._max_consecutive_fails = 15
        self._last_frame_time: float = 0.0
        self._stream_url = CameraConfig.STREAM_URL
        logger.info(f'CameraSource initialized | url={self._stream_url}')

    def start(self) -> bool:
        if self._running:
            logger.warning('CameraSource.start() called but already running')
            return True
        logger.info('Starting CameraSource...')
        self._running = True
        self._reconnect_attempts = 0
        connected = self._open_stream()
        if connected:
            self._emit_stream_event(StreamEventType.CONNECTED, 'Initial connection')
        else:
            logger.warning('Initial stream connection failed — capture thread will retry')
            self._emit_stream_event(StreamEventType.LOST, 'Initial connection failed')
        self._capture_thread = threading.Thread(target=self._capture_loop, name='CaptureThread', daemon=True)
        self._capture_thread.start()
        logger.info('Capture thread started')
        return connected

    def stop(self):
        logger.info('Stopping CameraSource...')
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                logger.warning('Capture thread did not stop within timeout')
        self._release_stream()
        logger.info('CameraSource stopped')

    def reconnect(self) -> bool:
        now = time.time()
        elapsed = now - self._last_reconnect_time
        if elapsed < CameraConfig.RECONNECT_INTERVAL_SEC:
            wait = CameraConfig.RECONNECT_INTERVAL_SEC - elapsed
            logger.debug(f'Reconnect called too soon — waiting {wait:.1f}s')
            time.sleep(wait)
        if self._reconnect_attempts >= CameraConfig.MAX_RECONNECT_ATTEMPTS:
            logger.error(f'Max reconnect attempts ({CameraConfig.MAX_RECONNECT_ATTEMPTS}) reached — giving up. Check phone IP and WiFi connection.')
            self._emit_stream_event(StreamEventType.LOST, f'Max reconnect attempts ({CameraConfig.MAX_RECONNECT_ATTEMPTS}) exceeded')
            return False
        self._reconnect_attempts += 1
        self._last_reconnect_time = time.time()
        logger.info(f'Reconnect attempt {self._reconnect_attempts}/{CameraConfig.MAX_RECONNECT_ATTEMPTS}')
        self._emit_stream_event(StreamEventType.RECONNECTING, f'Attempt {self._reconnect_attempts}')
        self._release_stream()
        success = self._open_stream()
        if success:
            self._reconnect_attempts = 0
            self._fail_counter = 0
            self._emit_stream_event(StreamEventType.CONNECTED, f'Reconnected after {self._reconnect_attempts} attempts')
            logger.info('Reconnection successful')
        else:
            logger.warning(f'Reconnect attempt {self._reconnect_attempts} failed')
        return success

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_frame_age_ms(self) -> float:
        if self._last_frame_time == 0.0:
            return 0.0
        return (time.time() - self._last_frame_time) * 1000

    def get_stream_info(self) -> dict:
        return {'url': self._stream_url, 'connected': self._connected, 'reconnect_attempts': self._reconnect_attempts, 'fail_counter': self._fail_counter, 'last_frame_age_ms': round(self.last_frame_age_ms, 1), 'thread_alive': self._capture_thread.is_alive() if self._capture_thread else False}

    def _open_stream(self) -> bool:
        with self._capture_lock:
            try:
                logger.info(f'Opening stream: {self._stream_url}')
                cap = cv2.VideoCapture(self._stream_url)
                time.sleep(0.5)
                if not cap.isOpened():
                    logger.warning(f'cv2.VideoCapture failed to open: {self._stream_url}')
                    cap.release()
                    self._connected = False
                    return False
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.FRAME_HEIGHT)
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning('Stream opened but test read failed')
                    cap.release()
                    self._connected = False
                    return False
                self._capture = cap
                self._connected = True
                logger.info(f'Stream opened successfully | resolution={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
                return True
            except Exception as e:
                logger.error(f'Exception opening stream: {e}', exc_info=True)
                self._connected = False
                return False

    def _release_stream(self):
        with self._capture_lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                    logger.debug('Stream released')
                except Exception as e:
                    logger.warning(f'Error releasing stream: {e}')
                finally:
                    self._capture = None
                    self._connected = False

    def _capture_loop(self):
        logger.info('Capture loop started')
        while self._running:
            if not self._connected:
                logger.debug('Not connected — waiting before retry...')
                time.sleep(CameraConfig.RECONNECT_INTERVAL_SEC)
                if self._running:
                    self.reconnect()
                continue
            with self._capture_lock:
                if self._capture is None:
                    self._connected = False
                    continue
                ret, raw_frame = self._capture.read()
            if not ret or raw_frame is None:
                self._fail_counter += 1
                logger.debug(f'Frame read failed ({self._fail_counter}/{self._max_consecutive_fails})')
                if self._fail_counter >= self._max_consecutive_fails:
                    logger.warning(f'Stream lost after {self._fail_counter} consecutive failed reads')
                    self._connected = False
                    self._emit_stream_event(StreamEventType.LOST, f'{self._fail_counter} consecutive read failures')
                    self._fail_counter = 0
                time.sleep(0.05)
                continue
            self._fail_counter = 0
            self._last_frame_time = time.time()
            normalized = self._normalize_frame(raw_frame)
            if normalized is not None:
                self._frame_buffer.push(normalized)
        logger.info('Capture loop exited')

    def _normalize_frame(self, frame):
        try:
            h, w = frame.shape[:2]
            if w != CameraConfig.FRAME_WIDTH or h != CameraConfig.FRAME_HEIGHT:
                frame = cv2.resize(frame, (CameraConfig.FRAME_WIDTH, CameraConfig.FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except Exception as e:
            logger.error(f'Frame normalization failed: {e}', exc_info=True)
            return None

    def _emit_stream_event(self, event_type: StreamEventType, reason: str=''):
        try:
            event = StreamEvent(event_type=event_type, reason=reason)
            self._event_bus.post(event)
            logger.debug(f'StreamEvent posted: {event_type.name} | {reason}')
        except Exception as e:
            logger.error(f'Failed to post StreamEvent: {e}', exc_info=True)