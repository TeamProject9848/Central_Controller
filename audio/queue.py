import threading
import logging
import queue
from typing import Optional
from config import AudioConfig
from core.event_bus import AudioCommand, AudioCommandType
from audio.backend import SpeechBackend, create_backend
logger = logging.getLogger(__name__)
ALERT_PHRASES: dict[str, str] = {'OBSTACLE_NEAR': 'Stop. Obstacle ahead.', 'OBSTACLE_MID': 'Caution. Object approaching.', 'PERSON_NEAR': 'Person very close.', 'VEHICLE_NEAR': 'Vehicle nearby. Stop.', 'STREAM_LOST': 'Camera disconnected.', 'STREAM_RECONNECTED': 'Camera reconnected.', 'OVERRIDE_ON': 'Walk mode on.', 'OVERRIDE_OFF': 'Walk mode off.', 'NAVIGATION_START': 'Navigation started.', 'NAVIGATION_STOP': 'Navigation stopped.', 'TASK_CANCELLED': 'Task cancelled.'}

class AudioQueue:

    def __init__(self, backend: Optional[SpeechBackend] = None):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._playback_thread: Optional[threading.Thread] = None
        self._running = False
        self._interrupt_flag = threading.Event()
        self._locked_to_alerts = False
        self._lock_mode_lock = threading.Lock()
        self._current_command: Optional[AudioCommand] = None
        self._tts_lock = threading.Lock()
        self._backend: SpeechBackend = backend or create_backend(rate=AudioConfig.TTS_SPEECH_RATE)
        self._played_count = 0
        self._dropped_count = 0
        self._interrupted_count = 0
        logger.info('AudioQueue initialized')

    def start(self):
        if self._running:
            logger.warning('AudioQueue already running')
            return
        self._running = True
        self._playback_thread = threading.Thread(target=self._playback_loop, name='AudioPlaybackThread', daemon=True)
        self._playback_thread.start()
        logger.info('AudioQueue playback thread started')

    def stop(self):
        logger.info('Stopping AudioQueue...')
        self._running = False
        self._interrupt_flag.set()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=3.0)
        self._shutdown_backend()
        logger.info('AudioQueue stopped')

    def post(self, command: AudioCommand):
        with self._lock_mode_lock:
            locked = self._locked_to_alerts
        if locked and command.priority > AudioConfig.PRIORITY_ALERT:
            self._dropped_count += 1
            logger.debug(f'AudioCommand dropped (locked to alerts) | type={command.command_type.name} | priority={command.priority}')
            return
        if command.command_type == AudioCommandType.STOP:
            self._interrupt_flag.set()
            self._stop_backend()
            logger.debug('STOP command received - interrupting current audio')
            return
        if command.command_type == AudioCommandType.CLEAR:
            self._clear_queue()
            self._interrupt_flag.set()
            logger.debug('CLEAR command received - queue flushed')
            return
        if AudioConfig.INTERRUPT_ON_HIGHER_PRIORITY and self._current_command is not None and (command.priority < self._current_command.priority):
            self._interrupt_flag.set()
            self._interrupted_count += 1
            logger.info(f'Interrupting P{self._current_command.priority} audio for P{command.priority} command')
        with self._sequence_lock:
            seq = self._sequence
            self._sequence += 1
        self._queue.put((command.priority, seq, command))
        logger.debug(f'AudioCommand queued | type={command.command_type.name} | priority=P{command.priority} | seq={seq}')

    def lock_to_alerts(self):
        with self._lock_mode_lock:
            self._locked_to_alerts = True
        self._flush_non_alerts()
        if self._current_command is not None and self._current_command.priority > AudioConfig.PRIORITY_ALERT:
            self._interrupt_flag.set()
        logger.info('AudioQueue locked to ALERT-only mode')

    def unlock_audio(self):
        with self._lock_mode_lock:
            self._locked_to_alerts = False
        logger.info('AudioQueue unlocked - all priorities accepted')

    def get_stats(self) -> dict:
        return {'played': self._played_count, 'dropped': self._dropped_count, 'interrupted': self._interrupted_count, 'queue_size': self._queue.qsize(), 'locked': self._locked_to_alerts, 'playing': self._current_command is not None}

    def _playback_loop(self):
        logger.info('Playback loop started')
        while self._running:
            try:
                priority, seq, command = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._interrupt_flag.clear()
            self._current_command = command
            try:
                self._play_command(command)
                self._played_count += 1
            except Exception as e:
                logger.error(f'Playback error for command {command.command_type.name}: {e}', exc_info=True)
            finally:
                self._current_command = None
        logger.info('Playback loop exited')

    def _play_command(self, command: AudioCommand):
        if command.command_type == AudioCommandType.ALERT:
            phrase = ALERT_PHRASES.get(command.alert_key, command.alert_key or 'Alert.')
            logger.info(f"ALERT: '{phrase}'")
            self._speak(phrase)
        elif command.command_type == AudioCommandType.SPEAK:
            if command.text:
                text = command.text[:AudioConfig.TTS_MAX_CHARS]
                if len(command.text) > AudioConfig.TTS_MAX_CHARS:
                    logger.debug(f'TTS text truncated to {AudioConfig.TTS_MAX_CHARS} chars')
                logger.info(f"TTS P{command.priority}: '{text[:60]}...' " if len(text) > 60 else f"TTS P{command.priority}: '{text}'")
                self._speak(text)
            else:
                logger.warning('SPEAK command had no text')

    def _speak(self, text: str):
        if self._interrupt_flag.is_set():
            logger.debug('Speech skipped - interrupt flag set before playback')
            return
        try:
            with self._tts_lock:
                if self._interrupt_flag.is_set():
                    return
                self._backend.speak(text, self._interrupt_flag, AudioConfig.TTS_SPEECH_RATE)
        except Exception as e:
            logger.error(f'TTS error: {e}', exc_info=True)

    def _stop_backend(self):
        try:
            self._backend.stop()
        except Exception:
            pass

    def _shutdown_backend(self):
        try:
            self._backend.shutdown()
        except Exception:
            pass

    def _clear_queue(self):
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared > 0:
            logger.info(f'AudioQueue flushed - {cleared} commands dropped')

    def _flush_non_alerts(self):
        kept = []
        dropped = 0
        while not self._queue.empty():
            try:
                priority, seq, command = self._queue.get_nowait()
                if priority == AudioConfig.PRIORITY_ALERT:
                    kept.append((priority, seq, command))
                else:
                    dropped += 1
            except queue.Empty:
                break
        for item in kept:
            self._queue.put(item)
        if dropped > 0:
            logger.info(f'Flushed {dropped} non-alert commands on lock')
