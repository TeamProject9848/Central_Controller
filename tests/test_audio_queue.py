import time
from audio.queue import AudioQueue, ALERT_PHRASES
from audio.backend import SpeechBackend
from core.event_bus import AudioCommand, AudioCommandType
from config import AudioConfig


class FakeBackend(SpeechBackend):
    """Test double that records speech calls and honors interrupts."""

    def __init__(self, speak_duration: float = 0.15):
        self.spoken = []
        self.interrupted = []
        self.stop_calls = 0
        self._speak_duration = speak_duration

    def speak(self, text: str, interrupt_event, rate: int) -> None:
        self.spoken.append(text)
        start = time.time()
        while time.time() - start < self._speak_duration:
            if interrupt_event.is_set():
                self.interrupted.append(text)
                return
            time.sleep(0.01)

    def stop(self) -> None:
        self.stop_calls += 1

    def shutdown(self) -> None:
        self.stop()


def _wait_for_queue_flush(delay: float = 0.35):
    """Small helper to allow the playback thread to drain the queue."""
    time.sleep(delay)


def test_priority_queue_respects_alert_first():
    backend = FakeBackend(speak_duration=0.05)
    audio_queue = AudioQueue(backend=backend)
    audio_queue.start()

    # Lower priority speech posted before alert.
    audio_queue.post(AudioCommand(command_type=AudioCommandType.SPEAK, text='low', priority=AudioConfig.PRIORITY_RESPONSE))
    audio_queue.post(AudioCommand(command_type=AudioCommandType.ALERT, alert_key='STREAM_LOST', priority=AudioConfig.PRIORITY_ALERT))

    _wait_for_queue_flush()
    audio_queue.stop()

    assert backend.spoken[0] == ALERT_PHRASES['STREAM_LOST']
    assert 'low' in backend.spoken


def test_high_priority_interrupts_lower_playback():
    backend = FakeBackend(speak_duration=0.3)
    audio_queue = AudioQueue(backend=backend)
    audio_queue.start()

    audio_queue.post(AudioCommand(command_type=AudioCommandType.SPEAK, text='long running', priority=AudioConfig.PRIORITY_DESCRIPTION))
    time.sleep(0.05)  # Allow playback to start
    audio_queue.post(AudioCommand(command_type=AudioCommandType.ALERT, alert_key='OBSTACLE_NEAR', priority=AudioConfig.PRIORITY_ALERT))

    _wait_for_queue_flush()
    audio_queue.stop()

    assert ALERT_PHRASES['OBSTACLE_NEAR'] in backend.spoken
    assert 'long running' in backend.interrupted  # should have been preempted


def test_lock_to_alerts_drops_non_alert_speech():
    backend = FakeBackend(speak_duration=0.05)
    audio_queue = AudioQueue(backend=backend)
    audio_queue.start()

    audio_queue.lock_to_alerts()
    audio_queue.post(AudioCommand(command_type=AudioCommandType.SPEAK, text='should drop', priority=AudioConfig.PRIORITY_RESPONSE))
    audio_queue.post(AudioCommand(command_type=AudioCommandType.ALERT, alert_key='NAVIGATION_START', priority=AudioConfig.PRIORITY_ALERT))

    _wait_for_queue_flush()
    audio_queue.stop()

    assert backend.spoken == [ALERT_PHRASES['NAVIGATION_START']]
    assert 'should drop' not in backend.spoken
