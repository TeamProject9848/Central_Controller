import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechBackend(ABC):
    """Abstract backend for speech synthesis/playback."""

    @abstractmethod
    def speak(self, text: str, interrupt_event: threading.Event, rate: int) -> None:
        """Play the given text. Respect interrupt_event for preemption."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop any ongoing playback immediately."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release backend resources."""
        ...


class ConsoleBackend(SpeechBackend):
    """Fallback backend that prints text to stdout."""

    def speak(self, text: str, interrupt_event: threading.Event, rate: int) -> None:
        if interrupt_event.is_set():
            return
        print(f'[AUDIO] {text}')

    def stop(self) -> None:
        # Nothing to stop for console output.
        return

    def shutdown(self) -> None:
        return


class Pyttsx3Backend(SpeechBackend):
    """pyttsx3-based backend kept as the current active engine."""

    def __init__(self, rate: int):
        import pyttsx3  # Delayed import to keep dependency optional

        self._engine = pyttsx3.init()
        self._rate = rate
        self._engine.setProperty('rate', rate)

    def speak(self, text: str, interrupt_event: threading.Event, rate: int) -> None:
        if interrupt_event.is_set():
            return
        try:
            if rate != self._rate:
                self._engine.setProperty('rate', rate)
                self._rate = rate
            self._engine.say(text)
            self._engine.startLoop(False)
            while self._engine.isBusy():
                if interrupt_event.is_set():
                    self._engine.stop()
                    logger.debug('pyttsx3 backend interrupted mid-speech')
                    break
                time.sleep(0.05)
            self._engine.endLoop()
        except Exception as exc:
            logger.error(f'pyttsx3 playback error: {exc}', exc_info=True)

    def stop(self) -> None:
        try:
            self._engine.stop()
        except Exception:
            # Stop should be best-effort; swallow backend errors.
            return

    def shutdown(self) -> None:
        self.stop()


def create_backend(rate: int, prefer_pyttsx3: bool = True) -> SpeechBackend:
    """
    Create the default speech backend.
    Keeps pyttsx3 as the active engine, falling back to console output if unavailable.
    """
    if prefer_pyttsx3:
        try:
            return Pyttsx3Backend(rate=rate)
        except ImportError:
            logger.warning('pyttsx3 not installed - falling back to console backend.')
        except Exception as exc:
            logger.warning(f'pyttsx3 backend init failed ({exc}) - falling back to console backend.')
    return ConsoleBackend()
