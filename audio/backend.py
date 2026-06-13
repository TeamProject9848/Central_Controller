import os
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional
import tempfile
import subprocess
import soundfile as sf
from kokoro import KPipeline

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
        # Delay importing/initializing the engine until first use to avoid
        # audio-driver initialization issues in the creating thread.
        self._engine = None
        self._rate = rate
        self._init_lock = threading.Lock()

    def _ensure_engine(self):
        if self._engine is not None:
            return
        with self._init_lock:
            if self._engine is not None:
                return
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', self._rate)
                logger.info('pyttsx3 engine initialized')
            except Exception as exc:
                logger.error(f'Failed to initialize pyttsx3 engine: {exc}', exc_info=True)
                raise

    def speak(self, text: str, interrupt_event: threading.Event, rate: int) -> None:
        if interrupt_event.is_set():
            return
        try:
            self._ensure_engine()
            if rate != self._rate:
                try:
                    self._engine.setProperty('rate', rate)
                    self._rate = rate
                except Exception:
                    logger.debug('Unable to set pyttsx3 rate — continuing with current rate')

            self._engine.say(text)
            self._engine.runAndWait()

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

class KokoroBackend(SpeechBackend):

    def __init__(self, rate: int):
        self._pipeline = KPipeline(lang_code="a")
        self._voice = "af_heart"

    def speak(self, text: str, interrupt_event: threading.Event, rate: int) -> None:

        if interrupt_event.is_set():
            return

        try:
            generator = self._pipeline(
                text,
                voice=self._voice
            )

            for _, _, audio in generator:

                if interrupt_event.is_set():
                    return

                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False
                ) as f:

                    sf.write(f.name, audio, 24000)
                    
                    path = f.name

                    subprocess.run(
                        ["aplay", f.name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                try:
                    os.remove(path)
                except Exception:
                    pass

        except Exception as exc:
            logger.error(
                f'Kokoro playback error: {exc}',
                exc_info=True
            )

    def stop(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


def create_backend(rate: int, prefer_pyttsx3: bool = True) -> SpeechBackend:
    # """
    # Create the default speech backend.
    # Keeps pyttsx3 as the active engine, falling back to console output if unavailable.
    # """
    # if prefer_pyttsx3:
    #     try:
    #         # Check import availability without importing at module level
    #         import importlib.util
    #         spec = importlib.util.find_spec('pyttsx3')
    #         if spec is not None:
    #             logger.info('Selecting pyttsx3 backend')
    #             # return Pyttsx3Backend(rate=rate)
    #             return KokoroBackend(rate=rate)
    #         else:
    #             logger.warning('pyttsx3 not found - falling back to console backend.')
    #     except Exception as exc:
    #         logger.warning(f'Error checking pyttsx3 availability ({exc}) - falling back to console backend.')
    # logger.info('Selecting console audio backend')
    # return ConsoleBackend()
    try:
        logger.info('Selecting Kokoro backend')
        return KokoroBackend(rate=rate)

    except Exception as exc:
        logger.warning(
            f'Kokoro backend init failed ({exc}) - falling back to console backend.'
        )

    logger.info('Selecting console audio backend')
    return ConsoleBackend()


