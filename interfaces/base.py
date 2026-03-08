import threading
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable, Optional
from time import time
logger = logging.getLogger(__name__)

class ModuleState(Enum):
    UNINITIALIZED = auto()
    RUNNING = auto()
    SUSPENDED = auto()
    STOPPED = auto()
    ERROR = auto()

class BaseModule(ABC):

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._state = ModuleState.UNINITIALIZED
        self._state_lock = threading.Lock()
        self._event_callback: Optional[Callable] = None
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._last_emit_at: Optional[float] = None
        self._emit_count = 0
        logger.debug(f'[{self.module_name}] Module created')

    def start(self) -> bool:
        with self._state_lock:
            if self._state == ModuleState.RUNNING:
                logger.warning(f'[{self.module_name}] start() called but already RUNNING')
                return True
            if self._state == ModuleState.SUSPENDED:
                logger.warning(f'[{self.module_name}] start() called while SUSPENDED — use resume() instead')
                return False
            logger.info(f'[{self.module_name}] Starting...')
            try:
                self._on_start()
                self._state = ModuleState.RUNNING
                self._started_at = time()
                logger.info(f'[{self.module_name}] Started successfully')
                return True
            except Exception as e:
                self._state = ModuleState.ERROR
                logger.error(f'[{self.module_name}] Failed to start: {e}', exc_info=True)
                return False

    def stop(self) -> bool:
        with self._state_lock:
            if self._state == ModuleState.STOPPED:
                logger.debug(f'[{self.module_name}] Already stopped')
                return True
            prev_state = self._state
            logger.info(f'[{self.module_name}] Stopping (was {prev_state.name})...')
            try:
                self._on_stop()
                self._state = ModuleState.STOPPED
                self._stopped_at = time()
                logger.info(f'[{self.module_name}] Stopped')
                return True
            except Exception as e:
                self._state = ModuleState.ERROR
                logger.error(f'[{self.module_name}] Error during stop: {e}', exc_info=True)
                return False

    def suspend(self) -> bool:
        with self._state_lock:
            if self._state != ModuleState.RUNNING:
                logger.warning(f'[{self.module_name}] suspend() called but state is {self._state.name} (must be RUNNING)')
                return False
            logger.info(f'[{self.module_name}] Suspending...')
            try:
                self._on_suspend()
                self._state = ModuleState.SUSPENDED
                logger.info(f'[{self.module_name}] Suspended')
                return True
            except Exception as e:
                self._state = ModuleState.ERROR
                logger.error(f'[{self.module_name}] Error during suspend: {e}', exc_info=True)
                return False

    def resume(self) -> bool:
        with self._state_lock:
            if self._state != ModuleState.SUSPENDED:
                logger.warning(f'[{self.module_name}] resume() called but state is {self._state.name} (must be SUSPENDED)')
                return False
            logger.info(f'[{self.module_name}] Resuming...')
            try:
                self._on_resume()
                self._state = ModuleState.RUNNING
                logger.info(f'[{self.module_name}] Resumed')
                return True
            except Exception as e:
                self._state = ModuleState.ERROR
                logger.error(f'[{self.module_name}] Error during resume: {e}', exc_info=True)
                return False

    def set_event_callback(self, callback: Callable):
        self._event_callback = callback
        logger.debug(f'[{self.module_name}] Event callback registered')

    def _emit(self, event):
        self._last_emit_at = time()
        self._emit_count += 1
        if self._event_callback is not None:
            try:
                self._event_callback(event)
            except Exception as e:
                logger.error(f'[{self.module_name}] Event callback raised: {e}', exc_info=True)
        else:
            logger.warning(f'[{self.module_name}] Event emitted but no callback registered: {type(event).__name__} — call set_event_callback() first')

    @property
    def state(self) -> ModuleState:
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self.state == ModuleState.RUNNING

    @property
    def is_suspended(self) -> bool:
        return self.state == ModuleState.SUSPENDED

    @property
    def is_healthy(self) -> bool:
        return self.state in (ModuleState.RUNNING, ModuleState.SUSPENDED)

    @property
    def uptime_seconds(self) -> Optional[float]:
        if self._started_at is None:
            return None
        end = self._stopped_at if self._stopped_at else time()
        return round(end - self._started_at, 2)

    def get_status(self) -> dict:
        return {'module': self.module_name, 'state': self.state.name, 'is_running': self.is_running, 'is_healthy': self.is_healthy, 'uptime_sec': self.uptime_seconds, 'emit_count': self._emit_count}

    @abstractmethod
    def _on_start(self):
        ...

    @abstractmethod
    def _on_stop(self):
        ...

    @abstractmethod
    def _on_suspend(self):
        ...

    @abstractmethod
    def _on_resume(self):
        ...