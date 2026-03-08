import logging
from abc import abstractmethod
from typing import Optional
from interfaces.base import BaseModule
logger = logging.getLogger(__name__)

class AudioInterface(BaseModule):

    def __init__(self):
        super().__init__('AudioManager')

    @abstractmethod
    def speak(self, text: str, rate: Optional[int]=None):
        ...

    @abstractmethod
    def play_alert(self, alert_key: str):
        ...

    @abstractmethod
    def stop(self):
        ...

    @abstractmethod
    def is_speaking(self) -> bool:
        ...