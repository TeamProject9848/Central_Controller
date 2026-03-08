import logging
from abc import abstractmethod
from interfaces.base import BaseModule
logger = logging.getLogger(__name__)

class InputInterface(BaseModule):

    def __init__(self):
        super().__init__('SignProcessor')

    def on_frame(self, frame):
        pass

    @abstractmethod
    def set_sensitivity(self, level: str):
        ...

    @abstractmethod
    def clear_buffer(self):
        ...