import logging
import signal
import sys
from config import SystemConfig
from core.controller import CentralController
from vision_module.vision_manager import VisionManager  # import stays at top

def setup_logging():
    level = getattr(logging, SystemConfig.LOG_LEVEL, logging.DEBUG)
    handlers = [logging.StreamHandler(sys.stdout)]
    if SystemConfig.LOG_FILE:
        import os
        os.makedirs('logs', exist_ok=True)
        handlers.append(logging.FileHandler(SystemConfig.LOG_FILE))
    logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S', handlers=handlers)

def main():
    setup_logging()
    logger = logging.getLogger('main')
    logger.info('Walk Assistance System starting...')

    controller = CentralController()
    controller.register_vision_module(VisionManager())  # ← moved here, after controller exists

    def handle_shutdown(sig, frame):
        logger.info('Shutdown signal received')
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    controller.start(blocking=True)

if __name__ == '__main__':
    main()