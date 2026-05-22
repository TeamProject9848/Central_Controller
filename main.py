import logging
import signal
import sys
from config import SystemConfig
from core.controller import CentralController
from vision_module.vision_manager import VisionManager  # import stays at top

class ColorFormatter(logging.Formatter):
    """Custom formatter adding colors to terminal logs based on level."""
    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # Your exact existing format
    FORMAT_STR = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    DATE_FMT = "%H:%M:%S"

    FORMATS = {
        logging.DEBUG: CYAN + FORMAT_STR + RESET,
        logging.INFO: GREEN + FORMAT_STR + RESET,
        logging.WARNING: YELLOW + FORMAT_STR + RESET,
        logging.ERROR: RED + FORMAT_STR + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT_STR + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT_STR)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FMT)
        return formatter.format(record)

def setup_logging():
    level = getattr(logging, SystemConfig.LOG_LEVEL, logging.INFO)
    
    # 1. Setup the console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    handlers = [console_handler]
    
    # 2. Setup the file handler with standard plain text (if enabled)
    if SystemConfig.LOG_FILE:
        import os
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(SystemConfig.LOG_FILE, encoding='utf-8')
        # Plain text formatter for the file
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s', 
            datefmt='%H:%M:%S'
        ))
        handlers.append(file_handler)
        
    # 3. Apply to root logger
    logging.basicConfig(level=level, handlers=handlers)

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