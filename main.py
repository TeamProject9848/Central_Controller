import logging
import signal
import sys

# Suppress PyTorch dynamo compiler errors and fall back to eager mode
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

from config import SystemConfig, FaceConfig
from core.controller import CentralController
from vision_module.vision_manager import VisionManager  # import stays at top
from network.websocket_server import CompanionWebSocketServer
import asyncio
from network.network_runner import start_network

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


def print_startup_banner(logger):
    logger.info("============================================================")
    logger.info("SYSTEM STATES & VISION MODES EXPLANATION")
    logger.info("------------------------------------------------------------")
    logger.info("STATES:")
    logger.info("  [IDLE]       : User stationary. Vision is REACTIVE (Sentinel motion only).")
    logger.info("  [NAVIGATION] : User walking. Vision is CONTINUOUS (YOLO @ 5 FPS).")
    logger.info("  [ALERT]      : Hazard detected! Vision is CONTINUOUS tracking. Audio locked.")
    logger.info("  [OVERRIDE]   : Max safety mode. Vision is CONTINUOUS (YOLO @ 10 FPS).")
    logger.info("  [SEMANTIC]   : Caption/OCR running. YOLO paused to free CPU.")
    logger.info("MODES:")
    logger.info("  [REACTIVE]   : YOLO sleeps to save battery, wakes only on pixel motion.")
    logger.info("  [CONTINUOUS] : YOLO runs continuously to track active hazards.")
    logger.info("============================================================")


def register_face_module_if_enabled(controller, logger):
    if not FaceConfig.ENABLE_FACE_MODULE:
        logger.info('Face module disabled by config')
        return
    try:
        from face.backend.real import RealFaceBackend
        from face.gallery_repo import PickleFaceGallery
        from face.module import FaceModule

        backend = RealFaceBackend(device=FaceConfig.FACE_BACKEND_DEVICE)
        gallery = PickleFaceGallery(FaceConfig.GALLERY_PATH)
        controller.register_face_module(FaceModule(backend=backend, gallery_repo=gallery))
        logger.info('Face module registered with real backend')
    except Exception as exc:
        logger.warning(f'Face module not registered: {exc}', exc_info=True)

def register_sign_module_if_enabled(controller, logger):
    try:
        from sign_language.backend import SignDetectionBackend
        controller.register_sign_module(SignDetectionBackend())
        logger.info('Sign language module registered successfully')
    except Exception as exc:
        logger.warning(f'Sign language module not registered: {exc}', exc_info=True)

def main():
    setup_logging()
    logger = logging.getLogger('main')
    
    # --- Add the call here ---
    print_startup_banner(logger)
    
    logger.info('Walk Assistance System starting...')
    # ... rest of main() remains exactly the same

    controller = CentralController()
    start_network(controller)
    controller.register_vision_module(VisionManager())  # ← moved here, after controller exists
    register_face_module_if_enabled(controller, logger)
    register_sign_module_if_enabled(controller, logger)

    def handle_shutdown(sig, frame):
        logger.info('Shutdown signal received')
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    controller.start(blocking=True)

if __name__ == '__main__':
    main()
