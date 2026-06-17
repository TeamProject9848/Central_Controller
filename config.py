class CameraConfig:
    PHONE_IP = '192.168.1.8'
    PHONE_PORT = 8080
    STREAM_URL = f'http://{PHONE_IP}:{PHONE_PORT}/video'
    SNAPSHOT_URL = f'http://{PHONE_IP}:{PHONE_PORT}/shot.jpg'
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FRAME_BUFFER_SIZE = 2
    STREAM_TIMEOUT_SEC = 3.0
    RECONNECT_INTERVAL_SEC = 5.0
    MAX_RECONNECT_ATTEMPTS = 5

class VisionConfig:
    SENTINEL_FPS = 30
    SENTINEL_MOTION_THRESHOLD = 25
    SENTINEL_MOTION_AREA_RATIO = 0.005  # drop from 0.02 to 0.005
    SAFETY_FPS = 5
    SAFETY_FPS_OVERRIDE = 10
    YOLO_CONFIDENCE = 0.5
    RISK_PERSISTENCE_FRAMES = 3
    RISK_COOLDOWN_SEC = 3.0
    HAZARD_CLASSES = ['person', 'vehicle', 'dog', 'car', 'truck', 'bus', 'motorcycle']
    DEPTH_FPS_MAX = 2
    DEPTH_REUSE_MS = 500
    DEPTH_NEAR_THRESHOLD = 0.3
    DEPTH_MID_THRESHOLD = 0.6
    SEMANTIC_TASK_TIMEOUT_SEC = 10.0
    OBJECT_MODEL_PATH = "yolov8m.pt"

class AudioConfig:
    PRIORITY_ALERT = 0
    PRIORITY_NAVIGATION_STATUS = 1
    PRIORITY_RESPONSE = 2
    PRIORITY_DESCRIPTION = 3
    ALERT_COOLDOWN_SEC = 2.0
    ALERT_RESOLVE_SEC = 1.5   # Seconds of confirmed safety before ALERT → IDLE
    TTS_SPEECH_RATE = 175
    TTS_MAX_CHARS = 200
    INTERRUPT_ON_HIGHER_PRIORITY = True

class SystemConfig:
    CONTROLLER_TICK_SEC = 0.033
    WIFI_LATENCY_COMPENSATION_MS = 150
    MAX_FRAME_AGE_MS = 300
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/runtime.log'
    DEBUG_DISPLAY = True
    DEBUG_WINDOW_NAME = 'Controller Debug'


class FaceConfig:
    ENABLE_FACE_MODULE = True
    FACE_BACKEND_DEVICE = 'cpu'
    FACE_CROP_SIZE = (160, 160)
    GALLERY_PATH = 'data/face_gallery.pkl'

    # Prompt priorities (relative to AudioConfig)
    PRIORITY_GUIDANCE = AudioConfig.PRIORITY_DESCRIPTION
    PRIORITY_RESULT = AudioConfig.PRIORITY_RESPONSE
    PRIORITY_CRITICAL = AudioConfig.PRIORITY_NAVIGATION_STATUS
    REQUIRED_POSES = ['front', 'left', 'right']
    MATCH_THRESHOLD = 0.6
    MATCH_THRESHOLD_STRONG = 0.6
    MATCH_THRESHOLD_WEAK = 0.8
    LIVENESS_REQUIRED = False

    # Default prompts keyed by message_key to keep controller mapping simple.
    PROMPTS = {
        'registration_start': 'Starting face registration. Look straight ahead.',
        'registration_left': 'Please turn your head to the left.',
        'registration_right': 'Please turn your head to the right.',
        'registration_retry': 'Capture was unclear, please hold still and try again.',
        'registration_complete': 'Face registration complete.',
        'registration_failed': 'Face registration failed. Please try again.',
        'identify_success': 'Identity confirmed.',
        'identify_unknown': 'Face not recognized.',
    }
