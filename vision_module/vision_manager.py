

import threading
import logging
import time
from typing import Optional
import numpy as np

from interfaces.vision_interface import VisionInterface
from vision_module.sentinel import Sentinel
from vision_module.safety_pipeline import SafetyPipeline, MODE_CONTINUOUS, MODE_REACTIVE
from vision_module.semantic_tasks import SemanticTasks

logger = logging.getLogger(__name__)


LEVEL_SENTINEL_ONLY     = "sentinel_only"
LEVEL_SENTINEL_SAFETY   = "sentinel_and_safety"
LEVEL_SENTINEL_MAX      = "sentinel_and_safety_max"
LEVEL_SENTINEL_SEMANTIC = "sentinel_and_semantic"


class VisionManager(VisionInterface):
    """
    Top-level vision module. Registered with the controller via
    controller.register_vision_module(VisionManager()).

    Coordinates sentinel, safety pipeline, and semantic tasks.
    Translates vision level strings from the controller into
    concrete configuration of each sub-component.
    """

    def __init__(self):
        super().__init__()

        # Sub-components — created here, started in _on_start()
        self._sentinel  = None
        self._safety    = None
        self._semantic  = None

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock    = threading.Lock()
        self._frame_event   = threading.Event() 

        self._worker_thread: Optional[threading.Thread] = None
        self._worker_running = False
        self._suspended      = threading.Event() 

        self._semantic_result_callback = None

        logger.debug("VisionManager created")


    def _on_start(self):
        """
        Load all models and start worker thread.
        Called by controller during system startup.
        """
        logger.info("VisionManager starting — loading models...")

        # Build sub-components
        self._safety   = SafetyPipeline(emit_callback=self._emit)
        self._sentinel = Sentinel(
            emit_callback=self._emit,
            on_motion_frame=self._safety.push_frame 
        )
        self._semantic = SemanticTasks()

        safety_ok   = self._safety.start()
        semantic_ok = self._semantic.load_models()

        if not safety_ok:
            logger.warning(
                "SafetyPipeline models failed to load — "
                "YOLO detection unavailable. Sentinel still active."
            )
        if not semantic_ok:
            logger.warning(
                "SemanticTasks models failed to load — "
                "caption/OCR unavailable."
            )

        # Start vision worker thread
        self._worker_running = True
        self._suspended.clear()
        self._worker_thread  = threading.Thread(
            target=self._worker_loop,
            name="VisionWorker",
            daemon=True
        )
        self._worker_thread.start()

        self._apply_vision_level(LEVEL_SENTINEL_ONLY)

        logger.info("VisionManager started")

    def _on_stop(self):
        """Stop worker thread and all sub-components."""
        self._worker_running = False
        self._frame_event.set()   

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

        if self._safety:
            self._safety.stop()

        logger.info("VisionManager stopped")

    def _on_suspend(self):
        """
        Pause frame processing. Worker thread idles.
        Safety pipeline suspends — YOLO stops running.
        Sentinel reset so first post-resume frame doesn't false-trigger.
        """
        self._suspended.set()

        if self._safety:
            self._safety.suspend()
        if self._sentinel:
            self._sentinel.reset()

        logger.info("VisionManager suspended")

    def _on_resume(self):
        """Resume frame processing with clean state."""
        self._suspended.clear()

        if self._safety:
            self._safety.resume()
        if self._sentinel:
            self._sentinel.reset()

        self._apply_vision_level(self.vision_level)

        logger.info("VisionManager resumed")


    def on_frame(self, frame):
        """
        Receive a frame from the controller each tick.

        Stores the latest frame and signals the worker thread.
        Does not block — controller's 33ms tick must not stall here.

        Args:
            frame: TimestampedFrame from FrameBuffer, or raw np.ndarray.
        """
        # Handle both TimestampedFrame and raw numpy array
        raw = frame.frame if hasattr(frame, 'frame') else frame

        with self._frame_lock:
            self._latest_frame = raw

        self._frame_event.set()

        # --- THE FIX: Prevent Pipeline Starvation ---
        # When in continuous mode, Sentinel's motion callback is disabled.
        # We must explicitly feed fresh frames to the SafetyPipeline.
        # This is safe because push_frame() is non-blocking in continuous mode.
        if self._safety and self._safety._mode == MODE_CONTINUOUS:
            self._safety.push_frame(raw)

    # Update this exact line:
    def _apply_vision_level(self, level: str, state_context: str = ""):
        """
        Configure sub-components based on the vision level set by controller.
        """
        logger.info(f"Vision level -> {level}" + (f" [{state_context}]" if state_context else ""))

        if level == LEVEL_SENTINEL_ONLY:
            if self._sentinel:
                self._sentinel.set_on_motion_frame(self._safety.push_frame)
            if self._safety:
                self._safety.set_mode(MODE_REACTIVE, context=state_context)

        elif level == LEVEL_SENTINEL_SAFETY:
            if self._sentinel:
                self._sentinel.set_on_motion_frame(None)
            if self._safety:
                self._safety.set_mode(MODE_CONTINUOUS, fps_override=False, context=state_context)

        elif level == LEVEL_SENTINEL_MAX:
            if self._sentinel:
                self._sentinel.set_on_motion_frame(None)
            if self._safety:
                self._safety.set_mode(MODE_CONTINUOUS, fps_override=True, context=state_context)

        elif level == LEVEL_SENTINEL_SEMANTIC:
            if self._sentinel:
                self._sentinel.set_on_motion_frame(self._safety.push_frame)
            if self._safety:
                self._safety.set_mode(MODE_REACTIVE, context=state_context)

        else:
            logger.warning(f"Unknown vision level: '{level}' — no change")

    def request_caption(self):
        """
        Start a BLIP captioning task on the current frame.
        Result delivered to controller via _on_semantic_result().
        Called by controller when handling REQUEST_CAPTION intent.
        """
        if not self._semantic:
            logger.warning("request_caption: SemanticTasks not initialized")
            return

        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            logger.warning("request_caption: no frame available yet")
            return

        started = self._semantic.request_caption(
            frame, callback=self._on_semantic_result
        )
        if not started:
            logger.warning("request_caption: SemanticTasks busy — rejected")

    def request_ocr(self):
        """
        Start an EasyOCR task on the current frame.
        Result delivered to controller via _on_semantic_result().
        """
        if not self._semantic:
            logger.warning("request_ocr: SemanticTasks not initialized")
            return

        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            logger.warning("request_ocr: no frame available yet")
            return

        started = self._semantic.request_ocr(
            frame, callback=self._on_semantic_result
        )
        if not started:
            logger.warning("request_ocr: SemanticTasks busy — rejected")

    def cancel_semantic_task(self):
        """
        Cancel any in-flight semantic task.
        Called by controller on entering ALERT or OVERRIDE state.
        """
        if self._semantic:
            self._semantic.cancel()


    def set_semantic_result_callback(self, callback):
        """
        Register where to deliver semantic task results.

        Controller calls this during setup so caption/OCR results
        reach the AudioQueue as SPEAK commands.

        Args:
            callback: function(text: str) — typically posts a SPEAK AudioCommand
        """
        self._semantic_result_callback = callback

    def _on_semantic_result(self, text: str):
        """
        Called by SemanticTasks when caption/OCR completes.
        Routes result to controller via registered callback.
        """
        logger.info(f"Semantic result: '{text[:60]}...' " if len(text) > 60 else
                    f"Semantic result: '{text}'")

        if self._semantic_result_callback:
            try:
                self._semantic_result_callback(text)
            except Exception as e:
                logger.error(f"Semantic result callback raised: {e}",
                             exc_info=True)
        else:
            logger.warning(
                "No semantic result callback registered — result dropped. "
                "Call set_semantic_result_callback() before requesting tasks."
            )


    def _worker_loop(self):
        """
        Vision worker thread. Waits for new frames, runs sentinel.
        Safety pipeline has its own thread for continuous mode.
        Sentinel runs here — one frame at a time, in order.
        """
        logger.info("VisionManager worker loop started")

        while self._worker_running:

            signalled = self._frame_event.wait(timeout=0.1)

            if not signalled:
                continue

            self._frame_event.clear()

            
            if self._suspended.is_set():
                continue

            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                continue

            try:
                self._sentinel.process_frame(frame)
            except Exception as e:
                logger.error(f"Sentinel process_frame raised: {e}",
                             exc_info=True)

        logger.info("VisionManager worker loop exited")


    def get_status(self) -> dict:
        """Extends BaseModule.get_status() with vision-specific stats."""
        base = super().get_status()
        base.update({
            "vision_level": self.vision_level,
            "sentinel":     self._sentinel.get_stats() if self._sentinel else None,
            "safety":       self._safety.get_stats()   if self._safety   else None,
            "semantic":     self._semantic.get_stats() if self._semantic  else None,
        })
        return base

    def get_debug_frame(self) -> Optional[np.ndarray]:
        """Returns the YOLO-annotated frame if available, otherwise the raw frame."""
        if self._safety and hasattr(self._safety, '_annotated_frame'):
            with getattr(self._safety, '_annotated_lock', threading.Lock()):
                if self._safety._annotated_frame is not None:
                    return self._safety._annotated_frame.copy()
        
        # Fallback to the latest raw frame
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None



if __name__ == "__main__":
    import sys, types, time

    config = types.ModuleType('config')
    class VisionConfig:
        SENTINEL_MOTION_THRESHOLD  = 25
        SENTINEL_MOTION_AREA_RATIO = 0.02
        SAFETY_FPS                 = 5
        SAFETY_FPS_OVERRIDE        = 10
        YOLO_CONFIDENCE            = 0.5
        RISK_PERSISTENCE_FRAMES    = 3
        DEPTH_NEAR_THRESHOLD       = 0.3
        DEPTH_MID_THRESHOLD        = 0.6
        DEPTH_REUSE_MS             = 500
        OBJECT_MODEL_PATH          = "yolov8m.pt"
    config.VisionConfig = VisionConfig
    sys.modules['config'] = config

    core_mod      = types.ModuleType('core')
    event_bus_mod = types.ModuleType('core.event_bus')
    from enum import Enum, auto
    from dataclasses import dataclass, field
    from time import time as _time

    class VisionEventType(Enum):
        NONE = auto(); MOTION = auto(); RISK = auto(); PERSON_LEFT = auto()

    @dataclass
    class VisionEvent:
        event_type: VisionEventType
        confidence: float = 0.0
        hazard_class: str = None
        depth_zone: str = None
        tracker_id: int = None
        timestamp: float = field(default_factory=_time)

    event_bus_mod.VisionEvent    = VisionEvent
    event_bus_mod.VisionEventType = VisionEventType
    sys.modules['core']           = core_mod
    sys.modules['core.event_bus'] = event_bus_mod

    iface_mod      = types.ModuleType('interfaces')
    vision_if_mod  = types.ModuleType('interfaces.vision_interface')
    base_mod       = types.ModuleType('interfaces.base')

    from enum import Enum as _Enum, auto as _auto
    class ModuleState(_Enum):
        UNINITIALIZED = _auto(); RUNNING = _auto()
        SUSPENDED = _auto(); STOPPED = _auto(); ERROR = _auto()

    from abc import ABC, abstractmethod
    from typing import Callable as _Callable

    class BaseModule(ABC):
        def __init__(self, name=""):
            self.module_name   = name
            self._state        = ModuleState.UNINITIALIZED
            self._state_lock   = threading.Lock()
            self._event_callback = None
            self._emit_count   = 0
        def set_event_callback(self, cb): self._event_callback = cb
        def _emit(self, event):
            self._emit_count += 1
            if self._event_callback: self._event_callback(event)
        def start(self):
            self._on_start(); self._state = ModuleState.RUNNING; return True
        def stop(self):
            self._on_stop(); self._state = ModuleState.STOPPED
        def suspend(self):
            self._on_suspend(); self._state = ModuleState.SUSPENDED
        def resume(self):
            self._on_resume(); self._state = ModuleState.RUNNING
        @property
        def is_running(self): return self._state == ModuleState.RUNNING
        @property
        def is_suspended(self): return self._state == ModuleState.SUSPENDED
        @property
        def state(self): return self._state
        def get_status(self): return {"module": self.module_name,
                                      "state": self._state.name}
        @abstractmethod
        def _on_start(self): ...
        @abstractmethod
        def _on_stop(self): ...
        @abstractmethod
        def _on_suspend(self): ...
        @abstractmethod
        def _on_resume(self): ...

    class VisionInterface(BaseModule, ABC):
        def __init__(self): super().__init__("VisionManager")
        @abstractmethod
        def on_frame(self, frame): ...
        @abstractmethod
        def _apply_vision_level(self, level: str): ...
        @abstractmethod
        def request_caption(self): ...
        @abstractmethod
        def request_ocr(self): ...
        @abstractmethod
        def cancel_semantic_task(self): ...

    base_mod.BaseModule             = BaseModule
    base_mod.ModuleState            = ModuleState
    vision_if_mod.VisionInterface   = VisionInterface
    sys.modules['interfaces']                  = iface_mod
    sys.modules['interfaces.base']             = base_mod
    sys.modules['interfaces.vision_interface'] = vision_if_mod

    vm_pkg   = types.ModuleType('vision_module')

    class MockSentinel:
        def __init__(self, emit_callback, on_motion_frame=None):
            self._on_motion_frame = on_motion_frame
            self._emit            = emit_callback
            self.frames           = 0
        def process_frame(self, frame): self.frames += 1
        def reset(self): pass
        def set_on_motion_frame(self, cb): self._on_motion_frame = cb
        def get_stats(self): return {"frames_processed": self.frames}

    class MockSafetyPipeline:
        def __init__(self, emit_callback):
            self._mode = MODE_REACTIVE; self.started = False
        def start(self): self.started = True; return True
        def stop(self): pass
        def suspend(self): pass
        def resume(self): pass
        def push_frame(self, f): pass
        def set_mode(self, mode, fps_override=False, context=""): self._mode = mode
        def get_stats(self): return {"mode": self._mode}
        @property
        def mode(self): return self._mode

    class MockSemanticTasks:
        def load_models(self): return False
        def request_caption(self, f, callback):
            callback("I see: a test scene"); return True
        def request_ocr(self, f, callback):
            callback("Text: hello world"); return True
        def cancel(self): pass
        def is_busy(self): return False
        def get_stats(self): return {}

    sentinel_mod = types.ModuleType('vision_module.sentinel')
    safety_mod   = types.ModuleType('vision_module.safety_pipeline')
    semantic_mod = types.ModuleType('vision_module.semantic_tasks')

    sentinel_mod.Sentinel       = MockSentinel
    safety_mod.SafetyPipeline   = MockSafetyPipeline
    safety_mod.MODE_CONTINUOUS  = MODE_CONTINUOUS
    safety_mod.MODE_REACTIVE    = MODE_REACTIVE
    semantic_mod.SemanticTasks  = MockSemanticTasks

    sys.modules['vision_module']                  = vm_pkg
    sys.modules['vision_module.sentinel']         = sentinel_mod
    sys.modules['vision_module.safety_pipeline']  = safety_mod
    sys.modules['vision_module.semantic_tasks']   = semantic_mod

    # Overwrite module-level imports with mocks for the test run
    global Sentinel, SafetyPipeline, SemanticTasks
    Sentinel = MockSentinel
    SafetyPipeline = MockSafetyPipeline
    SemanticTasks = MockSemanticTasks

    print("Running VisionManager tests...\n")

    events         = []
    semantic_out   = []

    def mock_emit(event):       events.append(event)
    def mock_sem_cb(text):      semantic_out.append(text)

    vm = VisionManager()
    vm.set_event_callback(mock_emit)

    vm.start()
    assert vm.is_running
    assert vm._sentinel  is not None
    assert vm._safety    is not None
    assert vm._semantic  is not None
    print("PASS  Test 1: start() initialises all sub-components")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vm.on_frame(frame)
    time.sleep(0.1) 
    assert vm._sentinel.frames >= 1
    print("PASS  Test 2: on_frame() delivers frame to sentinel via worker")

    vm._apply_vision_level(LEVEL_SENTINEL_SAFETY)
    assert vm._sentinel._on_motion_frame is None   
    assert vm._safety.mode == MODE_CONTINUOUS
    print("PASS  Test 3: sentinel_and_safety -> continuous, no motion callback")

    vm._apply_vision_level(LEVEL_SENTINEL_ONLY)
    assert vm._sentinel._on_motion_frame is not None  
    assert vm._safety.mode == MODE_REACTIVE
    print("PASS  Test 4: sentinel_only -> reactive, motion callback set")

    vm._apply_vision_level(LEVEL_SENTINEL_SEMANTIC)
    assert vm._sentinel._on_motion_frame is not None  
    assert vm._safety.mode == MODE_REACTIVE
    print("PASS  Test 5: sentinel_and_semantic -> reactive guard active")

    vm.set_semantic_result_callback(mock_sem_cb)
    vm.request_caption()
    time.sleep(0.2)
    assert len(semantic_out) == 1
    assert "I see:" in semantic_out[0]
    print(f"PASS  Test 6: request_caption() -> callback: '{semantic_out[0]}'")

    vm.request_ocr()
    time.sleep(0.2)
    assert len(semantic_out) == 2
    assert "Text:" in semantic_out[1]
    print(f"PASS  Test 7: request_ocr() -> callback: '{semantic_out[1]}'")

    vm.cancel_semantic_task()
    print("PASS  Test 8: cancel_semantic_task() runs without error")

    vm.suspend()
    assert vm.is_suspended
    vm.resume()
    assert vm.is_running
    print("PASS  Test 9: suspend/resume cycle works")

    status = vm.get_status()
    assert "vision_level" in status
    assert "sentinel" in status
    assert "safety" in status
    print(f"PASS  Test 10: get_status() includes vision keys")

    vm.stop()
    print("\nAll VisionManager tests passed.")