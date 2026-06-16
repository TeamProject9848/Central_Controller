

import threading
import logging
import time
from typing import Callable, Optional
import numpy as np

from config import VisionConfig
from vision_module.depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)


MODE_CONTINUOUS = "continuous"   # NAVIGATION / ALERT / OVERRIDE
MODE_REACTIVE   = "reactive"     # IDLE / SEMANTIC


class SafetyPipeline:
    """
    Hazard detection pipeline.

    Runs YOLO object detection, tracks objects across frames with ByteTrack,
    enriches with MiDaS depth zones, debounces the result over N frames,
    and emits RISK or NONE events.

    Thread safety:
        In CONTINUOUS mode, _continuous_loop() runs on its own thread.
        In REACTIVE mode, on_motion_frame() is called from the sentinel
        callback (vision worker thread).
        Both paths write to _object_depths and _state_buffer via locks.
    """

    def __init__(self, emit_callback: Callable):
        """
        Args:
            emit_callback: self._emit() from VisionManager.
        """
        self._emit = emit_callback

        self._object_model  = None   # YOLOv8 general detection
        self._tracker       = None   # ByteTrack
        self._depth         = DepthEstimator()

        self._mode          = MODE_REACTIVE
        self._target_fps    = VisionConfig.SAFETY_FPS
        self._frame_interval = 1.0 / self._target_fps

        self._running       = False
        self._suspended     = False
        self._loop_thread: Optional[threading.Thread] = None
        self._suspend_event = threading.Event()   # set = suspended
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock    = threading.Lock()

        self._object_depths: dict = {}   # {tracker_id: "NEAR"/"MID"/"FAR"/None}
        self._depths_lock   = threading.Lock()
        self._active_person_ids = set()

        self._state_buffer      = [False] * VisionConfig.RISK_PERSISTENCE_FRAMES
        self._last_risk_class   = None
        self._last_risk_depth   = None
        self._last_emit_state   = "SAFE"   # last confirmed emitted state
        self._debouncer_lock    = threading.Lock()

        self._alerted_trackers: dict[int, float] = {}   # {tracker_id: alert_time}
        self._alerted_lock      = threading.Lock()

        self._yolo_runs     = 0
        self._risk_events   = 0
        self._none_events   = 0

        logger.debug("SafetyPipeline initialized")

    

    def start(self) -> bool:
        """Load models and start continuous loop thread (if needed)."""
        if not self._load_models():
            return False

        self._depth.load_model()
        self._running   = True
        self._suspended = False
        self._suspend_event.clear()

        self._loop_thread = threading.Thread(
            target=self._continuous_loop,
            name="SafetyPipelineLoop",
            daemon=True
        )
        self._loop_thread.start()
        logger.info("SafetyPipeline started")
        return True

    def stop(self):
        """Stop the continuous loop and clean up."""
        self._running = False
        self._suspend_event.set()   # Unblock any waiting suspend
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=3.0)
        self._active_person_ids.clear()
        logger.info("SafetyPipeline stopped")

    def suspend(self):
        """Pause processing. Loop thread stays alive but skips inference."""
        self._suspended = True
        self._suspend_event.set()
        self._active_person_ids.clear()
        logger.debug("SafetyPipeline suspended")

    def resume(self):
        """Resume processing."""
        self._suspended = False
        self._suspend_event.clear()
        self._reset_debouncer()   # Clear stale state from before suspension
        logger.debug("SafetyPipeline resumed")

    
    # MODE SWITCHING — called by VisionManager on vision level change
    

    def set_mode(self, mode: str, fps_override: bool = False, context: str = ""):
        """
        Switch between CONTINUOUS and REACTIVE modes.

        Args:
            mode:         MODE_CONTINUOUS or MODE_REACTIVE
            fps_override: True when in ACTIVE_WALK_OVERRIDE state — uses
                          SAFETY_FPS_OVERRIDE instead of SAFETY_FPS.
        """
        self._mode = mode

        if fps_override:
            self._target_fps = VisionConfig.SAFETY_FPS_OVERRIDE
        else:
            self._target_fps = VisionConfig.SAFETY_FPS

        self._frame_interval = 1.0 / self._target_fps

        state_str = f" [{context}]" if context else ""
        logger.info(f"SafetyPipeline mode={mode} fps={self._target_fps}{state_str}")

    
    

    def push_frame(self, frame: np.ndarray):
        """
        Accept a new frame from VisionManager.

        In CONTINUOUS mode: stored for the loop thread to pick up.
        In REACTIVE mode:   this is called by sentinel's on_motion_frame
                            callback — run one YOLO check immediately.
        """
        with self._frame_lock:
            self._latest_frame = frame

        if self._mode == MODE_REACTIVE:
            self._run_yolo(frame)

    
    

    def _continuous_loop(self):
        """
        In CONTINUOUS mode: pulls latest frame and runs YOLO at target FPS.
        In REACTIVE mode:   sleeps — YOLO is triggered by push_frame instead.
        Checks suspension before each inference run.
        """
        logger.info("SafetyPipeline continuous loop started")

        while self._running:
            tick_start = time.time()

            if self._suspended:
                time.sleep(0.1)
                continue

            if self._mode == MODE_CONTINUOUS:
                with self._frame_lock:
                    frame = self._latest_frame

                if frame is not None:
                    self._run_yolo(frame)

            # Sleep remainder of FPS interval
            elapsed    = time.time() - tick_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("SafetyPipeline continuous loop exited")

    
    

    def _run_yolo(self, frame: np.ndarray):
        """
        Run one YOLO + ByteTrack + depth pass on a frame.
        Emits RISK or NONE based on debounced result.
        Annotates frame for debug display and uses adaptive MiDaS polling.
        """
        try:
            import supervision as sv
            import time

            self._yolo_runs += 1

            results     = self._object_model(frame, verbose=False)[0]
            detections  = sv.Detections.from_ultralytics(results)
            tracked     = self._tracker.update_with_detections(detections)

            h, w, _     = frame.shape
            best_score  = 0.0
            best_class  = None
            best_depth  = None
            best_tracker_id = None
            
            labels = [] # For supervision annotation

            # --- DETECT PERSON EXIT ---
            current_person_ids = set()
            for class_id, tracker_id in zip(tracked.class_id, tracked.tracker_id):
                class_name = self._object_model.model.names[class_id]
                if class_name == 'person':
                    current_person_ids.add(tracker_id)
            
            left_person_ids = self._active_person_ids - current_person_ids
            for left_id in left_person_ids:
                logger.info(f"Person tracker_id={left_id} left frame")
                self._emit_person_left(left_id)
            self._active_person_ids = current_person_ids

            # --- CLEANUP: Remove stale tracker IDs to prevent memory leaks ---
            active_trackers = set(tracked.tracker_id)
            with self._depths_lock:
                stale_keys = [k for k in self._object_depths.keys() if k not in active_trackers]
                for k in stale_keys:
                    del self._object_depths[k]
            with self._alerted_lock:
                stale_alert_keys = [k for k in self._alerted_trackers if k not in active_trackers]
                for k in stale_alert_keys:
                    del self._alerted_trackers[k]

            for xyxy, class_id, tracker_id in zip(
                tracked.xyxy,
                tracked.class_id,
                tracked.tracker_id
            ):
                x1, y1, x2, y2 = xyxy
                current_area = (x2 - x1) * (y2 - y1)
                class_name = self._object_model.model.names[class_id]

                # --- ADAPTIVE MIDAS CACHE LOGIC ---
                is_hazard = class_name in VisionConfig.HAZARD_CLASSES
                needs_update = False

                with self._depths_lock:
                    cached = self._object_depths.get(tracker_id)
                    
                    if cached is None:
                        needs_update = True
                    else:
                        if is_hazard:
                            zone, timestamp, last_area = cached
                            elapsed_ms = (time.time() - timestamp) * 1000
                            
                            if last_area > 0 and current_area > last_area * 1.10:
                                needs_update = True
                                
                            # --- NEW: Retry stuck or dropped calculations after 500ms ---
                            elif zone is None and elapsed_ms > 500:
                                needs_update = True
                                
                            elif zone == "FAR" and elapsed_ms > 1500:
                                needs_update = True
                            elif zone in ("MID", "NEAR") and elapsed_ms > 500:
                                needs_update = True

                if needs_update:
                    self._depth.request_depth(
                        frame, xyxy, tracker_id,
                        callback=self._on_depth_result
                    )
                    with self._depths_lock:
                        # Mark as pending, store the current area to track growth
                        self._object_depths[tracker_id] = (None, time.time(), current_area)

                # Fetch the latest depth zone for scoring and labels
                # Fetch the latest depth zone from cache
                with self._depths_lock:
                    cached = self._object_depths.get(tracker_id)
                    depth_zone = cached[0] if cached else None

                # Calculate area ratio EARLY
                area_ratio = current_area / (w * h)

                # Imminent Collision Override
                if is_hazard and area_ratio > 0.35:
                    depth_zone = "NEAR"

                # Build label for the debug frame
                labels.append(f"{class_name} id:{tracker_id} [{depth_zone or 'CALC'}]")
                
                

                # --- SCORE CALCULATION ---
                # Only score objects that are classified as hazards
                if is_hazard:
                    area_ratio  = current_area / (w * h)
                    bottom_pos  = y2 / h
                    h_score     = (area_ratio * 1.2) + (bottom_pos * 0.8)

                    # --- DEPTH-BASED SCORING BY OBJECT TYPE ---
                    # Person: Alert only when NEAR (far is safe)
                    # Moving objects (car, motorcycle, bus, truck): Alert at MID too
                    # Moving animals (dog): Alert at MID/FAR with reduced scores
                    # Bicycle (static): Alert only when NEAR

                    moving_vehicles = {'car', 'motorcycle', 'bus', 'truck'}
                    moving_animals = {'dog'}
                    static_objects = {'bicycle'}

                    # Check if this tracker was already alerted (for non-NEAR zones)
                    with self._alerted_lock:
                        already_alerted = tracker_id in self._alerted_trackers

                    if class_name == 'person':
                        # Person: alert once at NEAR, then suppress for same tracker
                        if already_alerted:
                            score = 0.0
                        elif depth_zone == "NEAR":
                            score = 1.0
                        elif depth_zone == "MID":
                            score = 0.0  # MID person is safe
                        elif depth_zone == "FAR":
                            score = 0.0  # FAR person is safe
                        else:
                            score = h_score * 0.3  # Unconfirmed depth: low priority

                    elif class_name in moving_vehicles:
                        # Vehicles: NEAR always alerts, MID alerts once then suppresses
                        if depth_zone == "NEAR":
                            score = 1.0  # NEAR always dangerous regardless of history
                        elif depth_zone == "MID":
                            if already_alerted:
                                score = 0.0  # Stationary at MID — already alerted, resolve to idle
                            else:
                                score = 0.8  # First detection at MID — alert
                        elif depth_zone == "FAR":
                            score = 0.0  # FAR vehicles are safe — no alert
                        else:
                            score = h_score * 0.4

                    elif class_name in moving_animals:
                        # Moving animals (dog): alert at NEAR, caution at MID+FAR
                        if depth_zone == "NEAR":
                            score = 1.0
                        elif depth_zone == "MID":
                            score = 0.7  # Dog at MID is concerning
                        elif depth_zone == "FAR":
                            score = 0.5  # Far dog is less urgent
                        else:
                            score = h_score * 0.35

                    elif class_name in static_objects:
                        # Static objects (bicycle): alert only at NEAR
                        if depth_zone == "NEAR":
                            score = 1.0
                        elif depth_zone == "MID":
                            score = 0.2  # Static at MID is minor
                        elif depth_zone == "FAR":
                            score = 0.0  # Static at FAR is safe
                        else:
                            score = h_score * 0.15

                    else:
                        # Fallback for any other hazard classes
                        if depth_zone == "NEAR":
                            score = 1.0
                        elif depth_zone == "MID":
                            score = 0.5
                        elif depth_zone == "FAR":
                            score = 0.3
                        else:
                            score = h_score * 0.3

                    if score > best_score:
                        best_score = score
                        best_class = class_name
                        best_depth = depth_zone
                        best_tracker_id = tracker_id

            # --- ANNOTATE AND STORE DEBUG FRAME ---
            annotated = frame.copy()
            annotated = self._box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = self._label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
            
            with getattr(self, '_annotated_lock', threading.Lock()):
                self._annotated_frame = annotated

            best_score = min(best_score, 1.0)
            # Record the tracker that caused the alert for per-tracker memory
            if best_score > VisionConfig.YOLO_CONFIDENCE and best_class is not None:
                # Find the tracker_id that produced the best score — it's the one being alerted
                for xyxy, class_id, tracker_id in zip(
                    tracked.xyxy, tracked.class_id, tracked.tracker_id
                ):
                    cls = self._object_model.model.names[class_id]
                    if cls == best_class:
                        with self._alerted_lock:
                            if tracker_id not in self._alerted_trackers:
                                self._alerted_trackers[tracker_id] = time.time()
                                logger.debug(f"Marked tracker {tracker_id} ({cls}) as alerted")
                        break
            self._update_debouncer(best_score, best_class, best_depth, best_tracker_id)

        except Exception as e:
            logger.error(f"SafetyPipeline YOLO run failed: {e}", exc_info=True)


    def _on_depth_result(self, tracker_id: int, depth_zone: Optional[str]):
        """Callback from DepthEstimator when MiDaS finishes."""
        import time
        with self._depths_lock:
            if tracker_id in self._object_depths:
                # Unpack the existing tuple. We want to keep the area, but update zone and time.
                _, _, last_area = self._object_depths[tracker_id]
                self._object_depths[tracker_id] = (depth_zone, time.time(), last_area)
                
        logger.debug(f"Depth result: tracker={tracker_id} zone={depth_zone}")
    
    

    def _update_debouncer(self,
                          score: float,
                          hazard_class: Optional[str],
                          depth_zone: Optional[str],
                          tracker_id: Optional[int] = None):
        """
        Debounce the per-frame hazard score before emitting events.

        In CONTINUOUS mode:
            Requires RISK_PERSISTENCE_FRAMES consecutive danger frames before
            emitting RISK to prevent flicker from brief detections.

        In REACTIVE mode:
            Emits RISK immediately on detection (single frame from motion trigger).
            Only debounces NONE to prevent flickering back to safe state.

        Thresholds (from notebook, kept as-is for now — tune after field test):
            score > 0.60 → DANGER (maps to RISK)
            score <= 0.60 → SAFE  (maps to NONE)
        """
        with self._debouncer_lock:
            is_danger = score > VisionConfig.YOLO_CONFIDENCE

            if self._mode == MODE_REACTIVE:
                # In REACTIVE mode: emit RISK immediately, debounce NONE only
                if is_danger and self._last_emit_state != "DANGER":
                    self._last_emit_state = "DANGER"
                    self._last_risk_class = hazard_class
                    self._last_risk_depth = depth_zone
                    logger.info(f"REACTIVE: Hazard detected immediately — {hazard_class} @ {score:.2f}")
                    self._emit_risk(hazard_class, score, depth_zone, tracker_id)
                elif not is_danger:
                    # Even in REACTIVE, require some stability before clearing
                    self._state_buffer.pop(0)
                    self._state_buffer.append(is_danger)
                    all_safe = not any(self._state_buffer)
                    if all_safe and self._last_emit_state != "SAFE":
                        self._last_emit_state = "SAFE"
                        self._emit_none()
            else:
                # CONTINUOUS mode: full debouncing for both RISK and NONE
                # Shift buffer
                self._state_buffer.pop(0)
                self._state_buffer.append(is_danger)

                # Only act when all frames agree
                all_danger = all(self._state_buffer)
                all_safe   = not any(self._state_buffer)

                if all_danger and self._last_emit_state != "DANGER":
                    self._last_emit_state = "DANGER"
                    self._last_risk_class = hazard_class
                    self._last_risk_depth = depth_zone
                    self._emit_risk(hazard_class, score, depth_zone, tracker_id)

                elif all_safe and self._last_emit_state != "SAFE":
                    self._last_emit_state = "SAFE"
                    self._emit_none()

    def _reset_debouncer(self):
        """Clear debouncer state — called on resume to avoid stale data."""
        with self._debouncer_lock:
            self._state_buffer    = [False] * VisionConfig.RISK_PERSISTENCE_FRAMES
            self._last_emit_state = "SAFE"
        self._active_person_ids.clear()
        logger.debug("Debouncer reset")

    
    
    

    def _emit_risk(self,
                   hazard_class: Optional[str],
                   confidence: float,
                   depth_zone: Optional[str],
                   tracker_id: Optional[int] = None):
        """
        Emit RISK event upward.
        This is what replaces speak_threaded() + play_alert_tone() from
        the notebook. Vision reports — controller decides and speaks.
        """
        from core.event_bus import VisionEvent, VisionEventType

        self._risk_events += 1
        logger.warning(
            f"RISK confirmed | class={hazard_class} "
            f"confidence={confidence:.2f} depth={depth_zone}"
        )
        try:
            self._emit(VisionEvent(
                event_type=VisionEventType.RISK,
                confidence=round(min(confidence, 1.0), 4),
                hazard_class=hazard_class,
                depth_zone=depth_zone,
                tracker_id=tracker_id,
                source="safety",
            ))
        except Exception as e:
            logger.error(f"Failed to emit RISK: {e}", exc_info=True)

    def _emit_none(self):
        """Emit NONE — scene confirmed clear after N consecutive safe frames."""
        from core.event_bus import VisionEvent, VisionEventType

        self._none_events += 1
        try:
            self._emit(VisionEvent(
                event_type=VisionEventType.NONE,
                confidence=0.0,
                source="safety",
            ))
        except Exception as e:
            logger.error(f"Failed to emit NONE: {e}", exc_info=True)

    def _emit_person_left(self, tracker_id: int):
        from core.event_bus import VisionEvent, VisionEventType
        try:
            self._emit(VisionEvent(
                event_type=VisionEventType.PERSON_LEFT,
                tracker_id=tracker_id,
            ))
        except Exception as e:
            logger.error(f"Failed to emit PERSON_LEFT: {e}", exc_info=True)

    

    def _load_models(self) -> bool:
        """Load YOLO and ByteTrack. Returns False if models unavailable."""
        try:
            from ultralytics import YOLO
            import supervision as sv

            self._object_model = YOLO(VisionConfig.OBJECT_MODEL_PATH)
            self._tracker      = sv.ByteTrack()

            # --- NEW ANNOTATORS ---
            self._box_annotator = sv.BoxAnnotator()
            self._label_annotator = sv.LabelAnnotator()
            self._annotated_frame = None
            self._annotated_lock = threading.Lock()

            logger.info("SafetyPipeline models loaded (YOLO + ByteTrack)")
            return True

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Model load failed: {e}", exc_info=True)
            return False

      
    

    def get_stats(self) -> dict:
        return {
            "mode":        self._mode,
            "target_fps":  self._target_fps,
            "yolo_runs":   self._yolo_runs,
            "risk_events": self._risk_events,
            "none_events": self._none_events,
            "depth":       self._depth.get_stats(),
            "suspended":   self._suspended,
        }


if __name__ == "__main__":
    import sys, types, time

    config = types.ModuleType('config')
    class VisionConfig:
        SAFETY_FPS              = 5
        SAFETY_FPS_OVERRIDE     = 10
        YOLO_CONFIDENCE         = 0.5
        RISK_PERSISTENCE_FRAMES = 3
        DEPTH_NEAR_THRESHOLD    = 0.3
        DEPTH_MID_THRESHOLD     = 0.6
        DEPTH_REUSE_MS          = 500
        OBJECT_MODEL_PATH       = "yolov8m.pt"
    config.VisionConfig = VisionConfig
    sys.modules['config'] = config

    core = types.ModuleType('core')
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
        source: str = None
        timestamp: float = field(default_factory=_time)

    event_bus_mod.VisionEvent = VisionEvent
    event_bus_mod.VisionEventType = VisionEventType
    sys.modules['core'] = core
    sys.modules['core.event_bus'] = event_bus_mod

    # Mock vision_module.depth_estimator
    vm_mod = types.ModuleType('vision_module')
    de_mod = types.ModuleType('vision_module.depth_estimator')

    class MockDepthEstimator:
        def load_model(self): return False
        def request_depth(self, f, b, tid, callback): callback(tid, None)
        def is_available(self): return False
        def is_busy(self): return False
        def get_stats(self): return {}

    de_mod.DepthEstimator = MockDepthEstimator
    sys.modules['vision_module'] = vm_mod
    sys.modules['vision_module.depth_estimator'] = de_mod

    print("Running SafetyPipeline tests...\n")
    events = []

    def mock_emit(event): events.append(event)

    sp = SafetyPipeline(emit_callback=mock_emit)

    
    assert sp._mode == MODE_REACTIVE
    print("PASS  Test 1: Default mode is REACTIVE")

    sp.set_mode(MODE_CONTINUOUS)
    assert sp._mode == MODE_CONTINUOUS
    assert sp._target_fps == VisionConfig.SAFETY_FPS
    print("PASS  Test 2: set_mode(CONTINUOUS) works")

    sp.set_mode(MODE_CONTINUOUS, fps_override=True)
    assert sp._target_fps == VisionConfig.SAFETY_FPS_OVERRIDE
    print("PASS  Test 3: fps_override doubles FPS")

    sp.set_mode(MODE_REACTIVE)
    assert sp._mode == MODE_REACTIVE
    print("PASS  Test 4: set_mode(REACTIVE) works")

    # Set to continuous mode so Test 5-7 debouncer assertions work
    sp.set_mode(MODE_CONTINUOUS)
    
    sp._update_debouncer(0.9, "person", "NEAR")  
    assert len(events) == 0                      
    sp._update_debouncer(0.9, "person", "NEAR")  
    assert len(events) == 0
    sp._update_debouncer(0.9, "person", "NEAR")  
    assert len(events) == 1
    assert events[-1].event_type == VisionEventType.RISK
    assert events[-1].hazard_class == "person"
    assert events[-1].depth_zone == "NEAR"
    print("PASS  Test 5: Debouncer requires 3 danger frames before RISK")

    
    sp._update_debouncer(0.9, "person", "NEAR")
    assert len(events) == 1   
    print("PASS  Test 6: No duplicate RISK when already in DANGER state")

    
    sp._update_debouncer(0.0, None, None)   
    assert len(events) == 1
    sp._update_debouncer(0.0, None, None)   
    assert len(events) == 1
    sp._update_debouncer(0.0, None, None)  
    assert len(events) == 2
    assert events[-1].event_type == VisionEventType.NONE
    print("PASS  Test 7: NONE emitted after 3 clear frames")

    
    sp._reset_debouncer()
    assert sp._last_emit_state == "SAFE"
    assert all(not x for x in sp._state_buffer)
    print("PASS  Test 8: reset_debouncer() clears state")

    
    sp.suspend()
    assert sp._suspended is True
    sp.resume()
    assert sp._suspended is False
    print("PASS  Test 9: suspend/resume work")

    
    stats = sp.get_stats()
    assert "mode" in stats and "yolo_runs" in stats and "risk_events" in stats
    print(f"PASS  Test 10: get_stats() = {stats}")

    print("\nAll SafetyPipeline tests passed.")
    print("NOTE: YOLO inference not tested here (requires model file).")