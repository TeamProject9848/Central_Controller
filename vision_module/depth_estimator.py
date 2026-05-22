import threading
import logging
import time
import numpy as np
from typing import Callable, Optional, Tuple

from config import VisionConfig

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Non-blocking MiDaS depth estimator.

    Caller submits a (frame, box, tracker_id, callback) request.
    DepthEstimator runs inference on its worker thread and calls
    callback(tracker_id, depth_zone) when done.

    Only one inference runs at a time — MiDaS is expensive and
    running it concurrently would starve the main vision pipeline.
    New requests while busy are dropped (caller retries next frame).

    Depth zones returned:
        "NEAR"  — object is very close, high danger
        "MID"   — object is approaching, caution
        "FAR"   — object is distant, low priority
        None    — MiDaS unavailable or inference failed
    """

    def __init__(self):
        # MiDaS model and transform — loaded lazily on first request
        self._midas           = None
        self._midas_transform = None
        self._device          = None
        self._model_loaded    = False
        self._load_lock       = threading.Lock()

        # One inference at a time
        self._busy_lock = threading.Lock()

        # Thresholds from config
        self._near_threshold = VisionConfig.DEPTH_NEAR_THRESHOLD
        self._mid_threshold  = VisionConfig.DEPTH_MID_THRESHOLD
        self._reuse_ms       = VisionConfig.DEPTH_REUSE_MS

        # Raw score range observed from MiDaS (for scaling)
        # From the notebook: 1000 = far, 3000 = very near
        self._score_min = 1000.0
        self._score_max = 3000.0

        # Stats
        self._requests  = 0
        self._completed = 0
        self._dropped   = 0
        self._failures  = 0

        logger.debug("DepthEstimator initialized (model not yet loaded)")

    
    
    

    def load_model(self) -> bool:
        """
        Load MiDaS model into memory.

        Called once by VisionManager during _on_start().
        Returns True if loaded successfully, False if unavailable.
        Safe to call multiple times — subsequent calls are no-ops.
        """
        with self._load_lock:
            if self._model_loaded:
                return self._midas is not None

            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading MiDaS on {self._device}...")

                model_type = "DPT_Hybrid"
                self._midas = torch.hub.load(
                    "intel-isl/MiDaS", model_type
                ).to(self._device)
                self._midas.eval()

                transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self._midas_transform = transforms.dpt_transform

                self._model_loaded = True
                logger.info("MiDaS loaded successfully")
                return True

            except Exception as e:
                logger.warning(
                    f"MiDaS failed to load: {e} — depth estimation disabled. "
                    f"Safety pipeline will use heuristic scores only."
                )
                self._midas        = None
                self._model_loaded = True  
                return False

    def request_depth(self,
                      frame: np.ndarray,
                      box: Tuple[float, float, float, float],
                      tracker_id: int,
                      callback: Callable[[int, Optional[str]], None]):
        """
        Submit a depth estimation request (non-blocking).

        Runs MiDaS on a background thread. Calls callback(tracker_id,
        depth_zone) when complete. If estimator is busy, request is
        dropped — caller should retry on the next frame.

        Args:
            frame:      Full BGR frame from controller.
            box:        Bounding box (x1, y1, x2, y2) of the object.
            tracker_id: ByteTrack ID — returned in callback for mapping.
            callback:   Called with (tracker_id, depth_zone) on completion.
                        depth_zone is "NEAR", "MID", "FAR", or None on failure.
        """
        self._requests += 1

        if self._midas is None:
            callback(tracker_id, None)
            return

        acquired = self._busy_lock.acquire(blocking=False)
        if not acquired:
            self._dropped += 1
            logger.debug(f"DepthEstimator busy — dropped request for tracker {tracker_id}")
            # --- NEW: Fire callback with None so the pipeline knows it failed ---
            callback(tracker_id, None)
            return

        # Run on background thread — release lock when done
        thread = threading.Thread(
            target=self._run_inference,
            args=(frame.copy(), box, tracker_id, callback),
            daemon=True,
            name=f"MiDaS-{tracker_id}"
        )
        thread.start()

    def is_available(self) -> bool:
        """True if MiDaS model loaded successfully."""
        return self._midas is not None

    def is_busy(self) -> bool:
        """True if inference is currently running."""
        acquired = self._busy_lock.acquire(blocking=False)
        if acquired:
            self._busy_lock.release()
            return False
        return True

    def get_stats(self) -> dict:
        return {
            "model_available": self.is_available(),
            "requests":        self._requests,
            "completed":       self._completed,
            "dropped":         self._dropped,
            "failures":        self._failures,
        }

    
    
    

    def _run_inference(self,
                       frame: np.ndarray,
                       box: Tuple,
                       tracker_id: int,
                       callback: Callable):
        """
        Runs on background thread. Releases busy lock when done.
        """
        depth_zone = None

        try:
            raw_score  = self._get_depth_score(frame, box)
            depth_zone = self._score_to_zone(raw_score)
            self._completed += 1
            raw_display = f"{raw_score:.1f}" if raw_score is not None else "N/A"
            logger.debug(
                f"MiDaS tracker={tracker_id} "
                f"raw={raw_display} "
                f"zone={depth_zone}"
            )
        except Exception as e:
            self._failures += 1
            logger.error(f"MiDaS inference failed for tracker {tracker_id}: {e}",
                         exc_info=True)
        finally:
            self._busy_lock.release()

        # Always call callback — even on failure (with None)
        try:
            callback(tracker_id, depth_zone)
        except Exception as e:
            logger.error(f"Depth callback raised: {e}", exc_info=True)

    def _get_depth_score(self,
                         frame: np.ndarray,
                         box: Tuple) -> Optional[float]:
        """
        Runs MiDaS on the bounding box region of the frame.
        Returns raw inverse-depth score (higher = closer).
        Mirrors the notebook's get_depth_for_box() logic.
        """
        import torch
        import cv2

        x1, y1, x2, y2 = map(int, box)

        # Pad the crop slightly for context
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)

        crop = frame[
            max(0, y1 - pad_y) : min(frame.shape[0], y2 + pad_y),
            max(0, x1 - pad_x) : min(frame.shape[1], x2 + pad_x)
        ]

        if crop.size == 0:
            return None

        img_rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_batch = self._midas_transform(img_rgb).to(self._device)

        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Sample center 80% of the box (avoids edge noise)
        h, w   = depth_map.shape
        cy1, cy2 = int(h * 0.1), int(h * 0.9)
        cx1, cx2 = int(w * 0.1), int(w * 0.9)

        return float(np.mean(depth_map[cy1:cy2, cx1:cx2]))

    def _score_to_zone(self, raw_score: Optional[float]) -> Optional[str]:
        """
        Map raw MiDaS inverse-depth score to a zone string.

        Scales the raw score to [0, 1] then applies thresholds.
        Returns None if raw_score is None (model unavailable).
        """
        if raw_score is None:
            return None

        # Scale to [0, 1] — matches notebook MIN/MAX_SCORE
        if raw_score <= self._score_min:
            scaled = 0.0
        elif raw_score >= self._score_max:
            scaled = 1.0
        else:
            scaled = (raw_score - self._score_min) / (self._score_max - self._score_min)

        # Map to zone — check highest threshold first (MID=0.6 > NEAR=0.3)
        # scaled >= 0.6 → NEAR (danger, very close)
        # scaled >= 0.3 → MID  (caution, approaching)
        # else          → FAR
        if scaled >= self._mid_threshold:
            return "NEAR"
        elif scaled >= self._near_threshold:
            return "MID"
        else:
            return "FAR"



if __name__ == "__main__":
    import sys, types

    config = types.ModuleType('config')
    class VisionConfig:
        DEPTH_NEAR_THRESHOLD = 0.3
        DEPTH_MID_THRESHOLD  = 0.6
        DEPTH_REUSE_MS       = 500
    config.VisionConfig = VisionConfig
    sys.modules['config'] = config

    import numpy as np
    print("Running DepthEstimator tests...\n")

    de = DepthEstimator()

    assert de._score_to_zone(3000.0) == "NEAR";  print("PASS  Test 1: 3000 -> NEAR")
    assert de._score_to_zone(1600.0) == "MID";   print("PASS  Test 2: 1600 -> MID")
    assert de._score_to_zone(1100.0) == "FAR";   print("PASS  Test 3: 1100 -> FAR")
    assert de._score_to_zone(500.0)  == "FAR";   print("PASS  Test 4: 500 -> FAR")
    assert de._score_to_zone(9999.0) == "NEAR";  print("PASS  Test 5: 9999 -> NEAR")
    assert de._score_to_zone(None)   is None;    print("PASS  Test 6: None -> None")

    results = []
    de.request_depth(np.zeros((480,640,3), dtype=np.uint8),
                     (10,10,100,100), tracker_id=42,
                     callback=lambda t,z: results.append((t,z)))
    assert results == [(42, None)];  print("PASS  Test 7: no model -> (42, None)")
    assert not de.is_busy();         print("PASS  Test 8: is_busy=False")
    assert not de.is_available();    print("PASS  Test 9: is_available=False")

    stats = de.get_stats()
    assert "requests" in stats;      print(f"PASS  Test 10: stats={stats}")

    print("\nAll DepthEstimator tests passed.")