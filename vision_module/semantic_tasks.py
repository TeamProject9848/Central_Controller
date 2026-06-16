

import threading
import logging
import numpy as np
from typing import Callable, Optional

from config import VisionConfig

logger = logging.getLogger(__name__)


class SemanticTasks:
    """
    On-demand image captioning and OCR.

    One task runs at a time. Results delivered via callback — never spoken
    directly. Cancellable at any point (running inference completes but
    result is dropped if cancelled mid-flight).
    """

    def __init__(self):
     
        self._blip_processor = None
        self._blip_model     = None
        self._ocr_reader     = None
        self._device         = None
        self._models_loaded  = False
        self._load_lock      = threading.Lock()

        # Task control
        self._busy_lock      = threading.Lock()
        self._cancelled      = threading.Event()
        self._current_thread: Optional[threading.Thread] = None

        # Stats
        self._caption_count  = 0
        self._ocr_count      = 0
        self._cancel_count   = 0

        logger.debug("SemanticTasks initialized")



    def load_models(self) -> bool:
        """
        Load BLIP and EasyOCR models.
        Called once during VisionManager._on_start().
        Returns True if at least BLIP or EasyOCR loaded successfully.
        """
        with self._load_lock:
            if self._models_loaded:
                return self._blip_model is not None or self._ocr_reader is not None

            # 1. Load BLIP
            try:
                import torch
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading BLIP on {self._device}...")

                self._blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self._blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(self._device)
                logger.info("BLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"BLIP model load failed: {e}")

            # 2. Load EasyOCR
            try:
                import easyocr
                logger.info("Loading EasyOCR...")
                import torch as _torch
                self._ocr_reader = easyocr.Reader(
                    ['en'], gpu=_torch.cuda.is_available()
                )
                logger.info("EasyOCR loaded successfully")
            except Exception as e:
                logger.warning(f"EasyOCR load failed: {e} — OCR will be unavailable.")

            self._models_loaded = True
            success = self._blip_model is not None or self._ocr_reader is not None
            if success:
                logger.info(
                    f"SemanticTasks models loaded. "
                    f"BLIP={self._blip_model is not None}, "
                    f"EasyOCR={self._ocr_reader is not None}"
                )
            return success

    def request_caption(self,
                        frame: np.ndarray,
                        callback: Callable[[str], None]) -> bool:
        """
        Run BLIP captioning on frame. Non-blocking.

        Args:
            frame:    BGR numpy array (current scene snapshot).
            callback: Called with caption string when done.
                      Called with error string if models unavailable.
        Returns:
            True if task started, False if already busy.
        """
        return self._start_task(
            target=self._run_caption,
            args=(frame.copy(), callback),
            name="CaptionTask"
        )

    def request_ocr(self,
                    frame: np.ndarray,
                    callback: Callable[[str], None]) -> bool:
        """
        Run EasyOCR text extraction on frame. Non-blocking.

        Args:
            frame:    BGR numpy array (current scene snapshot).
            callback: Called with extracted text string when done.
                      Called with error string if models unavailable.
        Returns:
            True if task started, False if already busy.
        """
        return self._start_task(
            target=self._run_ocr,
            args=(frame.copy(), callback),
            name="OCRTask"
        )

    def cancel(self):
        """
        Request cancellation of the current task.

        If a task is running, it will check this flag at the start
        and drop its result without calling the callback.
        Models don't support mid-inference stop — the thread finishes
        naturally but silently.
        """
        if self.is_busy():
            self._cancelled.set()
            self._cancel_count += 1
            logger.info("SemanticTasks cancel requested")

    def is_busy(self) -> bool:
        """True if a task is currently running."""
        acquired = self._busy_lock.acquire(blocking=False)
        if acquired:
            self._busy_lock.release()
            return False
        return True

    def is_available(self) -> bool:
        """True if at least one model is loaded and ready."""
        return self._blip_model is not None or self._ocr_reader is not None

    def get_stats(self) -> dict:
        return {
            "available":     self.is_available(),
            "busy":          self.is_busy(),
            "captions_done": self._caption_count,
            "ocr_done":      self._ocr_count,
            "cancelled":     self._cancel_count,
        }


    def _start_task(self,
                    target: Callable,
                    args: tuple,
                    name: str) -> bool:
        """
        Acquire busy lock and launch task on background thread.
        Returns False immediately if already busy.
        """
        acquired = self._busy_lock.acquire(blocking=False)
        if not acquired:
            logger.debug(f"SemanticTasks busy — {name} rejected")
            return False

        self._cancelled.clear()
        self._current_thread = threading.Thread(
            target=self._task_wrapper,
            args=(target, args),
            name=name,
            daemon=True
        )
        self._current_thread.start()
        return True

    def _task_wrapper(self, target: Callable, args: tuple):
        """Wraps task execution — always releases busy lock on exit."""
        try:
            target(*args)
        except Exception as e:
            logger.error(f"Semantic task raised: {e}", exc_info=True)
        finally:
            self._busy_lock.release()



    def _run_caption(self,
                     frame: np.ndarray,
                     callback: Callable[[str], None]):
        """
        Run BLIP captioning. Mirrors notebook's caption_image() with
        speak_threaded() replaced by callback(result).
        """
   
        if self._cancelled.is_set():
            logger.debug("Caption task cancelled before start")
            return

        if self._blip_model is None:
            callback("Scene description unavailable — model not loaded.")
            return

        try:
            import cv2

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs  = self._blip_processor(
                img_rgb, return_tensors="pt"
            ).to(self._device)

            import torch
            with torch.no_grad():
                out = self._blip_model.generate(
                    **inputs,
                    max_new_tokens=50
                )

            if self._cancelled.is_set():
                logger.debug("Caption task result dropped — cancelled")
                return

            caption = self._blip_processor.decode(
                out[0], skip_special_tokens=True
            )
            self._caption_count += 1
            logger.info(f"Caption result: '{caption}'")

  
            callback(f"I see: {caption}")

        except Exception as e:
            logger.error(f"Caption inference failed: {e}", exc_info=True)
            if not self._cancelled.is_set():
                callback("Sorry, I could not describe the scene.")

 

    def _run_ocr(self,
                 frame: np.ndarray,
                 callback: Callable[[str], None]):
        """
        Run EasyOCR text extraction. Mirrors notebook's perform_ocr()
        with speak_threaded() replaced by callback(result).
        """
        if self._cancelled.is_set():
            logger.debug("OCR task cancelled before start")
            return

        if self._ocr_reader is None:
            callback("Text reading unavailable — model not loaded.")
            return

        try:
            results = self._ocr_reader.readtext(frame)
            text    = " ".join([res[1] for res in results]).strip()

            if self._cancelled.is_set():
                logger.debug("OCR task result dropped — cancelled")
                return

            if not text:
                result_text = "No text detected."
            else:
                
                try:
                    from langdetect import detect
                    if detect(text) != 'en':
                        result_text = "Text detected but not in English."
                    else:
                        result_text = f"Text: {text}"
                except Exception:
                    result_text = f"Text: {text}"

            self._ocr_count += 1
            logger.info(f"OCR result: '{result_text}'")

            callback(result_text)

        except Exception as e:
            logger.error(f"OCR inference failed: {e}", exc_info=True)
            if not self._cancelled.is_set():
                callback("Sorry, I could not read the text.")



if __name__ == "__main__":
    import sys, types, time

    config = types.ModuleType('config')
    class VisionConfig:
        pass
    config.VisionConfig = VisionConfig
    sys.modules['config'] = config

    print("Running SemanticTasks tests...\n")

    st = SemanticTasks()

    assert not st.is_busy()
    print("PASS  Test 1: is_busy() = False at start")


    assert not st.is_available()
    print("PASS  Test 2: is_available() = False before load")

    
    results = []
    def cb(text): results.append(text)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    started = st.request_caption(frame, cb)
    assert started
    time.sleep(0.2)   # Let thread finish
    assert len(results) == 1
    assert "unavailable" in results[0].lower()
    print(f"PASS  Test 3: No model -> callback with error: '{results[0]}'")

    
    results.clear()
    started = st.request_ocr(frame, cb)
    assert started
    time.sleep(0.2)
    assert len(results) == 1
    assert "unavailable" in results[0].lower()
    print(f"PASS  Test 4: No model -> OCR callback with error: '{results[0]}'")

  
    import threading as _threading

    barrier = _threading.Event()
    slow_results = []

    def slow_task(frame, callback):
        # Simulate slow inference
        barrier.wait(timeout=2.0)
        callback("done")

    st._busy_lock.acquire()
    rejected = st.request_caption(frame, cb)
    assert not rejected
    st._busy_lock.release()
    print("PASS  Test 5: Second request while busy -> rejected (False)")

    st._cancelled.clear()
    assert not st._cancelled.is_set()

    st._busy_lock.acquire()
    st.cancel()
    assert st._cancelled.is_set()
    assert st._cancel_count == 1
    st._busy_lock.release()
    print("PASS  Test 6: cancel() sets flag and increments counter")

   
    results.clear()
    st._cancelled.set()  

  
    st._run_caption(frame, cb)
    assert len(results) == 0
    print("PASS  Test 7: Pre-cancelled task drops result without calling callback")

    
    stats = st.get_stats()
    assert all(k in stats for k in
               ["available", "busy", "captions_done", "ocr_done", "cancelled"])
    print(f"PASS  Test 8: get_stats() = {stats}")

    print("\nAll SemanticTasks tests passed.")
    print("NOTE: BLIP/EasyOCR inference not tested here (requires model download).")