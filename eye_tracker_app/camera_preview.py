from __future__ import annotations

import threading
import time
from typing import Optional

from eye_tracker_app.camera_gaze import CameraGaze


class CameraPreview:
    """Shows what the camera is seeing in a small OpenCV window.

    Reads frames from an existing CameraGaze instance (no second webcam handle).
    """

    def __init__(self, gaze: CameraGaze, window_name: str = "Camera Preview") -> None:
        self._gaze = gaze
        self._window_name = window_name
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="CameraPreview", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)

    def _loop(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception:
            return

        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, 420, 280)
        except Exception:
            pass

        try:
            while self._running:
                frame = self._gaze.get_latest_preview_frame_bgr()
                if frame is not None:
                    try:
                        cv2.imshow(self._window_name, frame)
                        cv2.waitKey(1)
                    except Exception:
                        # If GUI backend isn't available, just stop preview.
                        break
                time.sleep(0.02)
        finally:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
