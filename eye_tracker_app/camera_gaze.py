from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Optional


@dataclass(frozen=True)
class GazeFeatures:
    """Feature vector used for calibration and gaze prediction.

    Eyes-only: iris position features (no face pose/center/scale).
    """

    left_x: float
    left_y: float
    right_x: float
    right_y: float

    def as_list(self) -> list[float]:
        return [
            self.left_x,
            self.left_y,
            self.right_x,
            self.right_y,
        ]


class CameraGaze:
    """Continuously reads webcam frames and extracts eye features.

    Uses MediaPipe FaceMesh with iris landmarks (refine_landmarks=True).

    Threaded so Tkinter doesn't freeze.
    """

    def __init__(
        self,
        camera_index: int = 0,
        capture_width: int = 1280,
        capture_height: int = 720,
        preview_width: int = 360,
    ) -> None:
        self._camera_index = camera_index
        self._capture_width = capture_width
        self._capture_height = capture_height
        self._preview_width = preview_width
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_features: Optional[GazeFeatures] = None
        self._last_error: Optional[str] = None
        self._last_preview_bgr = None

    def start(self) -> None:
        if self._running:
            return
        self._set_latest(None, "Starting camera…")
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="CameraGaze", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)

    def get_latest(self) -> tuple[Optional[GazeFeatures], Optional[str]]:
        with self._lock:
            return self._last_features, self._last_error

    def get_latest_preview_frame_bgr(self):
        """Return the latest small BGR frame for preview, or None.

        Intentionally untyped to avoid importing numpy at import-time.
        """

        with self._lock:
            if self._last_preview_bgr is None:
                return None
            # Return a copy so consumers don't mutate shared memory.
            try:
                return self._last_preview_bgr.copy()
            except Exception:
                return self._last_preview_bgr

    def _set_latest(self, features: Optional[GazeFeatures], error: Optional[str]) -> None:
        with self._lock:
            self._last_features = features
            self._last_error = error

    def _set_preview_frame(self, frame_bgr) -> None:
        with self._lock:
            self._last_preview_bgr = frame_bgr

    def _run_loop(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore
        except Exception as e:  # pragma: no cover
            self._set_latest(None, f"Missing dependency: {e}")
            return

        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._set_latest(None, "Could not open webcam (check camera index/permissions)")
            return

        # Try to increase capture resolution so the iris landmarks are less noisy
        # when the face is far away.
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._capture_width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._capture_height))
        except Exception:
            pass

        self._set_latest(None, "Loading face model…")

        try:
            model_path = _ensure_face_landmarker_model()
        except Exception as e:
            self._set_latest(None, f"Model download failed: {e}")
            cap.release()
            return

        try:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            self._set_latest(None, f"Failed to init FaceLandmarker: {e}")
            cap.release()
            return

        self._set_latest(None, "Looking for face…")

        t0 = time.time()
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self._set_latest(None, "Failed to read webcam frame")
                    time.sleep(0.05)
                    continue

                frame_h, frame_w = frame.shape[:2]

                # Default preview is a downscaled full frame. If we detect a face,
                # we'll replace it with a zoomed-in face crop.
                preview = _resize_keep_aspect(frame, self._preview_width)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb))
                timestamp_ms = int((time.time() - t0) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if not result.face_landmarks:
                    if preview is not None:
                        self._set_preview_frame(preview)
                    self._set_latest(None, "No face detected")
                    time.sleep(0.01)
                    continue

                face_landmarks = result.face_landmarks[0]

                # Eye-only zoomed preview for readability.
                preview = _make_eye_only_preview(frame, face_landmarks, self._preview_width)

                # Composite preview already includes eye/iris overlays.
                if preview is not None:
                    self._set_preview_frame(preview)

                features = _extract_gaze_features(face_landmarks)
                self._set_latest(features, None)
        finally:
            try:
                landmarker.close()
            except Exception:
                pass
            cap.release()


def _ensure_face_landmarker_model() -> Path:
    """Ensure the MediaPipe FaceLandmarker task model exists locally.

    Downloads the official model if missing.
    """

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "face_landmarker.task"

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

    import urllib.request

    tmp_path = model_path.with_suffix(model_path.suffix + ".download")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    tmp_path.write_bytes(data)
    tmp_path.replace(model_path)
    return model_path


def _landmark_xy(landmarks, idx: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return float(lm.x), float(lm.y)


def _mean_xy(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _normalize_point(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Return p in the coordinate frame where a=(0,0) and b=(1,1).

    For stability we normalize x by horizontal eye span and y by vertical span
    between top/bottom eyelid points.
    """

    px, py = p
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    # Avoid division by tiny numbers (closed eye / bad detection)
    if abs(dx) < 1e-6:
        dx = 1e-6
    if abs(dy) < 1e-6:
        dy = 1e-6

    return (px - ax) / dx, (py - ay) / dy


def _extract_gaze_features(
    face_landmarks,
) -> GazeFeatures:
    """Compute gaze features from iris centers + head pose.

    MediaPipe FaceMesh landmark indices (with iris refinement):
    - Left eye corners: 33 (outer), 133 (inner)
    - Left eye top/bottom: 159 (top), 145 (bottom)
    - Left iris: 468-471
    - Right eye corners: 362 (outer), 263 (inner)
    - Right eye top/bottom: 386 (top), 374 (bottom)
    - Right iris: 473-476

    NOTE: Indices are from the standard MediaPipe FaceMesh topology.
    """

    left_outer = _landmark_xy(face_landmarks, 33)
    left_inner = _landmark_xy(face_landmarks, 133)
    left_top = _landmark_xy(face_landmarks, 159)
    left_bottom = _landmark_xy(face_landmarks, 145)
    left_iris = _mean_xy([_landmark_xy(face_landmarks, i) for i in (468, 469, 470, 471)])

    right_outer = _landmark_xy(face_landmarks, 362)
    right_inner = _landmark_xy(face_landmarks, 263)
    right_top = _landmark_xy(face_landmarks, 386)
    right_bottom = _landmark_xy(face_landmarks, 374)
    right_iris = _mean_xy([_landmark_xy(face_landmarks, i) for i in (473, 474, 475, 476)])

    left_x, left_y = _normalize_point(left_iris, left_outer, left_inner)
    right_x, right_y = _normalize_point(right_iris, right_outer, right_inner)

    # Normalize y using eyelid span instead of corner span for better vertical stability.
    left_x, left_y = _normalize_point(left_iris, left_outer, (left_inner[0], left_bottom[1]))
    right_x, right_y = _normalize_point(right_iris, right_outer, (right_inner[0], right_bottom[1]))

    # Clamp iris features a bit to avoid wild values when one eye is occluded.
    def clamp(v: float, lo: float = -0.5, hi: float = 1.5) -> float:
        return float(max(lo, min(hi, v)))

    return GazeFeatures(
        left_x=clamp(left_x),
        left_y=clamp(left_y),
        right_x=clamp(right_x),
        right_y=clamp(right_y),
    )


def _resize_keep_aspect(frame_bgr, target_w: int):
    try:
        import cv2  # type: ignore

        h, w = frame_bgr.shape[:2]
        if w <= 0 or h <= 0:
            return None
        target_h = max(1, int(h * (target_w / w)))
        return cv2.resize(frame_bgr, (int(target_w), int(target_h)))
    except Exception:
        return None


def _make_eye_crop(frame_bgr, face_landmarks, idxs: tuple[int, ...], margin: float = 0.35):
    """Crop around an eye using a set of landmark indices."""

    h, w = frame_bgr.shape[:2]
    xs = [face_landmarks[i].x for i in idxs]
    ys = [face_landmarks[i].y for i in idxs]

    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))

    mx = (max_x - min_x) * margin
    my = (max_y - min_y) * margin

    min_x = max(0.0, min_x - mx)
    max_x = min(1.0, max_x + mx)
    min_y = max(0.0, min_y - my)
    max_y = min(1.0, max_y + my)

    x1 = int(min_x * w)
    x2 = int(max_x * w)
    y1 = int(min_y * h)
    y2 = int(max_y * h)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def _draw_iris_dot(eye_bgr, crop_bounds: tuple[int, int, int, int], face_landmarks, iris_idxs: tuple[int, ...], frame_w: int, frame_h: int):
    """Draw a highlighted dot for iris center on an eye crop."""

    try:
        import cv2  # type: ignore

        x1, y1, x2, y2 = crop_bounds

        # Iris center in full-frame pixel coordinates.
        cx = (sum(face_landmarks[i].x for i in iris_idxs) / len(iris_idxs)) * frame_w
        cy = (sum(face_landmarks[i].y for i in iris_idxs) / len(iris_idxs)) * frame_h

        h, w = eye_bgr.shape[:2]

        # Map iris center into crop pixel coordinates.
        px = int((cx - x1) * (w / max(1, (x2 - x1))))
        py = int((cy - y1) * (h / max(1, (y2 - y1))))
        px = int(max(0, min(w - 1, px)))
        py = int(max(0, min(h - 1, py)))

        cv2.circle(eye_bgr, (px, py), 10, (0, 0, 255), thickness=2)
        cv2.circle(eye_bgr, (px, py), 3, (0, 0, 255), thickness=-1)
        cv2.line(eye_bgr, (px - 14, py), (px + 14, py), (0, 0, 255), 1)
        cv2.line(eye_bgr, (px, py - 14), (px, py + 14), (0, 0, 255), 1)
    except Exception:
        return


def _make_eye_only_preview(frame_bgr, face_landmarks, target_w: int):
    """Create an eye-only preview (bigger eye crops for readability)."""

    try:
        import cv2  # type: ignore

        frame_h, frame_w = frame_bgr.shape[:2]

        # Use a larger margin so the crop stays stable and readable.
        left_eye_pack = _make_eye_crop(frame_bgr, face_landmarks, (33, 133, 159, 145, 468, 469, 470, 471), margin=0.65)
        right_eye_pack = _make_eye_crop(frame_bgr, face_landmarks, (362, 263, 386, 374, 473, 474, 475, 476), margin=0.65)

        left_eye = None
        right_eye = None
        if left_eye_pack is not None:
            left_eye_raw, left_bounds = left_eye_pack
            _draw_iris_dot(left_eye_raw, left_bounds, face_landmarks, (468, 469, 470, 471), frame_w, frame_h)
            left_eye = _resize_keep_aspect(left_eye_raw, max(1, int(target_w * 1.35)))
        if right_eye_pack is not None:
            right_eye_raw, right_bounds = right_eye_pack
            _draw_iris_dot(right_eye_raw, right_bounds, face_landmarks, (473, 474, 475, 476), frame_w, frame_h)
            right_eye = _resize_keep_aspect(right_eye_raw, max(1, int(target_w * 1.35)))

        if left_eye is not None:
            cv2.putText(left_eye, "left eye", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if right_eye is not None:
            cv2.putText(right_eye, "right eye", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if left_eye is not None and right_eye is not None:
            return cv2.vconcat([left_eye, right_eye])
        if left_eye is not None:
            return left_eye
        if right_eye is not None:
            return right_eye

        return _resize_keep_aspect(frame_bgr, target_w)
    except Exception:
        return _resize_keep_aspect(frame_bgr, target_w)


def _annotate_preview(preview_bgr, face_landmarks) -> None:
    """Draw a small set of useful landmarks onto a resized BGR preview frame."""

    import cv2  # type: ignore

    h, w = preview_bgr.shape[:2]

    def draw_point(idx: int, color: tuple[int, int, int], r: int = 2) -> None:
        lm = face_landmarks[idx]
        x = int(max(0, min(w - 1, lm.x * w)))
        y = int(max(0, min(h - 1, lm.y * h)))
        cv2.circle(preview_bgr, (x, y), r, color, thickness=-1)

    # A few stable reference points
    for idx in (1, 4, 19, 94):
        draw_point(idx, (255, 255, 0), r=2)  # cyan-ish

    # Eye corners / eyelids
    for idx in (33, 133, 159, 145):
        draw_point(idx, (0, 255, 0), r=2)  # green
    for idx in (362, 263, 386, 374):
        draw_point(idx, (0, 255, 0), r=2)

    # Iris landmarks
    for idx in (468, 469, 470, 471):
        draw_point(idx, (0, 0, 255), r=2)  # red
    for idx in (473, 474, 475, 476):
        draw_point(idx, (0, 0, 255), r=2)

    cv2.putText(
        preview_bgr,
        "landmarks: on",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
