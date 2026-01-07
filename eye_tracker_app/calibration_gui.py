from __future__ import annotations

# Allow `python eye_tracker_app/calibration_gui.py` to work by ensuring the
# project root is on sys.path. (Normal usage is still `python -m eye_tracker_app.run_demo`.)
if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Optional

import numpy as np

import tkinter as tk

from eye_tracker_app.camera_gaze import CameraGaze, GazeFeatures
from eye_tracker_app.camera_preview import CameraPreview


@dataclass(frozen=True)
class CalibrationResult:
    screen_width: int
    screen_height: int
    points: list[tuple[int, int]]
    # Linear map from [1, *features] -> screen coords (least squares).
    coef_x: list[float]
    coef_y: list[float]
    # Face calibration (head pose) normalization.
    # Normalized pose: (pose - center) / range
    pose_center_yaw: float
    pose_center_pitch: float
    pose_center_roll: float
    pose_range_yaw: float
    pose_range_pitch: float
    pose_range_roll: float


def _make_calibration_points(width: int, height: int) -> list[tuple[int, int]]:
    # 3x3 grid: left/center/right x top/center/bottom
    xs = [int(width * 0.15), int(width * 0.50), int(width * 0.85)]
    ys = [int(height * 0.15), int(height * 0.50), int(height * 0.85)]
    return [(x, y) for y in ys for x in xs]


def run_calibration(camera_index: int = 0, show_preview: bool = True) -> Optional[CalibrationResult]:
    """Shows a full-screen calibration window with 9 points and records webcam eye features.

    User looks at each dot and presses Space to record a sample.

    Returns:
        CalibrationResult if completed, else None if aborted.
    """

    root = tk.Tk()
    root.title("Calibration (Dummy Points)")

    # Fullscreen for easier 'screen element' mapping.
    root.attributes("-fullscreen", True)
    root.configure(bg="black")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    points = _make_calibration_points(screen_width, screen_height)
    current_idx = 0
    dot_radius = 10

    gaze = CameraGaze(camera_index=camera_index)
    gaze.start()

    preview = CameraPreview(gaze)
    if show_preview:
        preview.start()

    samples: list[tuple[GazeFeatures, tuple[int, int]]] = []

    instructions_id = canvas.create_text(
        screen_width // 2,
        int(screen_height * 0.06),
        fill="white",
        font=("Segoe UI", 18),
        text="Calibration: Look at the dot and press SPACE to record (ESC to quit)",
    )

    status_id = canvas.create_text(
        screen_width // 2,
        int(screen_height * 0.10),
        fill="white",
        font=("Segoe UI", 14),
        text="Starting camera…",
    )

    dot_id: Optional[int] = None

    # --- Face calibration (head pose) ---
    face_steps = [
        ("Look straight (center) and press SPACE", "center"),
        ("Turn head LEFT and press SPACE", "left"),
        ("Turn head RIGHT and press SPACE", "right"),
        ("Tilt head UP and press SPACE", "up"),
        ("Tilt head DOWN and press SPACE", "down"),
    ]
    face_step_idx = 0
    face_samples: dict[str, tuple[float, float, float]] = {}

    def draw_current_point() -> None:
        nonlocal dot_id
        canvas.delete("dot")
        if current_idx >= len(points):
            return

        x, y = points[current_idx]
        dot_id = canvas.create_oval(
            x - dot_radius,
            y - dot_radius,
            x + dot_radius,
            y + dot_radius,
            fill="white",
            outline="",
            tags=("dot",),
        )

        canvas.itemconfigure(
            instructions_id,
            text=f"Calibration: Point {current_idx + 1}/{len(points)} — press SPACE to record (ESC to quit)",
        )

    def draw_face_step() -> None:
        canvas.delete("dot")
        canvas.itemconfigure(
            instructions_id,
            text=f"Face calibration ({face_step_idx + 1}/{len(face_steps)}): {face_steps[face_step_idx][0]} (ESC to quit)",
        )

    def refresh_status() -> None:
        features, err = gaze.get_latest()
        if err is not None:
            canvas.itemconfigure(status_id, text=f"Camera status: {err}")
        elif features is None:
            canvas.itemconfigure(status_id, text="Camera status: (no features yet)")
        else:
            canvas.itemconfigure(status_id, text="Camera status: tracking")
        root.after(100, refresh_status)

    def on_escape(_event: tk.Event) -> None:
        preview.stop()
        gaze.stop()
        root.destroy()

    def on_space(_event: tk.Event) -> None:
        nonlocal current_idx
        nonlocal face_step_idx

        features, err = gaze.get_latest()
        if err is not None or features is None:
            canvas.itemconfigure(status_id, text=f"Can't record yet: {err or 'no features'}")
            return

        # First: face calibration steps (head pose range capture)
        if face_step_idx < len(face_steps):
            key = face_steps[face_step_idx][1]
            face_samples[key] = (features.yaw_deg, features.pitch_deg, features.roll_deg)
            face_step_idx += 1
            if face_step_idx < len(face_steps):
                draw_face_step()
                return

            # After face calibration completes, move to point calibration.
            draw_current_point()
            return

        samples.append((features, points[current_idx]))

        current_idx += 1
        if current_idx >= len(points):
            preview.stop()
            gaze.stop()
            root.destroy()
            return
        draw_current_point()

    root.bind("<Escape>", on_escape)
    root.bind("<space>", on_space)

    draw_face_step()
    refresh_status()
    root.mainloop()

    preview.stop()
    gaze.stop()

    if current_idx < len(points):
        return None

    if len(samples) < 5:
        return None

    # Compute pose normalization from face calibration.
    if "center" not in face_samples:
        return None

    cy, cp, cr = face_samples["center"]
    # Ranges are based on opposite directions; fall back to safe defaults.
    yaw_left = face_samples.get("left", (cy - 15.0, cp, cr))[0]
    yaw_right = face_samples.get("right", (cy + 15.0, cp, cr))[0]
    pitch_up = face_samples.get("up", (cy, cp - 10.0, cr))[1]
    pitch_down = face_samples.get("down", (cy, cp + 10.0, cr))[1]

    range_yaw = max(5.0, abs(yaw_right - yaw_left) / 2.0)
    range_pitch = max(5.0, abs(pitch_down - pitch_up) / 2.0)
    range_roll = 10.0

    # Normalize pose in samples before fitting so the mapping is less sensitive
    # to user-specific head pose offsets.
    norm_samples: list[tuple[GazeFeatures, tuple[int, int]]] = []
    for f, pt in samples:
        norm_samples.append((_normalize_features(f, cy, cp, cr, range_yaw, range_pitch, range_roll), pt))

    coef_x, coef_y = _fit_linear_mapping(norm_samples)

    return CalibrationResult(
        screen_width=screen_width,
        screen_height=screen_height,
        points=points,
        coef_x=coef_x,
        coef_y=coef_y,
        pose_center_yaw=float(cy),
        pose_center_pitch=float(cp),
        pose_center_roll=float(cr),
        pose_range_yaw=float(range_yaw),
        pose_range_pitch=float(range_pitch),
        pose_range_roll=float(range_roll),
    )


def _normalize_features(
    f: GazeFeatures,
    cy: float,
    cp: float,
    cr: float,
    ry: float,
    rp: float,
    rr: float,
) -> GazeFeatures:
    def safe_div(a: float, b: float) -> float:
        return float(a / b) if abs(b) > 1e-6 else 0.0

    return GazeFeatures(
        left_x=f.left_x,
        left_y=f.left_y,
        right_x=f.right_x,
        right_y=f.right_y,
        yaw_deg=safe_div(f.yaw_deg - cy, ry),
        pitch_deg=safe_div(f.pitch_deg - cp, rp),
        roll_deg=safe_div(f.roll_deg - cr, rr),
        face_scale=f.face_scale,
    )


def _fit_linear_mapping(samples: list[tuple[GazeFeatures, tuple[int, int]]]) -> tuple[list[float], list[float]]:
    """Fit x,y screen coordinates as a linear function of gaze features."""

    A_rows: list[list[float]] = []
    xs: list[float] = []
    ys: list[float] = []

    for features, (sx, sy) in samples:
        A_rows.append([1.0, *features.as_list()])
        xs.append(float(sx))
        ys.append(float(sy))

    A = np.asarray(A_rows, dtype=np.float64)
    bx = np.asarray(xs, dtype=np.float64)
    by = np.asarray(ys, dtype=np.float64)

    coef_x, *_ = np.linalg.lstsq(A, bx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, by, rcond=None)

    return coef_x.tolist(), coef_y.tolist()
