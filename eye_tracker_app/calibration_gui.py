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


def _make_calibration_points(width: int, height: int) -> list[tuple[int, int]]:
    # 3x3 grid plus the 4 extreme corners.
    xs = [int(width * 0.15), int(width * 0.50), int(width * 0.85)]
    ys = [int(height * 0.15), int(height * 0.50), int(height * 0.85)]

    corner_pad_x = int(width * 0.02)
    corner_pad_y = int(height * 0.02)
    corners = [
        (corner_pad_x, corner_pad_y),
        (width - 1 - corner_pad_x, corner_pad_y),
        (corner_pad_x, height - 1 - corner_pad_y),
        (width - 1 - corner_pad_x, height - 1 - corner_pad_y),
    ]

    pts = [(x, y) for y in ys for x in xs] + corners
    # Preserve order but dedupe in case of tiny screens.
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for p in pts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


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

        features, err = gaze.get_latest()
        if err is not None or features is None:
            canvas.itemconfigure(status_id, text=f"Can't record yet: {err or 'no features'}")
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

    draw_current_point()
    refresh_status()
    root.mainloop()

    preview.stop()
    gaze.stop()

    if current_idx < len(points):
        return None

    if len(samples) < 5:
        return None

    coef_x, coef_y = _fit_linear_mapping(samples)

    return CalibrationResult(
        screen_width=screen_width,
        screen_height=screen_height,
        points=points,
        coef_x=coef_x,
        coef_y=coef_y,
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
