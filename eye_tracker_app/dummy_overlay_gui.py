from __future__ import annotations

# Allow `python eye_tracker_app/dummy_overlay_gui.py` to work by ensuring the
# project root is on sys.path. (Normal usage is still `python -m eye_tracker_app.run_demo`.)
if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass

import tkinter as tk

from eye_tracker_app.calibration_gui import CalibrationResult
from eye_tracker_app.camera_gaze import CameraGaze
from eye_tracker_app.camera_preview import CameraPreview


@dataclass(frozen=True)
class TargetRegion:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def run_dummy_overlay(calibration: CalibrationResult, camera_index: int = 0, show_preview: bool = True) -> None:
    """Second GUI: overlay + highlighting driven by webcam gaze prediction."""

    root = tk.Tk()
    root.title("Overlay (Dummy Gaze)")

    root.attributes("-fullscreen", True)
    root.configure(bg="black")

    screen_width = calibration.screen_width
    screen_height = calibration.screen_height

    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # Define a few dummy screen elements to highlight.
    targets = [
        TargetRegion("Button A", int(screen_width * 0.10), int(screen_height * 0.25), int(screen_width * 0.32), int(screen_height * 0.40)),
        TargetRegion("Button B", int(screen_width * 0.38), int(screen_height * 0.25), int(screen_width * 0.60), int(screen_height * 0.40)),
        TargetRegion("Input", int(screen_width * 0.10), int(screen_height * 0.50), int(screen_width * 0.60), int(screen_height * 0.62)),
        TargetRegion("Submit", int(screen_width * 0.65), int(screen_height * 0.50), int(screen_width * 0.85), int(screen_height * 0.62)),
    ]

    header_id = canvas.create_text(
        screen_width // 2,
        int(screen_height * 0.06),
        fill="white",
        font=("Segoe UI", 18),
        text="Overlay: gaze dot from camera calibration (ESC to quit)",
    )

    status_id = canvas.create_text(
        screen_width // 2,
        int(screen_height * 0.10),
        fill="white",
        font=("Segoe UI", 14),
        text="Starting camera…",
    )

    # Draw targets and keep ids so we can recolor on hover.
    target_rect_ids: dict[str, int] = {}
    target_text_ids: dict[str, int] = {}

    for target in targets:
        rect_id = canvas.create_rectangle(target.x1, target.y1, target.x2, target.y2, outline="white", width=3)
        text_id = canvas.create_text(
            (target.x1 + target.x2) // 2,
            (target.y1 + target.y2) // 2,
            fill="white",
            font=("Segoe UI", 16),
            text=target.name,
        )
        target_rect_ids[target.name] = rect_id
        target_text_ids[target.name] = text_id

    gaze = CameraGaze(camera_index=camera_index)
    gaze.start()

    preview = CameraPreview(gaze)
    if show_preview:
        preview.start()

    def on_escape(_event: tk.Event) -> None:
        preview.stop()
        gaze.stop()
        root.destroy()

    def draw_heatmap(cx: int, cy: int) -> None:
        canvas.delete("heatmap")

        # Simple heatmap look using concentric circles (Tkinter has no alpha by default)
        rings = [
            (80, "#330000"),
            (60, "#660000"),
            (40, "#990000"),
            (25, "#CC0000"),
            (12, "#FF0000"),
        ]

        for r, color in rings:
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=4, tags=("heatmap",))

        # Tiny center mark
        canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#FF0000", outline="", tags=("heatmap",))

    def set_gaze(x: int, y: int) -> None:
        active: str | None = None
        for target in targets:
            if target.contains(x, y):
                active = target.name
                break

        for target in targets:
            is_active = target.name == active
            rect_id = target_rect_ids[target.name]
            text_id = target_text_ids[target.name]
            canvas.itemconfigure(rect_id, outline=("yellow" if is_active else "white"))
            canvas.itemconfigure(text_id, fill=("yellow" if is_active else "white"))

    def _predict_axis(coef: list[float], features_list: list[float]) -> float:
        # coef length should be 1 + len(features_list)
        v = [1.0, *features_list]
        n = min(len(v), len(coef))
        return float(sum(v[i] * coef[i] for i in range(n)))

    def predict_and_draw() -> None:
        features, err = gaze.get_latest()
        if err is not None or features is None:
            canvas.itemconfigure(status_id, text=f"Camera status: {err or 'no face'}")
            root.after(33, predict_and_draw)
            return

        # Linear prediction using calibration coefficients.
        f = features.as_list()

        x = _predict_axis(calibration.coef_x, f)
        y = _predict_axis(calibration.coef_y, f)

        # Clamp to screen bounds.
        xi = int(max(0, min(screen_width - 1, x)))
        yi = int(max(0, min(screen_height - 1, y)))

        canvas.itemconfigure(status_id, text="Camera status: tracking")
        draw_heatmap(xi, yi)
        set_gaze(xi, yi)
        root.after(33, predict_and_draw)

    root.bind("<Escape>", on_escape)

    # Start with gaze at center and start prediction loop.
    draw_heatmap(screen_width // 2, screen_height // 2)
    set_gaze(screen_width // 2, screen_height // 2)
    predict_and_draw()

    root.mainloop()

    preview.stop()
    gaze.stop()
