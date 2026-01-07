from __future__ import annotations

# Allow `python eye_tracker_app/run_demo.py` to work by ensuring the
# project root is on sys.path. (Recommended usage is still `python -m eye_tracker_app.run_demo`.)
if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eye_tracker_app.calibration_gui import run_calibration
from eye_tracker_app.dummy_overlay_gui import run_dummy_overlay


def main() -> None:
    camera_index = 0
    show_preview = True
    calibration = run_calibration(camera_index=camera_index, show_preview=show_preview)
    if calibration is None:
        return

    run_dummy_overlay(calibration, camera_index=camera_index, show_preview=show_preview)


if __name__ == "__main__":
    main()
