from __future__ import annotations


def test_imports_smoke() -> None:
    # Import all public modules to ensure basic import-time correctness.
    import eye_tracker_app  # noqa: F401
    from eye_tracker_app import run_demo  # noqa: F401
    from eye_tracker_app import calibration_gui  # noqa: F401
    from eye_tracker_app import dummy_overlay_gui  # noqa: F401
    from eye_tracker_app import camera_gaze  # noqa: F401
