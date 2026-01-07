from __future__ import annotations

import math

import numpy as np

from eye_tracker_app.calibration_gui import _fit_linear_mapping
from eye_tracker_app.camera_gaze import GazeFeatures


def _predict(coef: list[float], f: GazeFeatures) -> float:
    v = [1.0, *f.as_list()]
    n = min(len(v), len(coef))
    return float(sum(v[i] * coef[i] for i in range(n)))


def test_fit_linear_mapping_recovers_known_linear_model() -> None:
    # Create a known linear mapping from features -> screen coords.
    # coef length is 1 + len(GazeFeatures.as_list())
    true_x = np.array([120.0, 500.0, -250.0, 40.0, 10.0], dtype=float)
    true_y = np.array([80.0, -100.0, 300.0, 25.0, -60.0], dtype=float)

    rng = np.random.default_rng(123)

    samples: list[tuple[GazeFeatures, tuple[int, int]]] = []
    for _ in range(25):
        f = GazeFeatures(
            left_x=float(rng.uniform(-0.2, 1.2)),
            left_y=float(rng.uniform(-0.2, 1.2)),
            right_x=float(rng.uniform(-0.2, 1.2)),
            right_y=float(rng.uniform(-0.2, 1.2)),
        )
        vec = np.array([1.0, *f.as_list()], dtype=float)
        sx = float(vec @ true_x)
        sy = float(vec @ true_y)
        samples.append((f, (int(round(sx)), int(round(sy)))))

    coef_x, coef_y = _fit_linear_mapping(samples)

    # The recovered coefficients should be close to the true ones.
    # (Rounding to ints introduces some small error.)
    for got, exp in zip(coef_x, true_x.tolist()):
        assert math.isfinite(got)
        assert abs(got - exp) < 5.0

    for got, exp in zip(coef_y, true_y.tolist()):
        assert math.isfinite(got)
        assert abs(got - exp) < 5.0


def test_prediction_uses_5_params() -> None:
    f = GazeFeatures(
        left_x=0.1,
        left_y=0.2,
        right_x=0.3,
        right_y=0.4,
    )
    coef = [1.0] + [0.0] * len(f.as_list())
    assert abs(_predict(coef, f) - 1.0) < 1e-9
