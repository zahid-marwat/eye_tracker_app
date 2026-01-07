# Eye Tracker App (Proof of Concept)

This is a minimal proof-of-concept scaffolding for an eye-tracking project.

Right now it includes **two GUIs**:

1) **Calibration GUI (dummy points)**: shows a 3×3 grid of points. You "accept" each point by pressing **Space**.
2) **Overlay GUI (dummy gaze)**: shows simple on-screen targets (rectangles). A "gaze" dot follows your **mouse cursor** and highlights the target you hover.

## Run

Install deps:

```powershell
pip install -r requirements.txt
```

Install dev deps (tests):

```powershell
pip install -r requirements-dev.txt
```

From the repo root:

```powershell
python -m eye_tracker_app.run_demo
```

## Controls

### Calibration
- `Space`: accept current calibration point
- `Esc`: quit

### Overlay
- Move mouse: simulated gaze moves
- `Esc`: quit

## Notes

- This does **not** do real gaze estimation yet; it’s a GUI and flow scaffold.
- Next step is to replace the simulated gaze provider with webcam + face/eye landmark tracking.

## Tests

```powershell
pytest
```
