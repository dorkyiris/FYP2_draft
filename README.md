# FYP 2: Vision-Based Tele-Rehabilitation System

Upper-limb stroke rehabilitation assessment using MediaPipe pose estimation and biomechanical angle analysis.

---

## Quick Start

```bash
pip install streamlit mediapipe opencv-python pandas numpy matplotlib seaborn pyyaml pytest
streamlit run app_refactored.py
```

---

## What It Does

The system analyses four clinical upper-limb exercises from video or a live webcam:

| # | Exercise | Pass Condition |
|---|---|---|
| 1 | Lifting an object | Shoulder angle ≥ 90° |
| 2 | Extending the elbow | Elbow angle ≥ 160° |
| 3 | Lifting the wrist | Wrist angle ≥ 15° |
| 4 | Opening the hand | Hand open ≥ 45° |

For each frame, the system extracts 33 body landmarks via MediaPipe, computes joint angles, evaluates clinical thresholds, and overlays real-time feedback on the video.

---

## Project Structure

```
FYP2/
├── app_refactored.py          Streamlit dashboard (4 modes)
├── config/
│   ├── exercises.yaml         Clinical exercise definitions and thresholds
│   ├── system.yaml            MediaPipe / smoothing / visualisation settings
│   └── loader.py              ConfigManager
├── rehabilitationcore/        Core engine — no UI/video dependencies
│   ├── models.py
│   ├── biomechanics.py
│   ├── exercises.py
│   ├── analyzer.py
│   ├── errors.py
│   └── logging_config.py
├── video/                     MediaPipe extraction, OpenCV rendering
├── tests/unit/                67 unit tests (pytest)
└── docs/                      Architecture, API, Dev guide, User guide
```

---

## Tests

```bash
pytest tests/ -v                          # run all 67 tests
pytest tests/ --cov=rehabilitationcore    # with coverage
```

---

## Changing Exercise Thresholds

Edit `config/exercises.yaml` — no code change needed. Restart the app after saving.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module graph, data flow, design decisions |
| [docs/API.md](docs/API.md) | Public classes and functions reference |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Adding exercises, testing patterns, file structure |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Running the app, modes, troubleshooting |

---

## Implementation Status

| Phase | Description | Status |
|---|---|---|
| 1 | Modularise monolith → testable core | Done |
| 2 | YAML configuration management | Done |
| 3 | Error handling and logging | Done |
| 4 | Documentation | Done |
| 5 | Expanded test infrastructure | Next |
| 6 | REST API | Planned |
| 7 | Docker + CI/CD | Planned |

---

**Research:** Nandana et al. 2026 — 91.7% classification accuracy (Ex1), 25.5% jitter reduction via EMA smoothing (ablation study).
