# Developer Guide

---

## Setup

```bash
# Clone and install dependencies
pip install streamlit mediapipe opencv-python pandas numpy matplotlib seaborn pyyaml pytest pytest-cov
```

Run the app:
```bash
streamlit run app_refactored.py
```

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=rehabilitationcore --cov=config --cov-report=term-missing
```

---

## Adding a New Exercise

All exercise definitions live in `config/exercises.yaml`. No Python changes needed.

**Step 1** — Add an entry to `config/exercises.yaml`:
```yaml
exercises:
  5:
    name: "Shoulder Rotation"
    description: "External shoulder rotation assessment"
    landmarks: [12, 14, 16, 24]  # Shoulder, Elbow, Wrist, Hip
    thresholds:
      shoulder:
        type: "minimum"
        value: 45.0
        feedback_pass: "✅ Rotation: PASS"
        feedback_fail: "❌ FAIL: Rotate Further!"
```

Threshold `type` options:

| Type | Behaviour | Required YAML keys |
|---|---|---|
| `minimum` | FAIL if angle < value | `value` |
| `maximum` | FAIL if angle > value | `value` |
| `range` | FAIL outside [min, max]; TRANSITIONING near target | `min`, `max`, `target` |

**Step 2** — Restart the app (or re-import the module). `EXERCISES` is built from YAML at import time, so any running Python process needs a restart.

**Step 3** — Add a test:
```python
def test_exercise_5_shoulder_rotation():
    ex = EXERCISES[5]
    assert ex.name == "Shoulder Rotation"
    assert ex.angle_thresholds["shoulder"].min_value == 45.0
```

> **Note:** The analyzer currently computes `shoulder` (hip→shoulder→elbow) and `elbow` (shoulder→elbow→wrist) angles. If your exercise needs a different angle type (e.g. `wrist`, `hand_open`), the exercise will load correctly but the analyzer will return `TRACKING`. Extending `_calculate_angles` in `analyzer.py` is the path to support new angle types.

---

## Testing Patterns

### Unit testing a pure function
```python
from rehabilitationcore.biomechanics import calculate_2d_angle

def test_straight_arm():
    angle = calculate_2d_angle((0.5, 0.3), (0.5, 0.5), (0.5, 0.7))
    assert abs(angle - 180.0) < 0.1
```

### Testing analyzer PASS/FAIL with mock landmarks
```python
from rehabilitationcore.models import Landmark, ExerciseStatus
from rehabilitationcore.analyzer import ExerciseAnalyzer
from rehabilitationcore.exercises import EXERCISES

def create_straight_arm():
    landmarks = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
    landmarks[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)  # Hip
    landmarks[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)  # Shoulder
    landmarks[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)  # Elbow
    landmarks[16] = Landmark(x=0.5, y=0.7, z=0, visibility=0.9)  # Wrist
    return landmarks

def test_exercise2_pass():
    analyzer = ExerciseAnalyzer()
    result = analyzer.analyze(create_straight_arm(), EXERCISES[2])
    assert result.status == ExerciseStatus.PASS
```

### Testing error paths
```python
from rehabilitationcore.errors import ExerciseNotFoundError
from config.loader import ConfigManager

def test_invalid_exercise_raises():
    cfg = ConfigManager()
    with pytest.raises(ExerciseNotFoundError) as exc:
        cfg.get_exercise(999)
    assert exc.value.exercise_id == 999
```

### Testing that the config loads a new exercise
```python
from rehabilitationcore.exercises import EXERCISES

def test_new_exercise_loaded():
    assert 5 in EXERCISES
    assert EXERCISES[5].name == "Shoulder Rotation"
```

---

## Module Responsibilities (short form)

| Module | Owns | Does NOT own |
|---|---|---|
| `biomechanics.py` | Angle math, smoothing, validation | Exercise semantics |
| `exercises.py` | Exercise registry | How angles are calculated |
| `analyzer.py` | PASS/FAIL evaluation, confidence | Rendering, file I/O |
| `video/__init__.py` | MediaPipe pose extraction | Angle calculation |
| `video/calculator.py` | Landmarks → DataFrame, angle columns | UI display |
| `video/renderer.py` | OpenCV drawing | Any analysis logic |
| `config/loader.py` | YAML loading, typed access | Exercise logic |

---

## Logging

Every module should use the namespaced logger:

```python
from rehabilitationcore.logging_config import get_logger
logger = get_logger("my_module")  # logs as "rehabilitation.my_module"
```

Call `configure_logging()` once at app startup to attach the console handler. In tests, use `caplog`:

```python
def test_warning_logged(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="rehabilitation"):
        # ... trigger the warning
    assert any("expected message" in r.message for r in caplog.records)
```

---

## File Structure

```
FYP2/
├── app_refactored.py          main Streamlit UI
├── config/
│   ├── exercises.yaml         exercise definitions (edit here to change thresholds)
│   ├── system.yaml            MediaPipe / smoothing / visualisation settings
│   └── loader.py              ConfigManager
├── rehabilitationcore/
│   ├── models.py              data classes
│   ├── biomechanics.py        pure math functions
│   ├── exercises.py           exercise registry
│   ├── analyzer.py            ExerciseAnalyzer
│   ├── errors.py              exception classes
│   └── logging_config.py      get_logger, configure_logging
├── video/
│   ├── __init__.py            PoseExtractionPipeline
│   ├── calculator.py          KinematicCalculator
│   └── renderer.py            VideoRenderer
├── tests/
│   └── unit/
│       ├── test_biomechanics.py
│       ├── test_analyzer.py
│       ├── test_config.py
│       └── test_errors.py
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    ├── DEVELOPMENT.md
    └── USER_GUIDE.md
```
