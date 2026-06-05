# API Reference

All public classes and functions in `rehabilitationcore/` and `config/`.

---

## rehabilitationcore.models

### `ExerciseStatus`
Enum with four values:

| Value | Meaning |
|---|---|
| `PASS` | Angle meets the clinical threshold |
| `FAIL` | Angle does not meet the threshold |
| `TRANSITIONING` | Angle is between target and threshold (range exercises) |
| `TRACKING` | Landmark visibility too low to evaluate |

---

### `Landmark`
Frozen dataclass representing a single MediaPipe pose landmark.

```python
@dataclass(frozen=True)
class Landmark:
    x: float          # normalised 0-1 horizontal position
    y: float          # normalised 0-1 vertical position
    z: float = 0.0    # depth (optional)
    visibility: float = 1.0

    def to_tuple(self) -> tuple        # returns (x, y)
    def is_visible(self, threshold: float = 0.65) -> bool
```

---

### `AngleThreshold`
Frozen dataclass for a single angle's pass/fail criteria.

```python
@dataclass(frozen=True)
class AngleThreshold:
    name: str
    min_value: Optional[float] = None   # FAIL if angle < min_value
    max_value: Optional[float] = None   # FAIL if angle > max_value
    target_value: Optional[float] = None
    feedback_pass: str = "✅ PASS"
    feedback_fail: str = "❌ FAIL"

    def evaluate(self, angle: float) -> ExerciseStatus
```

---

### `ExerciseResult`
Frozen dataclass returned by `ExerciseAnalyzer.analyze()`.

```python
@dataclass(frozen=True)
class ExerciseResult:
    exercise_id: int
    exercise_name: str
    status: ExerciseStatus
    primary_angle: float
    secondary_angle: Optional[float] = None
    angles: Dict[str, float] = None     # all computed angles
    feedback: str = ""
    confidence: float = 1.0             # mean landmark visibility
    frame_number: Optional[int] = None
```

---

## rehabilitationcore.biomechanics

### `calculate_2d_angle(p1, p2, p3) → float`
Returns the angle **at p2** formed by the three 2D points, in degrees `[0, 180]`.

```python
from rehabilitationcore import calculate_2d_angle

# Elbow angle: shoulder → elbow ← wrist
angle = calculate_2d_angle(
    (shoulder_x, shoulder_y),
    (elbow_x, elbow_y),
    (wrist_x, wrist_y),
)
```

### `smooth_signal(values, method="ema", span=3) → List[float]`
Smooths a list of angle measurements. `method` is `"ema"` or `"sma"`. `span=3` is the research-validated default.

### `validate_landmark_chain(landmarks, required_indices, min_visibility=0.65) → (bool, str)`
Checks that all required landmark indices exist and meet the visibility threshold. Returns `(True, "")` on success, `(False, reason)` on failure.

---

## rehabilitationcore.exercises

### `get_exercise(exercise_id: int) → ExerciseDefinition`
Looks up an exercise by ID. Raises `ExerciseNotFoundError` if the ID is not in the registry.

```python
from rehabilitationcore import get_exercise

ex = get_exercise(1)   # "Lifting an object"
print(ex.name, ex.landmarks_required)
```

### `list_exercises() → List[Tuple[int, str, str]]`
Returns a list of `(id, name, description)` for all registered exercises.

### `EXERCISES: Dict[int, ExerciseDefinition]`
The global exercise registry, built from `config/exercises.yaml` at module import.

---

## rehabilitationcore.analyzer

### `ExerciseAnalyzer(min_visibility=0.65)`

#### `analyze(landmarks, exercise, frame_number=None) → ExerciseResult`
Analyses a single frame. Always returns a result — never raises. Returns `TRACKING` status when landmarks are insufficient.

```python
from rehabilitationcore import ExerciseAnalyzer, get_exercise

analyzer = ExerciseAnalyzer()
result = analyzer.analyze(landmarks, get_exercise(2))

print(result.status)        # ExerciseStatus.PASS
print(result.primary_angle) # 163.4
print(result.feedback)      # "✅ Extension: PASS (Arm Straight) (163.4°)"
```

#### `analyze_sequence(landmark_sequence, exercise) → List[ExerciseResult]`
Analyses a list of frames. Frame numbers are assigned automatically (0-indexed).

```python
results = analyzer.analyze_sequence(landmark_frames, get_exercise(1))
pass_count = sum(1 for r in results if r.status.value == "PASS")
```

---

## rehabilitationcore.errors

```python
RehabSystemError          # base — catch-all for this library
├── ConfigError           # config file missing or malformed
│   └── ExerciseNotFoundError(exercise_id, available=[])
├── LandmarkError(message, landmark_idx=None, visibility=None)
└── AnalysisError         # unexpected pipeline failure
```

```python
from rehabilitationcore.errors import ExerciseNotFoundError

try:
    ex = get_exercise(99)
except ExerciseNotFoundError as e:
    print(e.exercise_id)   # 99
    print(e.available)     # [1, 2, 3, 4]
```

---

## rehabilitationcore.logging_config

### `get_logger(name: str) → logging.Logger`
Returns a logger named `rehabilitation.<name>`. Use this in any module that needs structured logging.

```python
from rehabilitationcore.logging_config import get_logger
logger = get_logger("my_module")
logger.info("Processing frame %d", frame_num)
```

### `configure_logging(level=logging.INFO) → None`
Attaches a console handler to the root `rehabilitation` logger. Idempotent — safe to call multiple times.

---

## config.loader

### `ConfigManager(config_dir="config")`

| Method | Returns | Raises |
|---|---|---|
| `get_exercise(id)` | `Dict` | `ExerciseNotFoundError` |
| `get_threshold(id, angle_name)` | `Dict` | `ExerciseNotFoundError`, `ValueError` |

```python
from config.loader import ConfigManager

cfg = ConfigManager()
ex1 = cfg.get_exercise(1)
print(ex1["name"])                         # "Lifting an object"
print(ex1["thresholds"]["shoulder"]["value"])  # 90.0
```

---

## video module

### `PoseExtractionPipeline` (context manager)
```python
from video import PoseExtractionPipeline

with PoseExtractionPipeline(min_detection_confidence=0.65) as pipeline:
    landmarks = pipeline.extract_frame(cv2_frame)     # List[Landmark]
    all_frames = pipeline.extract_video("clip.mp4")   # List[List[Landmark]]
```

### `KinematicCalculator` (static methods)
```python
from video.calculator import KinematicCalculator

df = KinematicCalculator.landmarks_to_dataframe(landmark_frames)
angles_df = KinematicCalculator.extract_kinematic_angles(df)
# columns: Shoulder_Angle, Elbow_Angle (EMA-smoothed)
```
