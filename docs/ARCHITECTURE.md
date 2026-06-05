# System Architecture

## Overview

The system is a **vision-based tele-rehabilitation platform** that analyses upper-limb stroke recovery exercises from video or a live webcam. The architecture is layered — UI, pipeline, core engine, and configuration are cleanly separated so that the biomechanical logic can be tested and reused independently of any UI framework.

---

## Module Dependency Graph

```
app_refactored.py (Streamlit UI)
    │
    ├── rehabilitationcore/       ← pure Python, no UI/video deps
    │   ├── models.py             ← data classes (Landmark, ExerciseResult …)
    │   ├── biomechanics.py       ← pure math (calculate_2d_angle, smooth_signal …)
    │   ├── exercises.py          ← exercise registry, built from config at import
    │   ├── analyzer.py           ← ExerciseAnalyzer (uses biomechanics + exercises)
    │   ├── errors.py             ← custom exception hierarchy
    │   └── logging_config.py     ← get_logger(), configure_logging()
    │
    ├── video/                    ← depends on rehabilitationcore + OpenCV/MediaPipe
    │   ├── __init__.py           ← PoseExtractionPipeline (MediaPipe wrapper)
    │   ├── calculator.py         ← KinematicCalculator (landmarks → DataFrame/angles)
    │   └── renderer.py           ← VideoRenderer (OpenCV drawing)
    │
    └── config/                   ← no Python logic deps; loaded at startup
        ├── exercises.yaml        ← exercise definitions and thresholds
        ├── system.yaml           ← MediaPipe, smoothing, visualisation defaults
        └── loader.py             ← ConfigManager (YAML → typed dicts)
```

No circular dependencies. Import order: `config → models → biomechanics → exercises → analyzer → video → app`.

---

## Data Flow: Single Frame Analysis

```
Camera / Video file
       │
       ▼
PoseExtractionPipeline.extract_frame(frame)
       │  MediaPipe returns 33 body landmarks
       ▼
List[Landmark]  (x, y, z, visibility per landmark)
       │
       ▼
ExerciseAnalyzer.analyze(landmarks, exercise)
       │
       ├─ validate_landmark_chain()   ← check required landmarks are visible
       ├─ _calculate_angles()         ← hip-shoulder-elbow (shoulder), shoulder-elbow-wrist (elbow)
       ├─ threshold.evaluate(angle)   ← PASS / FAIL / TRANSITIONING
       └─ compute confidence          ← mean visibility of required landmarks
       │
       ▼
ExerciseResult  (status, primary_angle, feedback, confidence, frame_number)
       │
       ▼
VideoRenderer.draw_clinical_overlay(frame, result)
       │  OpenCV draws skeleton + feedback text on the frame
       ▼
Annotated frame  →  displayed in Streamlit / written to output video
```

---

## Data Flow: CSV / Batch Analysis

```
CSV file  (pre-extracted landmarks, one row per frame)
       │
       ▼
KinematicCalculator.extract_kinematic_angles(df)
       │  computes shoulder + elbow angles, applies EMA smoothing (span=3)
       ▼
angles_df  (DataFrame: frame, Shoulder_Angle, Elbow_Angle)
       │
       ▼
ExerciseAnalyzer.analyze_sequence(landmark_frames, exercise)
       │
       ▼
List[ExerciseResult]  →  Streamlit plots / stats
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| `rehabilitationcore` has no UI/video imports | Makes it testable without a webcam or display |
| Exercises defined in YAML, not code | Clinical thresholds can be updated without a redeploy |
| `ExerciseAnalyzer` returns `TRACKING` instead of raising | Graceful degradation during real-time analysis |
| `validate_landmark_chain` returns `(bool, str)` tuple | Allows the analyzer to log and branch without exceptions in the hot path |
| EMA span=3 | Ablation study showed 25.5% jitter reduction vs no smoothing |

---

## Exception Hierarchy

```
RehabSystemError (Exception)
├── ConfigError
│   └── ExerciseNotFoundError
├── LandmarkError
└── AnalysisError
```

`ConfigError` and its subclasses are raised at startup (config load).  
`LandmarkError` and `AnalysisError` are for programmatic use (e.g. batch scripts).  
The real-time analyzer deliberately does **not** raise — it returns `TRACKING` status.
