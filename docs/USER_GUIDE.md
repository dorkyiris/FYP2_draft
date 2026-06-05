# User Guide

---

## Running the App

```bash
streamlit run app_refactored.py
```

The dashboard opens in your browser at `http://localhost:8501`.

---

## Application Modes

Use the sidebar to switch between modes and select a clinical exercise.

### Mode 1 — Static Data Analysis (CSV)

Upload a landmarks CSV file (pre-extracted from video). The app computes shoulder and elbow angles, applies EMA smoothing, and plots the kinematic profile for the selected exercise.

**Expected CSV format:**  
Each row is one video frame. Columns follow the pattern `Lm{index}_x`, `Lm{index}_y`, `Lm{index}_z`, `Lm{index}_v` for each MediaPipe landmark index (0–32).

**Output:** Time-series plot of joint angles, summary statistics, pass/fail frame count.

---

### Mode 2 — Upload Video Analysis (MP4)

Upload a `.mp4` exercise recording. The system runs MediaPipe pose extraction frame by frame, then analyses each frame with `ExerciseAnalyzer`.

**Output:** Annotated video download with skeleton overlay and live feedback text, plus a summary of results.

---

### Mode 3 — Live Webcam Analysis

Streams from your webcam in real time. Each frame is processed and the exercise feedback is overlaid instantly.

**Tips:**
- Ensure the full upper body is visible in the frame
- Good lighting improves landmark detection confidence
- If feedback shows "🔍 Tracking…", move further from the camera so the full arm is visible

---

### Mode 4 — Project Analytics & Stats

Displays pre-computed research results: model accuracy comparison across exercises, hyperparameter heatmaps, ablation study results. This mode reads from saved image assets and does not require a video or CSV input.

---

## Selecting an Exercise

The sidebar **Select Clinical Exercise** dropdown shows all exercises from `config/exercises.yaml`:

| ID | Exercise | Primary Angle | Pass Condition |
|---|---|---|---|
| 1 | Lifting an object | Shoulder (hip→shoulder→elbow) | ≥ 90° |
| 2 | Extending the elbow | Elbow (shoulder→elbow→wrist) | ≥ 160° |
| 3 | Lifting the wrist | Wrist | ≥ 15° (tracking only) |
| 4 | Opening the hand | Hand open | ≥ 45° (tracking only) |

Exercises 3 and 4 display TRACKING status because the angle types they require (`wrist`, `hand_open`) are not yet computed by the current analyzer engine.

---

## Changing a Threshold

Edit `config/exercises.yaml` — no code change required. For example, to lower the arm-lift threshold:

```yaml
  1:
    thresholds:
      shoulder:
        value: 80.0    # was 90.0
```

Restart the app after saving.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| "🔍 Tracking…" every frame | Landmark visibility low | Improve lighting; ensure full arm visible |
| `ConfigError: exercises.yaml not found` | Wrong working directory | Run `streamlit run app_refactored.py` from the `FYP2/` root |
| Video upload produces no output | Codec incompatibility | Re-encode with `ffmpeg -i input.mp4 -vcodec libx264 output.mp4` |
| Very jittery angle readings | EMA span too low | Increase `smoothing.span` in `config/system.yaml` |
