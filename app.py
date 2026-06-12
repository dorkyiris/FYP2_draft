"""
FYP2 — Vision-Based Rehabilitation Assessment
Benchmarking MediaPipe Pose vs YOLOv8 (Nandana et al. 2026 dataset)

Run with: streamlit run app.py
"""

import sys
import os
import time
import tempfile
import warnings
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Import feature-extraction logic from training script ───────────────────────
_SCRIPTS = Path(__file__).parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from train_classifier import (          # noqa: E402
    compute_yolo_angles,
    compute_mp_angles,
    build_features,
    angular_dtw,
    _resample,
    EXERCISES,
    YOLO_DIR,
    MP_DIR,
)

import numpy as np
import pandas as pd
import joblib
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR  = ASSETS_DIR / "models"
OUT_DIR    = BASE_DIR / "outputs"
CM_DIR     = OUT_DIR / "confusion_matrices"
MP_MODEL   = ASSETS_DIR / "pose_landmarker_lite.task"

EX_LABELS = {
    "1_Lifting an Object":   "Ex1 — Lifting an Object",
    "2_Extending the Elbow": "Ex2 — Extending the Elbow",
    "3_Lifting the Wrist":   "Ex3 — Lifting the Wrist",
    "4_Opening the Hand":    "Ex4 — Opening the Hand",
}

YOLO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

DTW_N = 60   # frames used for centroid resampling

# ── MediaPipe model download ───────────────────────────────────────────────────

def _ensure_mp_model():
    if not MP_MODEL.exists():
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_lite/float16/latest/"
               "pose_landmarker_lite.task")
        with st.spinner("Downloading MediaPipe pose model (~6 MB)…"):
            urllib.request.urlretrieve(url, MP_MODEL)

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="FYP2 — Rehab Exercise Assessment",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal global style ───────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stTabs"] button { font-size: 15px; font-weight: 600; }
.result-box { padding: 18px 24px; border-radius: 8px; font-size: 22px;
              font-weight: 700; text-align: center; margin: 12px 0; }
.complete   { background:#d4edda; color:#155724; border:1px solid #c3e6cb; }
.incomplete { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; }
.metric-row { display:flex; gap:16px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Cached: DTW centroids (mean angle sequences per class) ─────────────────────

@st.cache_resource(show_spinner="Building DTW centroids from training data…")
def load_dtw_centroids():
    """
    Load all training CSVs, compute mean angle trajectory per
    (detector, exercise, class) triplet.  Cached for the session lifetime.
    """
    centroids = {}
    for det, root, angle_fn in [
        ("YOLO",      YOLO_DIR, compute_yolo_angles),
        ("MediaPipe", MP_DIR,   compute_mp_angles),
    ]:
        for exercise in EXERCISES:
            for lname, lval in [("Complete", 1), ("Incomplete", 0)]:
                split_dir = root / exercise / lname / "Train"
                if not split_dir.exists():
                    continue
                seqs = []
                for p in sorted(split_dir.glob("*.csv")):
                    try:
                        df     = pd.read_csv(p)
                        angles = angle_fn(df)
                        valid  = np.abs(angles).sum(axis=1) > 0
                        angles = angles[valid]
                        if len(angles) >= 10:
                            seqs.append(_resample(angles, DTW_N))
                    except Exception:
                        pass
                if seqs:
                    centroids[(det, exercise, lval)] = np.mean(seqs, axis=0)
    return centroids

# ── Cached: RF model loader ────────────────────────────────────────────────────

@st.cache_resource
def load_rf_model(detector: str, exercise: str):
    slug = exercise.replace(" ", "_")
    path = MODEL_DIR / f"{detector}_{slug}_rf.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

# ── Feature extraction from a DataFrame ───────────────────────────────────────

def features_from_df(df: pd.DataFrame, detector: str):
    angle_fn = compute_yolo_angles if detector == "YOLO" else compute_mp_angles
    feats = build_features(df, angle_fn)
    if feats is None:
        return None, None
    angles = angle_fn(df)
    valid  = np.abs(angles).sum(axis=1) > 0
    angles = angles[valid]
    return feats, angles

# ── Classification ─────────────────────────────────────────────────────────────

def classify_rf(feats: np.ndarray, detector: str, exercise: str):
    model = load_rf_model(detector, exercise)
    if model is None:
        return None, None, "Model file not found."
    X    = feats.reshape(1, -1)
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0]   # [prob_class0, prob_class1]
    conf = prob[pred]
    return pred, conf, None


def classify_dtw(angles: np.ndarray, detector: str, exercise: str, centroids: dict):
    ref_c = centroids.get((detector, exercise, 1))
    ref_i = centroids.get((detector, exercise, 0))
    if ref_c is None or ref_i is None:
        return None, None, None, "DTW centroids unavailable for this combination."
    seq  = _resample(angles, DTW_N)
    d_c  = angular_dtw(seq, ref_c)
    d_i  = angular_dtw(seq, ref_i)
    pred = 1 if d_c <= d_i else 0
    return pred, d_c, d_i, None

# ── Result display ─────────────────────────────────────────────────────────────

def show_result(pred: int, classifier: str,
                conf=None, d_c=None, d_i=None):
    label = "Complete" if pred == 1 else "Incomplete"
    icon  = "✅" if pred == 1 else "❌"
    cls   = "complete" if pred == 1 else "incomplete"
    st.markdown(
        f'<div class="result-box {cls}">{icon} {label}</div>',
        unsafe_allow_html=True,
    )
    if classifier == "Random Forest" and conf is not None:
        st.progress(float(conf), text=f"Model confidence: {conf:.1%}")
    elif classifier == "Angular DTW" and d_c is not None and d_i is not None:
        total = d_c + d_i + 1e-9
        conf_c = d_i / total   # lower distance to Complete → higher Complete score
        st.caption(
            f"Distance to Complete centroid: **{d_c:.3f}°**  |  "
            f"Distance to Incomplete centroid: **{d_i:.3f}°**"
        )
        st.progress(float(conf_c), text=f"Complete score: {conf_c:.1%}")

# ── YOLO video processing ──────────────────────────────────────────────────────

def process_video_yolo(video_path: str, exercise: str, classifier: str,
                       centroids: dict):
    from ultralytics import YOLO as _YOLO

    yolo = _YOLO(str(BASE_DIR / "yolov8n-pose.pt"))

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 10:
        cap.release()
        return None, None, "Video too short (< 10 frames)."

    # ── First pass: extract keypoints into a DataFrame ─────────────────────────
    rows, frames_buf = [], []
    progress = st.progress(0, text="Extracting YOLO keypoints…")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo(frame, verbose=False)
        kps = results[0].keypoints
        if kps is not None and len(kps.xy) > 0 and kps.xy.shape[1] == 17:
            xy   = kps.xy[0].cpu().numpy()    # (17, 2)
            conf = kps.conf[0].cpu().numpy()  # (17,)
            row  = {"frame": frame_idx}
            for i in range(17):
                row[f"kp{i}_x"]    = xy[i, 0]
                row[f"kp{i}_y"]    = xy[i, 1]
                row[f"kp{i}_conf"] = conf[i]
            rows.append(row)
        frames_buf.append(frame)
        frame_idx += 1
        if total > 0:
            progress.progress(min(frame_idx / total * 0.5, 0.5))
    cap.release()
    progress.empty()

    if len(rows) < 10:
        return None, None, (
            f"YOLO detected keypoints in only {len(rows)} frames "
            f"(need ≥ 10). Check video quality."
        )

    df = pd.DataFrame(rows)

    # ── Classify ───────────────────────────────────────────────────────────────
    feats, angles = features_from_df(df, "YOLO")
    if feats is None:
        return None, None, "Feature extraction failed — not enough valid frames."

    if classifier == "Random Forest":
        pred, conf_val, err = classify_rf(feats, "YOLO", exercise)
        d_c = d_i = None
    else:
        pred, d_c, d_i, err = classify_dtw(angles, "YOLO", exercise, centroids)
        conf_val = None
    if err:
        return None, None, err

    # ── Second pass: draw skeleton overlay ────────────────────────────────────
    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"avc1")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    color = (34, 139, 34) if pred == 1 else (220, 50, 47)   # green / red (BGR)
    label = ("✓ Complete" if pred == 1 else "✗ Incomplete")

    row_map = {int(r["frame"]): r for r in rows}
    progress2 = st.progress(0, text="Rendering annotated video…")

    for fi, frame in enumerate(frames_buf):
        if fi in row_map:
            r = row_map[fi]
            kp_pts = [(int(r[f"kp{i}_x"]), int(r[f"kp{i}_y"])) for i in range(17)]
            for a, b in YOLO_SKELETON:
                if (r.get(f"kp{a}_conf", 0) > 0.3 and
                        r.get(f"kp{b}_conf", 0) > 0.3):
                    cv2.line(frame, kp_pts[a], kp_pts[b], color, 2)
            for i, pt in enumerate(kp_pts):
                if r.get(f"kp{i}_conf", 0) > 0.3:
                    cv2.circle(frame, pt, 4, color, -1)
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"YOLO | {label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        writer.write(frame)
        progress2.progress(min((fi + 1) / len(frames_buf), 1.0))

    writer.release()
    progress2.empty()

    result_bundle = {
        "pred": pred, "conf": conf_val, "d_c": d_c, "d_i": d_i,
        "n_frames": len(frames_buf), "det_frames": len(rows),
    }
    return out_path, result_bundle, None

# ── MediaPipe video processing ─────────────────────────────────────────────────

def process_video_mp(video_path: str, exercise: str, classifier: str,
                     centroids: dict):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )

    _ensure_mp_model()

    base_opts = mpt.BaseOptions(model_asset_path=str(MP_MODEL))
    opts = PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 10:
        cap.release()
        return None, None, "Video too short (< 10 frames)."

    rows, frames_buf = [], []
    progress = st.progress(0, text="Extracting MediaPipe landmarks…")
    frame_idx = 0

    with PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int(frame_idx * 1000 / fps)
            result = lm.detect_for_video(mp_img, ts_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]   # list of NormalizedLandmark
                row = {"frame": frame_idx}
                for i, lmk in enumerate(lms):
                    row[f"lm{i}_x"]   = lmk.x
                    row[f"lm{i}_y"]   = lmk.y
                    row[f"lm{i}_z"]   = lmk.z
                    row[f"lm{i}_vis"] = getattr(lmk, "visibility", 1.0)
                rows.append(row)
            frames_buf.append(frame)
            frame_idx += 1
            if total > 0:
                progress.progress(min(frame_idx / total * 0.5, 0.5))
    cap.release()
    progress.empty()

    if len(rows) < 10:
        return None, None, (
            f"MediaPipe detected landmarks in only {len(rows)} frames "
            f"(need ≥ 10). Check video quality / lighting."
        )

    df = pd.DataFrame(rows)

    feats, angles = features_from_df(df, "MediaPipe")
    if feats is None:
        return None, None, "Feature extraction failed — not enough valid frames."

    if classifier == "Random Forest":
        pred, conf_val, err = classify_rf(feats, "MediaPipe", exercise)
        d_c = d_i = None
    else:
        pred, d_c, d_i, err = classify_dtw(angles, "MediaPipe", exercise, centroids)
        conf_val = None
    if err:
        return None, None, err

    # Annotated video
    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"avc1")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    color    = (34, 139, 34) if pred == 1 else (220, 50, 47)
    label    = ("✓ Complete" if pred == 1 else "✗ Incomplete")

    # MediaPipe pose connections (hand-coded for Tasks API output)
    MP_CONNECTIONS = [
        (11,12),(11,13),(12,14),(13,15),(14,16),  # arms/shoulders
        (11,23),(12,24),(23,24),                   # torso
        (23,25),(24,26),(25,27),(26,28),            # legs
    ]

    row_map = {int(r["frame"]): r for r in rows}
    progress2 = st.progress(0, text="Rendering annotated video…")

    for fi, frame in enumerate(frames_buf):
        if fi in row_map:
            r  = row_map[fi]
            pts = [(int(r.get(f"lm{i}_x", 0) * w),
                    int(r.get(f"lm{i}_y", 0) * h)) for i in range(33)]
            for a, b in MP_CONNECTIONS:
                vis_a = r.get(f"lm{a}_vis", 0)
                vis_b = r.get(f"lm{b}_vis", 0)
                if vis_a > 0.3 and vis_b > 0.3:
                    cv2.line(frame, pts[a], pts[b], color, 2)
            for i in range(33):
                if r.get(f"lm{i}_vis", 0) > 0.3:
                    cv2.circle(frame, pts[i], 4, color, -1)
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"MediaPipe | {label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        writer.write(frame)
        progress2.progress(min((fi + 1) / len(frames_buf), 1.0))

    writer.release()
    progress2.empty()

    result_bundle = {
        "pred": pred, "conf": conf_val, "d_c": d_c, "d_i": d_i,
        "n_frames": len(frames_buf), "det_frames": len(rows),
    }
    return out_path, result_bundle, None

# ── Metrics table with colour highlights ───────────────────────────────────────

def _f1_color(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0.7:
        return "background-color: #d4edda; color: #155724"
    if v >= 0.4:
        return "background-color: #fff3cd; color: #856404"
    return "background-color: #f8d7da; color: #721c24"

# ══════════════════════════════════════════════════════════════════════════════
# APP — HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.title("FYP2 — Vision-Based Rehabilitation Exercise Assessment")
st.caption("MediaPipe Pose vs YOLOv8 | Nandana et al. 2026 Dataset | "
           "Binary classification: Complete (1) vs Incomplete (0)")

tab1, tab2 = st.tabs(["Analyse Exercise", "Results & Evaluation"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE EXERCISE
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # Controls row
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 2, 1])

    with ctrl_col1:
        detector = st.selectbox("Detector", ["YOLO", "MediaPipe"])
    with ctrl_col2:
        classifier = st.selectbox("Classifier",
                                  ["Random Forest", "Angular DTW"])
    with ctrl_col3:
        exercise = st.selectbox(
            "Exercise",
            EXERCISES,
            format_func=lambda e: EX_LABELS[e],
        )
    with ctrl_col4:
        mode = st.selectbox("Input mode", ["CSV Upload", "Video Upload"])

    st.divider()

    # ── Sub-mode A: CSV Upload ─────────────────────────────────────────────────
    if mode == "CSV Upload":
        st.markdown(
            f"**Upload a pre-extracted keypoint CSV** "
            f"({detector} format — same column layout as training data)"
        )

        if detector == "YOLO":
            st.caption(
                "Expected columns: `frame, kp0_x, kp0_y, kp0_conf, …, kp16_conf` "
                "(52 columns total, 17 COCO keypoints)"
            )
        else:
            st.caption(
                "Expected columns: `frame, lm0_x, lm0_y, lm0_z, lm0_vis, …, lm32_vis` "
                "(133 columns total, 33 MediaPipe landmarks)"
            )

        uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                    label_visibility="collapsed")

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Cannot read CSV: {e}")
                st.stop()

            st.caption(f"Loaded: {len(df)} rows × {len(df.columns)} columns")

            with st.spinner("Extracting features…"):
                feats, angles = features_from_df(df, detector)

            if feats is None:
                st.warning(
                    "Not enough valid frames after removing all-zero rows "
                    "(need ≥ 5). The detector may have failed on most frames."
                )
                st.stop()

            # Classify
            if classifier == "Random Forest":
                with st.spinner("Running Random Forest…"):
                    pred, conf, err = classify_rf(feats, detector, exercise)
                if err:
                    st.error(err); st.stop()
                show_result(pred, classifier, conf=conf)

            else:  # Angular DTW
                centroids = load_dtw_centroids()
                with st.spinner("Computing Angular DTW…"):
                    pred, d_c, d_i, err = classify_dtw(
                        angles, detector, exercise, centroids)
                if err:
                    st.error(err); st.stop()
                show_result(pred, classifier, d_c=d_c, d_i=d_i)

            # Angle trajectory plot
            with st.expander("View extracted angle trajectories"):
                angle_fn = (compute_yolo_angles if detector == "YOLO"
                            else compute_mp_angles)
                ang = angle_fn(df)
                valid = np.abs(ang).sum(axis=1) > 0
                ang   = ang[valid]
                if len(ang) > 0:
                    fig, ax = plt.subplots(figsize=(9, 3))
                    labels = ["L Elbow", "R Elbow", "L Shoulder", "R Shoulder"]
                    for k, lbl in enumerate(labels):
                        ax.plot(ang[:, k], label=lbl, linewidth=1.4)
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("Angle (°)")
                    ax.set_title(f"{detector} — {EX_LABELS[exercise]} — angle trajectories")
                    ax.legend(fontsize=8, loc="upper right")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

    # ── Sub-mode B: Video Upload ───────────────────────────────────────────────
    else:
        st.markdown(
            "**Upload a raw video** (MP4 / AVI). "
            "The selected detector runs frame-by-frame to extract keypoints, "
            "then classifies the full sequence."
        )

        uploaded_video = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )

        if uploaded_video is not None:
            # Save to temp file
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_in.write(uploaded_video.read())
            tmp_in.flush()
            tmp_in.close()

            t_start = time.time()

            centroids = load_dtw_centroids()

            if detector == "YOLO":
                out_path, bundle, err = process_video_yolo(
                    tmp_in.name, exercise, classifier, centroids)
            else:
                out_path, bundle, err = process_video_mp(
                    tmp_in.name, exercise, classifier, centroids)

            os.unlink(tmp_in.name)
            elapsed = time.time() - t_start

            if err:
                st.warning(err)
                st.stop()

            # Result
            show_result(
                bundle["pred"], classifier,
                conf=bundle.get("conf"),
                d_c=bundle.get("d_c"),
                d_i=bundle.get("d_i"),
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Total frames", bundle["n_frames"])
            m2.metric("Frames with detections", bundle["det_frames"])
            m3.metric("Processing time", f"{elapsed:.1f} s")

            # Annotated video
            if out_path and Path(out_path).exists():
                st.markdown("**Annotated video**")
                with open(out_path, "rb") as f:
                    st.video(f.read())
                os.unlink(out_path)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    # ── Comparison bar chart ───────────────────────────────────────────────────
    st.subheader("Accuracy comparison across all conditions")
    chart_path = OUT_DIR / "comparison_chart.png"
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
        st.caption(
            "Grouped bar chart comparing Random Forest and Angular DTW accuracy "
            "for YOLO and MediaPipe across all 4 exercises. "
            "Dashed line = Nandana et al. 2026 baseline (~40 %). "
            "YOLO Angular DTW achieves 100 % on Ex2 (Extending the Elbow) — a "
            "single-joint motion with a highly consistent angle trajectory."
        )
    else:
        st.info("Run `scripts/evaluate_and_report.py` to generate this chart.")

    st.divider()

    # ── DTW distribution ──────────────────────────────────────────────────────
    st.subheader("Angular DTW distance distributions (YOLO — test set)")
    dtw_path = OUT_DIR / "dtw_distribution.png"
    if dtw_path.exists():
        st.image(str(dtw_path), use_container_width=True)
        st.caption(
            "Violin plots showing the distribution of Angular DTW distances "
            "(mean over 4 angles, normalised by path length) from each test "
            "sample to its own-class centroid. Lower = motion trajectory closer "
            "to training reference. Ex2 shows tight, well-separated distributions "
            "explaining the 100 % DTW accuracy."
        )
    else:
        st.info("Run `scripts/evaluate_and_report.py` to generate this plot.")

    st.divider()

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("Full metrics table")
    metrics_path = OUT_DIR / "metrics_table.csv"
    if metrics_path.exists():
        df_met = pd.read_csv(metrics_path)

        # Display options
        det_filter = st.multiselect(
            "Filter by detector",
            options=df_met["Detector"].unique().tolist(),
            default=df_met["Detector"].unique().tolist(),
        )
        df_show = df_met[df_met["Detector"].isin(det_filter)].copy()

        styled = (
            df_show.style
            .map(_f1_color, subset=["F1"])
            .format({
                "Accuracy":  "{:.2%}",
                "Precision": "{:.2%}",
                "Recall":    "{:.2%}",
                "F1":        "{:.2%}",
            }, na_rep="—")
            .set_properties(**{"text-align": "center"})
        )
        st.dataframe(styled, use_container_width=True, height=380)
        st.caption(
            "F1 colour: green > 0.70 | yellow 0.40–0.70 | red < 0.40. "
            "Mean Angular DTW is N/A for Random Forest (distance metric "
            "applies only to the DTW nearest-centroid classifier)."
        )
    else:
        st.info("Run `scripts/evaluate_and_report.py` to generate the metrics table.")

    st.divider()

    # ── Confusion matrix viewer ────────────────────────────────────────────────
    st.subheader("Confusion matrices")

    if CM_DIR.exists():
        cm_files = sorted(CM_DIR.glob("*.png"))
        if cm_files:
            cm_col1, cm_col2, cm_col3 = st.columns(3)

            with cm_col1:
                cm_det = st.selectbox("Detector", ["YOLO", "MediaPipe"],
                                       key="cm_det")
            with cm_col2:
                cm_ex = st.selectbox(
                    "Exercise", EXERCISES,
                    format_func=lambda e: EX_LABELS[e],
                    key="cm_ex",
                )
            with cm_col3:
                cm_clf = st.selectbox("Classifier", ["RF", "DTW"], key="cm_clf")

            slug  = cm_ex.replace(" ", "_")
            fname = f"{cm_det}_{slug}_{cm_clf}.png"
            cm_path = CM_DIR / fname

            if cm_path.exists():
                st.image(str(cm_path), width=420)
                st.caption(
                    f"{cm_det} — {EX_LABELS[cm_ex]} — "
                    f"{'Random Forest' if cm_clf=='RF' else 'Angular DTW'}. "
                    "Rows = Actual, Columns = Predicted. "
                    "Numbers inside cells: count (role)."
                )
            else:
                available = [f.name for f in cm_files]
                st.warning(
                    f"`{fname}` not found. "
                    f"Available: {', '.join(available[:4])}…"
                )
        else:
            st.info("No confusion matrix images found in `outputs/confusion_matrices/`.")
    else:
        st.info("Run `scripts/evaluate_and_report.py` to generate confusion matrices.")
