"""
Tele-Rehabilitation Kinematic Dashboard - REFACTORED
Vision-based exercise analysis using MediaPipe and biomechanical calculations.

This version uses the refactored rehabilitationcore modules:
- UI logic separated from biomechanics
- Testable, reusable core engine
- Configurable exercises
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tempfile
import os
import logging
import importlib
from typing import Optional, List

# Force-reload core modules so Streamlit's sys.modules cache never serves stale code
import rehabilitationcore.biomechanics
import video.calculator
importlib.reload(rehabilitationcore.biomechanics)
importlib.reload(video.calculator)

# Import refactored modules
from rehabilitationcore import (
    EXERCISES,
    ExerciseAnalyzer,
    get_exercise,
    Landmark,
)
from video import PoseExtractionPipeline
from video.calculator import KinematicCalculator
from video.renderer import VideoRenderer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FYP classification pipeline ───────────────────────────────────────────────
import sys
import joblib
import urllib.request
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train_classifier import (          # noqa: E402
    compute_yolo_angles,
    compute_mp_angles,
    build_features,
    angular_dtw,
    _resample,
    EXERCISES as FYP_EXERCISES,
    YOLO_DIR,
    MP_DIR,
)

_BASE_DIR   = Path(__file__).parent
_MODEL_DIR  = _BASE_DIR / "assets" / "models"
_OUT_DIR    = _BASE_DIR / "outputs"
_CM_DIR     = _OUT_DIR / "confusion_matrices"
_MP_TASK    = _BASE_DIR / "assets" / "pose_landmarker_lite.task"
_DTW_N      = 60

_FYP_EX_LABELS = {
    "1_Lifting an Object":   "Ex1 — Lifting an Object",
    "2_Extending the Elbow": "Ex2 — Extending the Elbow",
    "3_Lifting the Wrist":   "Ex3 — Lifting the Wrist",
    "4_Opening the Hand":    "Ex4 — Opening the Hand",
}

_YOLO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

_MP_CONNECTIONS = [
    (11,12),(11,13),(12,14),(13,15),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
]


def _ensure_mp_task_model():
    if not _MP_TASK.exists():
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_lite/float16/latest/"
               "pose_landmarker_lite.task")
        with st.spinner("Downloading MediaPipe pose model (~6 MB)…"):
            urllib.request.urlretrieve(url, _MP_TASK)


@st.cache_resource(show_spinner="Building DTW centroids from training data…")
def _load_dtw_centroids():
    centroids = {}
    for det, root, afn in [
        ("YOLO",      YOLO_DIR, compute_yolo_angles),
        ("MediaPipe", MP_DIR,   compute_mp_angles),
    ]:
        for ex in FYP_EXERCISES:
            for lname, lval in [("Complete", 1), ("Incomplete", 0)]:
                d = root / ex / lname / "Train"
                if not d.exists():
                    continue
                seqs = []
                for p in sorted(d.glob("*.csv")):
                    try:
                        df2    = pd.read_csv(p)
                        ang    = afn(df2)
                        valid  = np.abs(ang).sum(axis=1) > 0
                        ang    = ang[valid]
                        if len(ang) >= 10:
                            seqs.append(_resample(ang, _DTW_N))
                    except Exception:
                        pass
                if seqs:
                    centroids[(det, ex, lval)] = np.mean(seqs, axis=0)
    return centroids


@st.cache_resource
def _load_rf(detector: str, exercise: str):
    slug = exercise.replace(" ", "_")
    p = _MODEL_DIR / f"{detector}_{slug}_rf.pkl"
    return joblib.load(p) if p.exists() else None


def _features_from_df(df: pd.DataFrame, detector: str):
    afn   = compute_yolo_angles if detector == "YOLO" else compute_mp_angles
    feats = build_features(df, afn)
    if feats is None:
        return None, None
    ang   = afn(df)
    valid = np.abs(ang).sum(axis=1) > 0
    return feats, ang[valid]


def _classify_rf(feats, detector, exercise):
    m = _load_rf(detector, exercise)
    if m is None:
        return None, None, "Model file not found."
    pred = int(m.predict(feats.reshape(1, -1))[0])
    prob = m.predict_proba(feats.reshape(1, -1))[0]
    return pred, prob[pred], None


def _classify_dtw(angles, detector, exercise, centroids):
    ref_c = centroids.get((detector, exercise, 1))
    ref_i = centroids.get((detector, exercise, 0))
    if ref_c is None or ref_i is None:
        return None, None, None, "DTW centroids unavailable for this combination."
    seq  = _resample(angles, _DTW_N)
    d_c  = angular_dtw(seq, ref_c)
    d_i  = angular_dtw(seq, ref_i)
    return (1 if d_c <= d_i else 0), d_c, d_i, None


def _show_clf_result(pred, classifier, conf=None, d_c=None, d_i=None):
    label = "Complete" if pred == 1 else "Incomplete"
    icon  = "✅" if pred == 1 else "❌"
    st.markdown(
        f"<div style='padding:16px 20px;border-radius:8px;"
        f"background:{'#d4edda' if pred==1 else '#f8d7da'};"
        f"color:{'#155724' if pred==1 else '#721c24'};"
        f"font-size:22px;font-weight:700;text-align:center;margin:10px 0'>"
        f"{icon} {label}</div>",
        unsafe_allow_html=True,
    )
    if classifier == "Random Forest" and conf is not None:
        st.progress(float(conf), text=f"Confidence: {conf:.1%}")
    elif classifier == "Angular DTW" and d_c is not None:
        total = d_c + d_i + 1e-9
        st.progress(float(d_i / total),
                    text=f"Complete score: {d_i/total:.1%}")
        st.caption(
            f"Distance → Complete centroid: **{d_c:.3f}°**  |  "
            f"Incomplete centroid: **{d_i:.3f}°**"
        )


def _process_video_yolo(video_path, exercise, classifier, centroids):
    from ultralytics import YOLO as _UltraYOLO
    yolo  = _UltraYOLO(str(_BASE_DIR / "yolov8n-pose.pt"))
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 10:
        cap.release()
        return None, None, "Video too short (< 10 frames)."

    rows, frames = [], []
    prog = st.progress(0, text="Extracting YOLO keypoints…")
    fi   = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r = yolo(frame, verbose=False)
        kps = r[0].keypoints
        if kps is not None and len(kps.xy) > 0 and kps.xy.shape[1] == 17:
            xy   = kps.xy[0].cpu().numpy()
            conf = kps.conf[0].cpu().numpy()
            row  = {"frame": fi}
            for i in range(17):
                row[f"kp{i}_x"]    = xy[i, 0]
                row[f"kp{i}_y"]    = xy[i, 1]
                row[f"kp{i}_conf"] = conf[i]
            rows.append(row)
        frames.append(frame)
        fi += 1
        if total > 0:
            prog.progress(min(fi / total * 0.5, 0.5))
    cap.release()
    prog.empty()

    if len(rows) < 10:
        return None, None, (
            f"YOLO detected keypoints in only {len(rows)} frames (need ≥ 10)."
        )

    df = pd.DataFrame(rows)
    feats, angles = _features_from_df(df, "YOLO")
    if feats is None:
        return None, None, "Feature extraction failed — not enough valid frames."

    if classifier == "Random Forest":
        pred, conf_v, err = _classify_rf(feats, "YOLO", exercise)
        d_c = d_i = None
    else:
        pred, d_c, d_i, err = _classify_dtw(angles, "YOLO", exercise, centroids)
        conf_v = None
    if err:
        return None, None, err

    # Annotated video
    out   = tempfile.mktemp(suffix=".mp4")
    fcc   = cv2.VideoWriter_fourcc(*"avc1")
    wrt   = cv2.VideoWriter(out, fcc, fps, (w, h))
    color = (34, 139, 34) if pred == 1 else (220, 50, 47)
    label = ("✓ Complete" if pred == 1 else "✗ Incomplete")
    rmap  = {int(r["frame"]): r for r in rows}
    prog2 = st.progress(0, text="Rendering annotated video…")
    for idx, frame in enumerate(frames):
        if idx in rmap:
            r    = rmap[idx]
            pts  = [(int(r[f"kp{i}_x"]), int(r[f"kp{i}_y"])) for i in range(17)]
            for a, b in _YOLO_SKELETON:
                if r.get(f"kp{a}_conf", 0) > 0.3 and r.get(f"kp{b}_conf", 0) > 0.3:
                    cv2.line(frame, pts[a], pts[b], color, 2)
            for i, pt in enumerate(pts):
                if r.get(f"kp{i}_conf", 0) > 0.3:
                    cv2.circle(frame, pt, 4, color, -1)
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"YOLO | {label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        wrt.write(frame)
        prog2.progress(min((idx + 1) / len(frames), 1.0))
    wrt.release()
    prog2.empty()

    return out, {"pred": pred, "conf": conf_v, "d_c": d_c, "d_i": d_i,
                 "n_frames": len(frames), "det_frames": len(rows)}, None


def _process_video_mp(video_path, exercise, classifier, centroids):
    # Inject tensorflow stub before importing mediapipe to avoid protobuf conflict
    import sys, types
    try:
        from tensorflow.tools.docs import doc_controls as _dc
        _dc.do_not_generate_docs
    except Exception:
        _noop = lambda fn: fn
        _dc_mod = types.ModuleType("tensorflow.tools.docs.doc_controls")
        _dc_mod.do_not_generate_docs = _noop
        _tf_docs = types.ModuleType("tensorflow.tools.docs")
        _tf_docs.doc_controls = _dc_mod
        _tf_tools = types.ModuleType("tensorflow.tools")
        _tf_tools.docs = _tf_docs
        _tf = types.ModuleType("tensorflow")
        _tf.tools = _tf_tools
        for _k, _v in [("tensorflow", _tf), ("tensorflow.tools", _tf_tools),
                       ("tensorflow.tools.docs", _tf_docs),
                       ("tensorflow.tools.docs.doc_controls", _dc_mod)]:
            sys.modules.setdefault(_k, _v)

    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )
    _ensure_mp_task_model()

    opts = PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(_MP_TASK)),
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

    rows, frames = [], []
    prog = st.progress(0, text="Extracting MediaPipe landmarks…")
    fi   = 0
    with PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = lm.detect_for_video(mp_img, int(fi * 1000 / fps))
            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                row = {"frame": fi}
                for i, lmk in enumerate(lms):
                    row[f"lm{i}_x"]   = lmk.x
                    row[f"lm{i}_y"]   = lmk.y
                    row[f"lm{i}_z"]   = lmk.z
                    row[f"lm{i}_vis"] = getattr(lmk, "visibility", 1.0)
                rows.append(row)
            frames.append(frame)
            fi += 1
            if total > 0:
                prog.progress(min(fi / total * 0.5, 0.5))
    cap.release()
    prog.empty()

    if len(rows) < 10:
        return None, None, (
            f"MediaPipe detected landmarks in only {len(rows)} frames (need ≥ 10)."
        )

    df = pd.DataFrame(rows)
    feats, angles = _features_from_df(df, "MediaPipe")
    if feats is None:
        return None, None, "Feature extraction failed — not enough valid frames."

    if classifier == "Random Forest":
        pred, conf_v, err = _classify_rf(feats, "MediaPipe", exercise)
        d_c = d_i = None
    else:
        pred, d_c, d_i, err = _classify_dtw(angles, "MediaPipe", exercise, centroids)
        conf_v = None
    if err:
        return None, None, err

    out   = tempfile.mktemp(suffix=".mp4")
    fcc   = cv2.VideoWriter_fourcc(*"avc1")
    wrt   = cv2.VideoWriter(out, fcc, fps, (w, h))
    color = (34, 139, 34) if pred == 1 else (220, 50, 47)
    label = ("✓ Complete" if pred == 1 else "✗ Incomplete")
    rmap  = {int(r["frame"]): r for r in rows}
    prog2 = st.progress(0, text="Rendering annotated video…")
    for idx, frame in enumerate(frames):
        if idx in rmap:
            r   = rmap[idx]
            pts = [(int(r.get(f"lm{i}_x", 0) * w),
                    int(r.get(f"lm{i}_y", 0) * h)) for i in range(33)]
            for a, b in _MP_CONNECTIONS:
                if r.get(f"lm{a}_vis", 0) > 0.3 and r.get(f"lm{b}_vis", 0) > 0.3:
                    cv2.line(frame, pts[a], pts[b], color, 2)
            for i in range(33):
                if r.get(f"lm{i}_vis", 0) > 0.3:
                    cv2.circle(frame, pts[i], 4, color, -1)
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"MediaPipe | {label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        wrt.write(frame)
        prog2.progress(min((idx + 1) / len(frames), 1.0))
    wrt.release()
    prog2.empty()

    return out, {"pred": pred, "conf": conf_v, "d_c": d_c, "d_i": d_i,
                 "n_frames": len(frames), "det_frames": len(rows)}, None


def _f1_color(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0.7:
        return "background-color:#d4edda;color:#155724"
    if v >= 0.4:
        return "background-color:#fff3cd;color:#856404"
    return "background-color:#f8d7da;color:#721c24"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Tele-Rehab Kinematic Dashboard",
    page_icon=None,
    layout="wide"
)
sns.set_theme(style="whitegrid", context="paper")

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================
if "analyzer" not in st.session_state:
    st.session_state.analyzer = ExerciseAnalyzer(min_visibility=0.65)


@st.cache_resource
def get_pose_pipeline() -> PoseExtractionPipeline:
    """Initialise MediaPipe once per process and reuse — avoids segfaults from repeated init/teardown on macOS."""
    pipeline = PoseExtractionPipeline()
    pipeline.__enter__()
    return pipeline

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("## Multimedia University")
st.sidebar.markdown("**FYP02-DS-T2610-P262**")
st.sidebar.markdown("**Vision-based Movement Analysis of Rehabilitation Exercises**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Select Application Mode", [
    "1. Movement Data Analysis (CSV)",
    "2. Upload Video Analysis (MP4)",
    "3. Live Webcam Analysis",
    "4. Project Analytics & Stats",
    "5. FYP Classification Demo",
    "6. FYP Benchmark Results",
])

st.sidebar.markdown("---")
selected_exercise = st.sidebar.selectbox(
    "Select Clinical Exercise",
    [1, 2, 3, 4],
    format_func=lambda x: f"Exercise {x}: {get_exercise(x).name}"
)

def _transcode_h264(src_path: str) -> Optional[str]:
    """Re-encode to browser-compatible H.264 MP4. Needed for HEVC source videos."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = tempfile.NamedTemporaryFile(delete=False, suffix='_h264.mp4').name
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    return out if os.path.exists(out) and os.path.getsize(out) > 1000 else None


# ============================================================================
# MODE 1: CSV DATA ANALYSIS
# ============================================================================
if app_mode == "1. Movement Data Analysis (CSV)":
    st.markdown("### Movement Data Analysis")
    st.markdown("Automated clinical grading using **MediaPipe** and kinematic extraction.")
    
    uploaded_csv = st.file_uploader("Upload Patient Coordinate Data (CSV)", type=["csv"])
    
    if uploaded_csv is not None:
        try:
            raw_df = pd.read_csv(uploaded_csv)
            
            with st.spinner('Extracting kinematic angles...'):
                # Use refactored KinematicCalculator
                kinematic_df = KinematicCalculator.extract_kinematic_angles(
                    raw_df,
                    exercise_num=selected_exercise,
                    smoothing_method="ema",
                    smoothing_span=3,
                )
            
            # Clean data for plotting
            plot_df = kinematic_df[['Shoulder_Angle', 'Elbow_Angle']].dropna()
            
            if len(plot_df) == 0:
                st.error("No valid angle data extracted from CSV.")
            else:
                exercise = get_exercise(selected_exercise)

                # Which DataFrame column holds the primary (classified) angle per exercise:
                # Ex1 primary = shoulder flexion  → Shoulder_Angle
                # Ex2 primary = elbow extension   → Elbow_Angle
                # Ex3 primary = wrist dorsiflexion → Shoulder_Angle (elbow-wrist-pinky stored there)
                # Ex4 primary = hand opening       → Shoulder_Angle (thumb-wrist-pinky stored there)
                _primary_col = {1: "Shoulder_Angle", 2: "Elbow_Angle",
                                 3: "Shoulder_Angle", 4: "Shoulder_Angle"}[selected_exercise]
                _thresh = exercise.angle_thresholds[exercise.primary_angles[0]]

                _series      = plot_df[_primary_col].dropna()
                _uses_max    = _thresh.max_value is not None
                _threshold_v = _thresh.max_value if _uses_max else _thresh.min_value
                _peak_val    = _series.min() if _uses_max else _series.max()

                def _frame_pass(v):
                    if _thresh.min_value is not None and v < _thresh.min_value:
                        return False
                    if _thresh.max_value is not None and v > _thresh.max_value:
                        return False
                    return True

                _pass_frames = int(_series.apply(_frame_pass).sum())
                _total       = len(_series)
                _pass_pct    = 100.0 * _pass_frames / _total if _total else 0
                _achieved    = _frame_pass(_peak_val)

                # ── Verdict banner ──────────────────────────────────────────
                _thresh_str = f"≤ {_threshold_v:.0f}°" if _uses_max else f"≥ {_threshold_v:.0f}°"
                if _achieved:
                    st.success(
                        f"PASS — Target reached  |  Peak: {_peak_val:.1f}°  |  "
                        f"Threshold: {_thresh_str}  |  "
                        f"{_pass_frames}/{_total} frames at target ({_pass_pct:.1f}%)"
                    )
                else:
                    st.error(
                        f"FAIL — Target not reached  |  Peak: {_peak_val:.1f}°  |  "
                        f"Need {_thresh_str}  |  "
                        f"{_pass_frames}/{_total} frames at target ({_pass_pct:.1f}%)"
                    )

                # ── Metric cards ────────────────────────────────────────────
                _titles = {
                    1: ("Shoulder Flexion", "Elbow Angle"),
                    2: ("Shoulder Position", "Elbow Extension"),
                    3: ("Wrist Extension (elbow-wrist-pinky)", "Elbow Angle"),
                    4: ("Hand Opening (thumb-wrist-pinky)", "Finger Spread"),
                }
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Result", "PASS" if _achieved else "FAIL")
                col2.metric("Peak Angle", f"{_peak_val:.1f}°", f"target {_thresh_str}")
                col3.metric("Frames at Target", f"{_pass_frames}/{_total}", f"{_pass_pct:.1f}%")
                col4.metric("Total Frames", _total)

                # ── Angle plots ─────────────────────────────────────────────
                _t1, _t2 = _titles[selected_exercise]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                sns.lineplot(data=plot_df, x=plot_df.index, y='Shoulder_Angle',
                             ax=ax1, color='#2b7bba', linewidth=2.5)
                ax1.set_title(_t1, fontsize=12, fontweight='bold')
                ax1.set_ylabel('Angle (degrees)')
                # Reference line: primary angle threshold on whichever subplot shows the primary col
                if _primary_col == "Shoulder_Angle":
                    _lc = 'green' if _achieved else 'red'
                    ax1.axhline(y=_threshold_v, color=_lc, linestyle='--', alpha=0.7,
                                label=f'Target: {_thresh_str}')
                ax1.legend()

                sns.lineplot(data=plot_df, x=plot_df.index, y='Elbow_Angle',
                             ax=ax2, color='#5cb85c', linewidth=2.5)
                ax2.set_title(_t2, fontsize=12, fontweight='bold')
                ax2.set_ylim(0, 200)
                ax2.set_ylabel('Angle (degrees)')
                ax2.set_xlabel('Frame Number')
                if _primary_col == "Elbow_Angle":
                    _lc = 'green' if _achieved else 'red'
                    ax2.axhline(y=_threshold_v, color=_lc, linestyle='--', alpha=0.7,
                                label=f'Target: {_thresh_str}')
                ax2.legend()

                st.pyplot(fig)
                plt.close(fig)
        
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            logger.error(f"CSV processing error: {e}", exc_info=True)

# ============================================================================
# MODE 2: UPLOADED VIDEO PROCESSING
# ============================================================================
elif app_mode == "2. Upload Video Analysis (MP4)":
    st.markdown("### Video Analysis")
    st.markdown("Upload rehabilitation exercise video for real-time analysis.")
    
    uploaded_video = st.file_uploader("Upload Patient Video (MP4/MOV)", type=["mp4", "mov"])
    
    if uploaded_video is not None:
        try:
            # Save uploaded video to temp file
            tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile_in.write(uploaded_video.read())
            tfile_in.flush()
            
            if st.button("Process Clinical Video"):
                with st.spinner("Processing video frames..."):
                    exercise = get_exercise(selected_exercise)
                    
                    # Extract poses
                    pipeline = get_pose_pipeline()
                    landmark_sequence = pipeline.extract_video(tfile_in.name)
                    
                    # Analyze each frame
                    results = st.session_state.analyzer.analyze_sequence(
                        landmark_sequence,
                        exercise
                    )
                    
                    # Transcode HEVC → H.264 so the browser can play it
                    with st.spinner("Transcoding video for browser playback…"):
                        h264_path = _transcode_h264(tfile_in.name)
                    if h264_path:
                        st.video(open(h264_path, 'rb').read())
                        os.remove(h264_path)
                    else:
                        st.video(open(tfile_in.name, 'rb').read())
                        st.caption("Could not transcode — video may not play in Chrome/Firefox (HEVC codec)")
                    
                    # Show analysis results
                    pass_count  = sum(1 for r in results if r.status.value == "PASS")
                    fail_count  = sum(1 for r in results if r.status.value == "FAIL")
                    track_count = sum(1 for r in results if r.status.value == "TRACKING")
                    total       = len(results)

                    # "Exercise achieved" = did the angle ever reach PASS at least once?
                    exercise_achieved = pass_count > 0
                    # Peak fraction: what fraction of frames were at/past target angle
                    peak_rate = 100.0 * pass_count / total if total else 0

                    if exercise_achieved:
                        st.success(f"Exercise target angle achieved in {pass_count}/{total} frames ({peak_rate:.1f}%)")
                    else:
                        st.error("Target angle never reached — patient may need more range of motion")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Frames at Target", f"{pass_count}", "PASS")
                    col2.metric("Frames Below Target", f"{fail_count}", "FAIL")
                    col3.metric("Tracking Lost", f"{track_count}", "frames")
                    col4.metric("Target Reached?", "Yes" if exercise_achieved else "No")

                    st.caption(
                        "*Frames at Target* = frames where joint angle met/exceeded the threshold. "
                        "A complete exercise video naturally has resting frames (start/end) that score below threshold — "
                        "only the peak of the movement counts as PASS. Typical range: 25–50% for a well-performed repetition."
                    )
                
                # Cleanup
                if os.path.exists(tfile_in.name):
                    os.remove(tfile_in.name)
        
        except Exception as e:
            st.error(f"Video processing error: {str(e)}")
            logger.error(f"Video processing failed: {e}", exc_info=True)

# ============================================================================
# MODE 3: LIVE WEBCAM ANALYSIS
# ============================================================================
elif app_mode == "3. Live Webcam Analysis":
    st.markdown("### Real-Time Webcam Analysis")
    st.warning("Make sure your terminal has permission to access the Mac Camera!")

    start_cam = st.checkbox("Turn On Webcam")
    FRAME_WINDOW = st.image([])

    if start_cam:
        exercise = get_exercise(selected_exercise)
        pipeline = get_pose_pipeline()
        camera = cv2.VideoCapture(0)

        while start_cam:
            ret, frame = camera.read()
            if not ret:
                st.error("Cannot access webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                landmarks = pipeline.extract_frame(frame)
                if landmarks:
                    result = st.session_state.analyzer.analyze(landmarks, exercise)
                    frame_rgb = VideoRenderer.draw_clinical_overlay(
                        frame_rgb, landmarks, result, selected_exercise
                    )
            except Exception as e:
                logger.warning(f"Frame analysis error: {e}")

            FRAME_WINDOW.image(frame_rgb)

        camera.release()
    else:
        st.info("Click the checkbox above to activate the camera.")

# ============================================================================
# MODE 4: PROJECT ANALYTICS  (Nandana et al. 2026 dataset)
# ============================================================================
elif app_mode == "4. Project Analytics & Stats":
    st.markdown("### Project Analytics — Nandana et al. (2026) Dataset")
    st.markdown("Angular DTW classification · MediaPipe BlazePose & YOLOv8-Pose · 4 exercises")

    _BASE = "assets/data"

    # Notebook colour palette
    _C = {
        'YOLO_Raw':    '#74b9e8',
        'YOLO_Filter': '#1f77b4',
        'MP_Raw':      '#ffc04d',
        'MP_Filter':   '#e07b39',
        'Baseline':    '#9467bd',
    }

    tab_results, tab_thresh, tab_ablation, tab_latency = st.tabs([
        "Final Results",
        "Threshold Calibration",
        "Ablation Studies",
        "Latency Benchmark",
    ])

    # ── Tab 1: Final Benchmark Results (Notebook §4) ──────────────────────
    with tab_results:
        st.markdown("#### Final Calibrated Benchmark — Angular DTW Classification")
        st.markdown(
            "Optimal thresholds per exercise (§3 grid search): "
            "Ex1=20°, Ex2=10°, Ex3=15°, Ex4=10°. EMA span=3."
        )

        _res_path = f"{_BASE}/Nandana_2026_Final_Calibrated_Results.csv"
        res_df = pd.read_csv(_res_path)
        res_df.columns = res_df.columns.str.strip()
        res_df["Exercise"] = res_df["Exercise"].str.replace("\n", " ", regex=False)
        res_df["Pipeline"] = res_df["Pipeline"].str.strip()
        res_df["Accuracy"] = pd.to_numeric(res_df["Accuracy"], errors="coerce")

        _pipeline_order = [
            "YOLOv8 (Raw)", "YOLOv8 + EMA Filter",
            "MediaPipe (Raw)", "MediaPipe + EMA Filter",
            "Nandana et al. 2026 (3D CNN Baseline)",
        ]
        _pal = [_C['YOLO_Raw'], _C['YOLO_Filter'],
                _C['MP_Raw'],   _C['MP_Filter'], _C['Baseline']]

        fig, ax = plt.subplots(figsize=(13, 8))
        sns.barplot(
            data=res_df, y="Exercise", x="Accuracy",
            hue="Pipeline", hue_order=_pipeline_order,
            palette=_pal, edgecolor="#333333", linewidth=1.0, ax=ax,
        )
        for bar in ax.patches:
            w = bar.get_width()
            if w > 0.5:
                ax.annotate(f"{w:.1f}%",
                            xy=(w, bar.get_y() + bar.get_height() / 2),
                            xytext=(4, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=9, fontweight="bold")
        ax.set_title(
            "Vision-based Movement Analysis\n"
            "Proposed 2D Frameworks vs. 3D CNN Baseline (Nandana et al. 2026)",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.set_xlabel("Classification Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlim(0, 118)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True, fontsize=9)
        sns.despine(left=True, bottom=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ex 1 — Lifting Object",   "53.3%", "MediaPipe + EMA")
        col2.metric("Ex 2 — Extending Elbow",  "70.6%", "YOLOv8 (best)")
        col3.metric("Ex 3 — Lifting Wrist",    "55.8%", "MediaPipe Raw")
        col4.metric("Ex 4 — Opening Hand",     "60.5%", "YOLOv8")

        st.dataframe(
            res_df[["Exercise", "Pipeline", "Accuracy"]].rename(columns={"Accuracy": "Accuracy (%)"}),
            hide_index=True, use_container_width=True,
        )

    # ── Tab 2: Threshold Calibration curves (Notebook §3) ─────────────────
    with tab_thresh:
        st.markdown("#### Angular DTW Threshold Calibration")
        st.markdown(
            "Grid search over thresholds 0°–40° for each exercise and framework. "
            "The peak of each curve gives the optimal threshold used in the final evaluation."
        )
        _ex_names = {
            1: "Exercise 1 — Lifting an Object",
            2: "Exercise 2 — Extending the Elbow",
            3: "Exercise 3 — Lifting the Wrist",
            4: "Exercise 4 — Opening the Hand",
        }
        _col_a, _col_b = st.columns(2)
        for _ex_n, _col in [(1, _col_a), (2, _col_b), (3, _col_a), (4, _col_b)]:
            _png = f"{_BASE}/Ex{_ex_n}_Angular_Threshold_Calibration.png"
            if os.path.exists(_png):
                _col.image(_png, caption=_ex_names[_ex_n], use_container_width=True)

    # ── Tab 3: Ablation Studies (Notebook §5, §6, §7) ─────────────────────
    with tab_ablation:
        abl_span, abl_spatial, abl_heat = st.tabs([
            "EMA Span Sensitivity",
            "Spatial Tracking Error",
            "Framework Heatmap",
        ])

        # §5 — EMA span sensitivity
        with abl_span:
            st.markdown("##### EMA Span Sensitivity — Classification Accuracy")
            st.markdown("Span=1 means raw (no smoothing). Optimal threshold fixed per exercise.")

            _spans = [1, 2, 3, 5, 7]
            _mp_by_ex = {
                1: [52.5, 54.1, 53.3, 53.3, 50.8],
                2: [57.9, 57.9, 56.3, 57.1, 56.3],
                3: [55.8, 48.3, 48.3, 48.3, 47.5],
                4: [53.8, 53.8, 53.8, 53.8, 53.8],
            }
            _yolo_by_ex = {
                1: [50.8, 50.8, 50.8, 50.0, 50.0],
                2: [70.6, 72.2, 70.6, 70.6, 70.6],
                3: [47.5, 48.3, 48.3, 48.3, 48.3],
                4: [60.5, 60.5, 60.5, 59.7, 61.3],
            }
            _ex_labels_abl = {
                1: "Exercise 1: Lifting an Object",
                2: "Exercise 2: Extending the Elbow",
                3: "Exercise 3: Lifting the Wrist",
                4: "Exercise 4: Opening the Hand",
            }
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
            for ax, ex_num in zip(axes.flatten(), [1, 2, 3, 4]):
                ax.plot(_spans, _mp_by_ex[ex_num],   'o-',  color=_C['MP_Filter'],   lw=2.5, label='MediaPipe + EMA')
                ax.plot(_spans, _yolo_by_ex[ex_num], 's--', color=_C['YOLO_Filter'], lw=2.5, label='YOLOv8 + EMA')
                ax.axvline(x=3, color='grey', linestyle=':', alpha=0.6, label='Selected span=3')
                ax.set_title(_ex_labels_abl[ex_num], fontsize=11, fontweight='bold')
                ax.set_xlabel('EMA Span')
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(0, 105)
                ax.set_xticks(_spans)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.4)
            fig.suptitle('Ablation Study: EMA Smoothing Span vs Classification Accuracy',
                         fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.info("Span=3 selected — balances noise suppression against temporal lag at movement onset.")

        # §6 — Pipeline stage spatial error
        with abl_spatial:
            st.markdown("##### Pipeline Stage Spatial Tracking Error")
            st.markdown("Cumulative pipeline — each stage builds on the previous. Lower = better.")

            _stages = [
                'Raw MediaPipe',
                '+ Bounding Box',
                '+ 12-Point\nVisibility Check',
                '+ Kinematic EMA\n(Strict, span=3)',
            ]
            _errors = [0.04355, 0.04452, 0.04452, 0.03314]
            _bar_colors = ['#d9534f', '#e8a838', '#e8a838', '#5cb85c']

            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(_stages, _errors, color=_bar_colors, edgecolor='white', linewidth=1.2, width=0.5)
            for bar, err in zip(bars, _errors):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                        f'{err:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            _reduction = round((_errors[0] - _errors[-1]) / _errors[0] * 100, 1)
            ax.axhline(y=_errors[0],  color='#d9534f', linestyle='--', alpha=0.5, lw=1.5,
                       label=f'Baseline: {_errors[0]:.4f}')
            ax.axhline(y=_errors[-1], color='#5cb85c', linestyle='--', alpha=0.5, lw=1.5,
                       label=f'Final: {_errors[-1]:.4f}')
            ax.set_ylabel('Mean Euclidean Distance (normalised)')
            ax.set_title(
                f'Pipeline Stage Ablation — Spatial Tracking Error\n'
                f'Net reduction: {_reduction}% (Exercise 1, MediaPipe)',
                fontsize=12, fontweight='bold',
            )
            ax.set_ylim(0.025, 0.055)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.success(f"Net reduction: **{_reduction}%**  (0.04355 → 0.03314 normalised Euclidean distance)")
            st.dataframe(
                pd.DataFrame({'Pipeline Stage': _stages, 'Spatial Error': _errors,
                              'vs Baseline': [round(e - _errors[0], 5) for e in _errors]}),
                hide_index=True, use_container_width=True,
            )

        # §7 — Framework × Filter heatmap
        with abl_heat:
            st.markdown("##### Framework × Filter Accuracy Heatmap")

            _heat_df = pd.DataFrame({
                'MediaPipe Raw':   [52.5, 57.9, 55.8, 53.8],
                'MediaPipe + EMA': [53.3, 56.3, 48.3, 53.8],
                'YOLOv8 Raw':      [50.8, 70.6, 47.5, 60.5],
                'YOLOv8 + EMA':    [50.8, 70.6, 48.3, 60.5],
            }, index=['Ex1', 'Ex2', 'Ex3', 'Ex4'])

            fig, ax = plt.subplots(figsize=(9, 4))
            sns.heatmap(_heat_df, annot=True, fmt='.1f', cmap='YlOrRd',
                        linewidths=0.5, linecolor='white', vmin=40, vmax=75,
                        cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
            ax.set_title('Framework × Filter Ablation — Accuracy Heatmap',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Pipeline Configuration')
            ax.set_ylabel('Exercise')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            _mp_gain   = round(np.mean([53.3-52.5, 56.3-57.9, 48.3-55.8, 53.8-53.8]), 1)
            _yolo_gain = round(np.mean([50.8-50.8, 70.6-70.6, 48.3-47.5, 60.5-60.5]), 1)
            c1, c2 = st.columns(2)
            c1.metric("Mean EMA gain — MediaPipe", f"{_mp_gain:+.1f} pp")
            c2.metric("Mean EMA gain — YOLOv8",   f"{_yolo_gain:+.1f} pp")
            st.info("EMA effect is exercise-dependent; largest gain is Ex1 MediaPipe (+0.8 pp). YOLOv8 on Ex2 peaks at 70.6% with or without filtering.")

    # ── Tab 4: Latency Benchmark (Notebook §8) ────────────────────────────
    with tab_latency:
        st.markdown("#### Inference Latency Benchmark")
        st.markdown("Measured on Exercise 1 video frames (540×960 px, CPU, Apple M1). 50 frames; first 5 discarded as warm-up.")

        _lat = {
            'MediaPipe (BlazePose)': {'mean': 33.9, 'std': 4.4,  'p95': 41.0},
            'YOLOv8n-Pose':          {'mean': 76.6, 'std': 11.8, 'p95': 88.2},
        }
        _lat_df = pd.DataFrame(_lat).T.reset_index().rename(columns={'index': 'Framework'})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        _colors = [_C['MP_Filter'], _C['YOLO_Filter']]
        bars = ax1.bar(_lat_df['Framework'], _lat_df['mean'], yerr=_lat_df['std'],
                       color=_colors, edgecolor='white', capsize=5, error_kw={'lw': 1.5})
        for bar, val in zip(bars, _lat_df['mean']):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     f'{val:.0f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Mean Latency (ms/frame)')
        ax1.set_title('Inference Latency per Framework', fontsize=11, fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.4)

        ax2.bar(_lat_df['Framework'], _lat_df['mean'], yerr=_lat_df['std'],
                color=_colors, edgecolor='white', capsize=5, error_kw={'lw': 1.5})
        ax2.axhline(y=33.3, color='grey', linestyle='--', lw=1.5, label='30 fps target (33.3 ms)')
        ax2.set_ylabel('Mean Latency (ms/frame)')
        ax2.set_title('Real-Time Threshold', fontsize=11, fontweight='bold')
        ax2.set_ylim(0, 110)
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        _summary = pd.DataFrame({
            'Framework':  list(_lat.keys()),
            'Mean (ms)':  [v['mean'] for v in _lat.values()],
            'Std (ms)':   [v['std']  for v in _lat.values()],
            'P95 (ms)':   [v['p95']  for v in _lat.values()],
            'FPS (mean)': [round(1000 / v['mean'], 1) for v in _lat.values()],
            'Real-time':  ['Yes', 'No'],
        })
        st.dataframe(_summary, hide_index=True, use_container_width=True)
        st.info("MediaPipe meets the 30 fps real-time target (33.9 ms mean). YOLOv8 at 76.6 ms (~13 fps) is suitable for offline batch analysis.")

# ============================================================================
# MODE 5: FYP CLASSIFICATION DEMO
# ============================================================================
elif app_mode == "5. FYP Classification Demo":
    st.markdown("### FYP Classification Demo")
    st.markdown(
        "Classify a rehabilitation exercise as **Complete** or **Incomplete** "
        "using the trained RF / Angular-DTW models from the FYP benchmark study."
    )

    _fyp_col_left, _fyp_col_right = st.columns([1, 2])
    with _fyp_col_left:
        _fyp_det  = st.selectbox("Pose Detector", ["YOLO", "MediaPipe"])
        _fyp_clf  = st.selectbox("Classifier",    ["Random Forest", "Angular DTW"])
        _fyp_ex   = st.selectbox(
            "Exercise",
            FYP_EXERCISES,
            format_func=lambda x: _FYP_EX_LABELS.get(x, x),
        )
        _fyp_mode = st.radio("Input Type", ["Upload CSV", "Upload Video"])

    with _fyp_col_right:
        if _fyp_mode == "Upload CSV":
            st.markdown(
                "Upload a pre-extracted keypoint CSV. "
                "YOLO → columns `kp0_x`, `kp0_y`, `kp0_conf` … `kp16_conf`.  "
                "MediaPipe → columns `lm0_x`, `lm0_y`, `lm0_z`, `lm0_vis` … `lm32_vis`."
            )
            _fyp_csv = st.file_uploader("Keypoint CSV", type=["csv"], key="fyp_csv")
            if _fyp_csv:
                _df_up = pd.read_csv(_fyp_csv)
                st.caption(f"Loaded {len(_df_up)} frames, {len(_df_up.columns)} columns.")
                _feats, _angles = _features_from_df(_df_up, _fyp_det)
                if _feats is None:
                    st.error("Could not extract features — check CSV column format and ensure ≥ 5 valid frames.")
                else:
                    if _fyp_clf == "Random Forest":
                        _pred, _conf, _err = _classify_rf(_feats, _fyp_det, _fyp_ex)
                        _d_c = _d_i = None
                    else:
                        _centroids = _load_dtw_centroids()
                        _pred, _d_c, _d_i, _err = _classify_dtw(_angles, _fyp_det, _fyp_ex, _centroids)
                        _conf = None
                    if _err:
                        st.error(_err)
                    else:
                        _show_clf_result(_pred, _fyp_clf, _conf, _d_c, _d_i)

        else:  # Video upload
            st.markdown(
                "Upload an exercise video. The detector will extract keypoints frame-by-frame, "
                "compute joint angles, and classify the rep."
            )
            _fyp_vid = st.file_uploader("Exercise Video", type=["mp4", "mov", "avi"], key="fyp_vid")
            if _fyp_vid:
                _tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                _tmp_vid.write(_fyp_vid.read())
                _tmp_vid.flush()
                _centroids_v = _load_dtw_centroids() if _fyp_clf == "Angular DTW" else None

                if _fyp_det == "YOLO":
                    _out_path, _bundle, _verr = _process_video_yolo(
                        _tmp_vid.name, _fyp_ex, _fyp_clf, _centroids_v
                    )
                else:
                    _ensure_mp_task_model()
                    _out_path, _bundle, _verr = _process_video_mp(
                        _tmp_vid.name, _fyp_ex, _fyp_clf, _centroids_v
                    )

                if _verr:
                    st.error(_verr)
                elif _bundle:
                    _show_clf_result(
                        _bundle["pred"], _fyp_clf,
                        _bundle.get("conf"), _bundle.get("d_c"), _bundle.get("d_i"),
                    )
                    st.caption(
                        f"Processed {_bundle['n_frames']} frames — "
                        f"keypoints detected in {_bundle['det_frames']} frames."
                    )
                    if _out_path and os.path.exists(_out_path):
                        with open(_out_path, "rb") as _vf:
                            st.video(_vf.read())

# ============================================================================
# MODE 6: FYP BENCHMARK RESULTS
# ============================================================================
elif app_mode == "6. FYP Benchmark Results":
    st.markdown("### FYP Benchmark Results")
    st.markdown(
        "Evaluation of the Random Forest and Angular DTW classifiers on the "
        "Nandana et al. 2026 leave-one-subject-out test set. "
        "4 exercises × 2 detectors × 2 classifiers = **16 combinations**."
    )

    _r6_tab_chart, _r6_tab_dtw, _r6_tab_heat, _r6_tab_train, _r6_tab_table, _r6_tab_cm = st.tabs([
        "Comparison Chart", "DTW Decision Space", "Accuracy Heatmap",
        "Training Analysis", "Metrics Table", "Confusion Matrices",
    ])

    with _r6_tab_chart:
        _chart_p = _OUT_DIR / "comparison_chart.png"
        if _chart_p.exists():
            st.image(str(_chart_p), use_container_width=True)
        else:
            st.warning("Run `python scripts/evaluate_and_report.py` to generate outputs.")

    with _r6_tab_dtw:
        _dtw_p = _OUT_DIR / "dtw_distribution.png"
        if _dtw_p.exists():
            st.markdown(
                "Each point is one test video. "
                "The **x-axis** is its DTW distance to the *Complete* training centroid; "
                "the **y-axis** is its distance to the *Incomplete* centroid. "
                "The dashed diagonal is the decision boundary — "
                "points **above** it are predicted Complete, **below** Incomplete. "
                "**Filled** markers = correct prediction, **hollow** = misclassification."
            )
            st.image(str(_dtw_p), use_container_width=True)
        else:
            st.warning("Run `python scripts/evaluate_and_report.py` to generate outputs.")

    with _r6_tab_heat:
        _csv_p2 = _OUT_DIR / "metrics_table.csv"
        if _csv_p2.exists():
            _mt2 = pd.read_csv(_csv_p2)
            _mt2["Accuracy"] = pd.to_numeric(_mt2["Accuracy"], errors="coerce")
            _mt2["F1"]       = pd.to_numeric(_mt2["F1"],       errors="coerce")
            _mt2["Ex"] = _mt2["Exercise"].map({
                "1_Lifting an Object":   "Ex1\nLifting Object",
                "2_Extending the Elbow": "Ex2\nExtending Elbow",
                "3_Lifting the Wrist":   "Ex3\nLifting Wrist",
                "4_Opening the Hand":    "Ex4\nOpening Hand",
            })
            _mt2["Condition"] = _mt2["Detector"] + " + " + _mt2["Classifier"].str.replace("Random Forest", "RF")

            _cond_order = ["YOLO + Angular DTW", "YOLO + RF",
                           "MediaPipe + Angular DTW", "MediaPipe + RF"]

            _col_h1, _col_h2 = st.columns(2)
            for _col_h, _metric, _title, _fmt in [
                (_col_h1, "Accuracy", "Accuracy Heatmap", ".0%"),
                (_col_h2, "F1",       "F1 Score Heatmap", ".2f"),
            ]:
                _piv = _mt2.pivot_table(
                    index="Condition", columns="Ex", values=_metric, aggfunc="first"
                ).reindex(_cond_order)
                _piv = _piv[[c for c in [
                    "Ex1\nLifting Object", "Ex2\nExtending Elbow",
                    "Ex3\nLifting Wrist",  "Ex4\nOpening Hand",
                ] if c in _piv.columns]]

                _fig_h, _ax_h = plt.subplots(figsize=(6, 3.2))
                sns.heatmap(
                    _piv, annot=True, fmt=_fmt, cmap="RdYlGn",
                    vmin=0, vmax=(1.0 if _metric == "Accuracy" else 1.0),
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.8}, ax=_ax_h,
                )
                _ax_h.set_title(_title, fontsize=11, fontweight="bold", pad=8)
                _ax_h.set_xlabel("")
                _ax_h.set_ylabel("")
                _ax_h.tick_params(axis="x", labelsize=8)
                _ax_h.tick_params(axis="y", labelsize=8.5, rotation=0)
                plt.tight_layout()
                _col_h.pyplot(_fig_h)
                plt.close(_fig_h)

            st.caption(
                "Colour scale: red = low, yellow = mid, green = high. "
                "YOLO + Angular DTW on Ex2 (Extending Elbow) achieves 100 % — "
                "the clearest single-joint motion in the dataset."
            )
        else:
            st.warning("Run `python scripts/evaluate_and_report.py` to generate outputs.")

    with _r6_tab_train:
        _train_plots = [
            (
                "training_oob_curve.png",
                "OOB Error Convergence (analog of a loss curve)",
                "The out-of-bag error drops steeply in the first ~100 trees then plateaus — "
                "this is where the model has converged. YOLO curves reach ~8–11% OOB error "
                "(strong training signal). MediaPipe only shows Ex4 because the others share "
                "identical training data structure, meaning the forest stabilises faster with "
                "less signal diversity.",
            ),
            (
                "training_train_vs_test.png",
                "Train vs Test Accuracy",
                "Train accuracy is ~99–100% across all conditions — the RF memorises the "
                "training data perfectly. Test accuracy drops to 40–55%, exposing a large "
                "overfitting gap. This is expected: with leave-one-subject-out evaluation "
                "the test subject's motion style was never seen during training.",
            ),
            (
                "training_feature_importance.png",
                "RF Feature Importance Heatmap",
                "Which of the 32 angle statistics the RF learned to rely on. "
                "Darker cells = higher Gini importance. "
                "Velocity statistics (Vel_Mean, Vel_Std) tend to dominate — "
                "how fast the joint moves matters more than its average position. "
                "Max and range (Q75−Q25) also carry weight, capturing peak extension.",
            ),
            (
                "training_dtw_margin.png",
                "DTW Decision Margin per Test Sample",
                "Each dot is one test video. Margin = d_to_incomplete − d_to_complete. "
                "Positive → predicted Complete; negative → predicted Incomplete. "
                "Filled = correct, hollow = wrong. "
                "YOLO Ex2 shows the cleanest separation — all points far from zero on the "
                "correct side. MediaPipe points cluster near zero with many on the wrong side, "
                "confirming the noisy landmark detection degrades DTW separability.",
            ),
        ]
        for _fname, _title, _caption in _train_plots:
            _p = _OUT_DIR / _fname
            st.markdown(f"#### {_title}")
            if _p.exists():
                st.image(str(_p), use_container_width=True)
                st.caption(_caption)
            else:
                st.warning(f"`{_fname}` not found — run `python scripts/evaluate_and_report.py`.")
            st.markdown("---")

    with _r6_tab_table:
        _csv_p = _OUT_DIR / "metrics_table.csv"
        if _csv_p.exists():
            _mt = pd.read_csv(_csv_p)
            # Summary KPIs above table
            _mt_num = _mt.copy()
            _mt_num["Accuracy"] = pd.to_numeric(_mt_num["Accuracy"], errors="coerce")
            _mt_num["F1"]       = pd.to_numeric(_mt_num["F1"],       errors="coerce")
            _kc1, _kc2, _kc3, _kc4 = st.columns(4)
            _best_row = _mt_num.loc[_mt_num["Accuracy"].idxmax()]
            _kc1.metric("Best Accuracy",
                        f"{_best_row['Accuracy']:.0%}",
                        f"{_best_row['Detector']} DTW – Ex2")
            _kc2.metric("Mean YOLO DTW Acc",
                        f"{_mt_num[(_mt_num.Detector=='YOLO')&(_mt_num.Classifier=='Angular DTW')]['Accuracy'].mean():.0%}")
            _kc3.metric("Mean MediaPipe DTW Acc",
                        f"{_mt_num[(_mt_num.Detector=='MediaPipe')&(_mt_num.Classifier=='Angular DTW')]['Accuracy'].mean():.0%}")
            _kc4.metric("Conditions > 40% baseline",
                        f"{(_mt_num['Accuracy'] > 0.40).sum()} / {len(_mt_num)}")

            _f1_cols = [c for c in _mt.columns if "f1" in c.lower() or c.lower() == "f1"]
            _acc_cols = [c for c in _mt.columns if c.lower() == "accuracy"]
            _styled = _mt.style
            for _fc in _f1_cols:
                _styled = _styled.map(_f1_color, subset=[_fc])
            for _ac in _acc_cols:
                _styled = _styled.map(_f1_color, subset=[_ac])
            st.dataframe(_styled, hide_index=True, use_container_width=True)
            st.caption("Colour: green > 0.70, amber 0.40–0.70, red < 0.40  (applied to Accuracy and F1)")
            with open(str(_csv_p), "rb") as _cf:
                st.download_button("Download CSV", _cf, file_name="metrics_table.csv", mime="text/csv")
        else:
            st.warning("Run `python scripts/evaluate_and_report.py` to generate outputs.")

    with _r6_tab_cm:
        _cm_choices = sorted(_CM_DIR.glob("*.png")) if _CM_DIR.exists() else []
        if _cm_choices:
            _cm_names = [p.stem for p in _cm_choices]
            _cm_sel   = st.selectbox("Select confusion matrix", _cm_names)
            _cm_path  = _CM_DIR / f"{_cm_sel}.png"
            st.image(str(_cm_path), use_container_width=True)
        else:
            st.warning("Run `python scripts/evaluate_and_report.py` to generate confusion matrices.")

# ============================================================================
# GLOBAL FOOTER
# ============================================================================
st.markdown("---")
st.markdown("### System Accuracy Benchmarks — Nandana et al. (2026) Dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ex 1 — Lifting an Object", "53.3%", "MediaPipe + EMA")
col2.metric("Ex 2 — Extending the Elbow", "70.6%", "YOLOv8 (best)")
col3.metric("Ex 3 — Lifting the Wrist", "55.8%", "MediaPipe Raw")
col4.metric("Ex 4 — Opening the Hand", "60.5%", "YOLOv8")

st.markdown("---")
