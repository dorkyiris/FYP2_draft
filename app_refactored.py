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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Tele-Rehab Kinematic Dashboard",
    page_icon="⚕️",
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
st.sidebar.markdown("## 🎓 Multimedia University")
st.sidebar.markdown("**FYP02-DS-T2610-P262**")
st.sidebar.markdown("**Vision-based Movement Analysis of Rehabilitation Exercises**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Select Application Mode", [
    "1. Movement Data Analysis (CSV)",
    "2. Upload Video Analysis (MP4)",
    "3. Live Webcam Analysis 🔴",
    "4. Project Analytics & Stats"
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
                st.success(f"✅ Data Processed Successfully! ({len(plot_df)} frames)")
                
                # Get exercise definition for titles
                exercise = get_exercise(selected_exercise)
                
                # Prepare visualization titles
                titles = {
                    1: ('Shoulder Flexion (Lifting)', 'Elbow Angle (Constraint)', 90, 160),
                    2: ('Shoulder Position', 'Elbow Extension', 90, 160),
                    3: ('Wrist Extension Angle', 'Elbow Angle (Constraint)', 15, 160),
                    4: ('Hand Opening Angle', 'Finger Spread', 45, 60),
                }
                
                title_1, title_2, target_line_1, fail_line_2 = titles[selected_exercise]
                
                # Create plots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                sns.lineplot(
                    data=plot_df,
                    x=plot_df.index,
                    y='Shoulder_Angle',
                    ax=ax1,
                    color='#2b7bba',
                    linewidth=2.5
                )
                ax1.set_title(title_1, fontsize=12, fontweight='bold')
                ax1.axhline(y=target_line_1, color='red', linestyle='--', alpha=0.5, label=f'Target: {target_line_1}°')
                ax1.set_ylabel('Angle (degrees)')
                ax1.legend()
                
                sns.lineplot(
                    data=plot_df,
                    x=plot_df.index,
                    y='Elbow_Angle',
                    ax=ax2,
                    color='#5cb85c',
                    linewidth=2.5
                )
                ax2.set_title(title_2, fontsize=12, fontweight='bold')
                ax2.set_ylim(0, 200)
                ax2.axhline(
                    y=fail_line_2,
                    color='orange',
                    linestyle='--',
                    alpha=0.7,
                    label=f'Target Threshold: {fail_line_2}°'
                )
                ax2.set_ylabel('Angle (degrees)')
                ax2.set_xlabel('Frame Number')
                ax2.legend()
                
                st.pyplot(fig)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Shoulder Angle Mean", f"{plot_df['Shoulder_Angle'].mean():.1f}°")
                with col2:
                    st.metric("Elbow Angle Mean", f"{plot_df['Elbow_Angle'].mean():.1f}°")
                with col3:
                    st.metric("Total Frames Analyzed", len(plot_df))
        
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
            
            if st.button("▶️ Process Clinical Video"):
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
                        st.caption("⚠️ Could not transcode — video may not play in Chrome/Firefox (HEVC codec)")
                    
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
                        st.success(f"✅ Exercise target angle **achieved** in {pass_count}/{total} frames ({peak_rate:.1f}%)")
                    else:
                        st.error("❌ Target angle never reached — patient may need more range of motion")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Frames at Target", f"{pass_count}", "PASS")
                    col2.metric("Frames Below Target", f"{fail_count}", "FAIL")
                    col3.metric("Tracking Lost", f"{track_count}", "frames")
                    col4.metric("Target Reached?", "Yes ✅" if exercise_achieved else "No ❌")

                    st.caption(
                        "ℹ️ *Frames at Target* = frames where joint angle met/exceeded the threshold. "
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
elif app_mode == "3. Live Webcam Analysis 🔴":
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

    _BASE = "An_upper_limb_stroke_rehabilitation_exercise_video"

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
# GLOBAL FOOTER
# ============================================================================
st.markdown("---")
st.markdown("### 🏆 System Accuracy Benchmarks (Nandana et al. 2026 dataset)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ex 1 — Lifting an Object", "53.3%", "MediaPipe + EMA")
col2.metric("Ex 2 — Extending the Elbow", "70.6%", "YOLOv8 (best)")
col3.metric("Ex 3 — Lifting the Wrist", "55.8%", "MediaPipe Raw")
col4.metric("Ex 4 — Opening the Hand", "60.5%", "YOLOv8")

st.markdown("---")
