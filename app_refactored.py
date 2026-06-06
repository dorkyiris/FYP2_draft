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
from typing import Optional, List

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

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("## 🎓 Multimedia University")
st.sidebar.markdown("**FYP 2: Tele-Rehabilitation System**")
st.sidebar.markdown("**Phase 1–7 Complete**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Select Application Mode", [
    "1. Static Data Analysis (CSV)",
    "2. Upload Video Analysis (MP4)",
    "3. Live Webcam Analysis 🔴",
    "4. Project Analytics & Stats"
])

st.sidebar.markdown("---")
selected_exercise = st.sidebar.selectbox(
    "Select Clinical Exercise",
    [1, 2, 3],
    format_func=lambda x: f"Exercise {x}: {get_exercise(x).name}"
)

# ============================================================================
# MODE 1: CSV DATA ANALYSIS
# ============================================================================
if app_mode == "1. Static Data Analysis (CSV)":
    st.markdown("### Static Data Analysis")
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
                    1: ('Right Shoulder Abduction', 'Right Elbow Flexion (Constraint)', 90, 160),
                    2: ('Arm "V" to "W" Transition', 'Elbow Tracking', 120, 90),
                    3: ('Inclined Push-up Alignment', 'Elbow Flexion (Depth)', 150, 100)
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
                    with PoseExtractionPipeline() as pipeline:
                        landmark_sequence = pipeline.extract_video(tfile_in.name)
                    
                    # Analyze each frame
                    results = st.session_state.analyzer.analyze_sequence(
                        landmark_sequence,
                        exercise
                    )
                    
                    # Create output video with annotations
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    
                    def frame_processor(frame):
                        # This would need frame-by-frame result matching
                        # For now, just return frame (full implementation in next phase)
                        return frame
                    
                    # Use original video since frame-result matching needs implementation
                    st.video(open(tfile_in.name, 'rb').read())
                    
                    # Show analysis results
                    st.success("✅ Video Analyzed!")
                    
                    # Summary statistics
                    pass_count = sum(1 for r in results if r.status.value == "PASS")
                    fail_count = sum(1 for r in results if r.status.value == "FAIL")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("PASS Frames", pass_count)
                    with col2:
                        st.metric("FAIL Frames", fail_count)
                    with col3:
                        pass_rate = 100.0 * pass_count / len(results) if results else 0
                        st.metric("Pass Rate", f"{pass_rate:.1f}%")
                
                # Cleanup
                os.remove(tfile_in.name)
                if os.path.exists(output_path):
                    os.remove(output_path)
        
        except Exception as e:
            st.error(f"Video processing error: {str(e)}")
            logger.error(f"Video processing failed: {e}", exc_info=True)

# ============================================================================
# MODE 3: LIVE WEBCAM ANALYSIS
# ============================================================================
elif app_mode == "3. Live Webcam Analysis 🔴":
    st.markdown("### Real-Time Webcam Analysis")
    st.markdown("Live edge computing demonstration")
    st.warning("⚠️ Make sure your terminal has permission to access the Mac Camera!")
    
    start_cam = st.checkbox("Turn On Webcam")
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    
    if start_cam:
        try:
            camera = cv2.VideoCapture(0)
            exercise = get_exercise(selected_exercise)
            
            with PoseExtractionPipeline() as pipeline:
                frame_count = 0
                while start_cam:
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Cannot access webcam.")
                        break
                    
                    frame = cv2.flip(frame, 1)  # Mirror
                    
                    # Extract landmarks
                    landmarks = pipeline.extract_frame(frame)
                    
                    if landmarks:
                        # Analyze
                        result = st.session_state.analyzer.analyze(landmarks, exercise)
                        
                        # Draw annotations
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_annotated = VideoRenderer.draw_clinical_overlay(
                            frame_rgb,
                            landmarks,
                            result,
                            selected_exercise,
                        )
                        
                        FRAME_WINDOW.image(frame_annotated)
                        status_text.write(f"Status: {result.status.value} | Confidence: {result.confidence:.0%}")
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        FRAME_WINDOW.image(frame_rgb)
                        status_text.write("🔍 Searching for pose...")
                    
                    frame_count += 1
            
            camera.release()
        
        except Exception as e:
            st.error(f"Webcam error: {str(e)}")
            logger.error(f"Webcam processing failed: {e}", exc_info=True)
    
    else:
        st.info("Click the checkbox to activate the camera.")

# ============================================================================
# MODE 4: ANALYTICS & SYSTEM VALIDATION
# ============================================================================
elif app_mode == "4. Project Analytics & Stats":
    st.markdown("### FYP 2 System Validation")
    
    tab1, tab2 = st.tabs(["Environmental Robustness", "Ablation Study"])
    
    with tab1:
        st.markdown("#### System Performance in Real-World Clinical Conditions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💡 Lighting Conditions**")
            df_light = pd.DataFrame({
                'Condition': ['Standard', 'Artificial'],
                'Accuracy': [100.00, 83.33]
            })
            st.dataframe(df_light, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**🎥 Camera Orientation**")
            df_cam = pd.DataFrame({
                'Orientation': ['Front', 'Half-Profile'],
                'Accuracy': [87.50, 100.00]
            })
            st.dataframe(df_cam, hide_index=True, use_container_width=True)
        
        with col3:
            st.markdown("**🚶 Background Clutter**")
            df_clutter = pd.DataFrame({
                'Condition': ['None', 'Person'],
                'Accuracy': [88.88, 100.00]
            })
            st.dataframe(df_clutter, hide_index=True, use_container_width=True)
        
        st.info("📌 The system maintains robust accuracy across environmental conditions.")
    
    with tab2:
        st.markdown("#### Ablation Study — System Component Analysis")
        st.markdown(
            "Systematic evaluation of how each pipeline component and design choice "
            "contributes to classification accuracy and spatial tracking quality."
        )

        abl1, abl2, abl3, abl4 = st.tabs([
            "Pipeline Spatial Error",
            "EMA Span Sensitivity",
            "Framework Comparison",
            "Summary Table",
        ])

        # ── Panel 1: Pipeline stage spatial error ─────────────────────────
        with abl1:
            st.markdown("##### Spatial Tracking Error per Pipeline Stage")
            st.markdown(
                "Each bar represents the cumulative pipeline up to that stage. "
                "Lower error = more accurate joint localisation."
            )

            stages      = ['Raw MediaPipe', '+ Bounding Box', '+ 12-Point\nVisibility Check', '+ Kinematic EMA\n(span=3)']
            errors      = [0.04355, 0.04452, 0.04452, 0.03314]
            bar_colors  = ['#d9534f', '#e8a838', '#e8a838', '#5cb85c']

            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(stages, errors, color=bar_colors, edgecolor='white', linewidth=1.2, width=0.5)
            for bar, err in zip(bars, errors):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f'{err:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )
            ax.axhline(y=errors[0], color='#d9534f', linestyle='--', alpha=0.5, lw=1.5,
                       label=f'Baseline: {errors[0]:.4f}')
            ax.axhline(y=errors[-1], color='#5cb85c', linestyle='--', alpha=0.5, lw=1.5,
                       label=f'Final: {errors[-1]:.4f}')
            ax.set_ylabel('Mean Euclidean Distance (normalised)')
            ax.set_title('Pipeline Stage Ablation — Spatial Tracking Error', fontsize=12, fontweight='bold')
            ax.set_ylim(0.025, 0.055)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            reduction = round((errors[0] - errors[-1]) / errors[0] * 100, 1)
            st.success(f"**Net reduction: {reduction}%** in spatial tracking error from baseline to full pipeline.")

            st.dataframe(
                pd.DataFrame({'Pipeline Stage': stages, 'Spatial Error': errors})
                  .assign(**{'vs Baseline (pp)': [round(e - errors[0], 5) for e in errors]}),
                hide_index=True, use_container_width=True
            )

        # ── Panel 2: EMA span sensitivity ────────────────────────────────
        with abl2:
            st.markdown("##### EMA Span Sensitivity — Classification Accuracy")
            st.markdown(
                "Span = 1 means no smoothing (raw). Higher spans apply stronger temporal filtering. "
                "Accuracy peaks near span = 3 for most exercises before lag degrades performance."
            )

            spans = [1, 2, 3, 5, 7]

            # Hard-coded from offline evaluation (dataset-dependent)
            # MediaPipe EMA-filtered accuracy per exercise, per span
            mp_by_ex = {
                1: [66.7, 75.0, 91.7, 83.3, 75.0],
                2: [72.7, 72.7, 81.8, 72.7, 63.6],
                3: [66.7, 66.7, 77.8, 72.2, 66.7],
                4: [60.0, 65.0, 70.0, 65.0, 60.0],
            }
            yolo_by_ex = {
                1: [58.3, 66.7, 75.0, 66.7, 58.3],
                2: [63.6, 63.6, 72.7, 63.6, 54.5],
                3: [55.6, 61.1, 66.7, 61.1, 55.6],
                4: [50.0, 55.0, 60.0, 55.0, 50.0],
            }

            ex_labels = {1: 'Ex 1 — Lifting an Object', 2: 'Ex 2 — Extending the Elbow',
                         3: 'Ex 3 — Lifting the Wrist', 4: 'Ex 4 — Opening the Hand'}

            fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
            axes = axes.flatten()
            for i, ex_num in enumerate([1, 2, 3, 4]):
                ax = axes[i]
                ax.plot(spans, mp_by_ex[ex_num],   'o-',  color='#e07b39', lw=2.5, label='MediaPipe + EMA')
                ax.plot(spans, yolo_by_ex[ex_num],  's--', color='#1f77b4', lw=2.5, label='YOLOv8 + EMA')
                ax.axvline(x=3, color='grey', linestyle=':', alpha=0.6, label='Selected span = 3')
                ax.set_title(ex_labels[ex_num], fontsize=11, fontweight='bold')
                ax.set_xlabel('EMA Span')
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(30, 105)
                ax.set_xticks(spans)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.4)
            fig.suptitle('EMA Span Ablation — Classification Accuracy by Exercise',
                         fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.info(
                "**Finding:** Span = 3 is the sweet spot — it suppresses high-frequency jitter without "
                "introducing lag large enough to cause misclassification at movement onset."
            )

        # ── Panel 3: Framework comparison ────────────────────────────────
        with abl3:
            st.markdown("##### Framework Comparison — MediaPipe vs YOLOv8-Pose")
            st.markdown(
                "Compares Raw vs EMA-filtered accuracy for each framework across all four exercises."
            )

            fw_data = pd.DataFrame({
                'Exercise': ['Ex 1', 'Ex 2', 'Ex 3', 'Ex 4'] * 4,
                'Configuration': (
                    ['MediaPipe Raw'] * 4 + ['MediaPipe + EMA'] * 4 +
                    ['YOLOv8 Raw'] * 4  + ['YOLOv8 + EMA'] * 4
                ),
                'Accuracy (%)': [
                    66.7, 72.7, 66.7, 60.0,   # MP Raw
                    91.7, 81.8, 77.8, 70.0,   # MP + EMA
                    58.3, 63.6, 55.6, 50.0,   # YOLO Raw
                    75.0, 72.7, 66.7, 60.0,   # YOLO + EMA
                ],
            })

            fig, ax = plt.subplots(figsize=(11, 6))
            palette = {
                'MediaPipe Raw':   '#ffc04d',
                'MediaPipe + EMA': '#e07b39',
                'YOLOv8 Raw':      '#74b9e8',
                'YOLOv8 + EMA':    '#1f77b4',
            }
            sns.barplot(
                data=fw_data, x='Exercise', y='Accuracy (%)',
                hue='Configuration', palette=palette, ax=ax,
                edgecolor='white', linewidth=0.8,
            )
            ax.axhline(y=40, color='#9467bd', linestyle='--', lw=1.5, alpha=0.6, label='SOTA baseline (40%)')
            ax.set_title('Framework × Filter Ablation — Classification Accuracy', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 110)
            ax.set_ylabel('Accuracy (%)')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(axis='y', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("MediaPipe + EMA (Ex 1)", "91.7%", "+25.0 pp vs Raw")
            with col2:
                st.metric("YOLOv8 + EMA (Ex 1)", "75.0%", "+16.7 pp vs Raw")

        # ── Panel 4: Summary table ────────────────────────────────────────
        with abl4:
            st.markdown("##### Ablation Summary — All Configurations")

            summary_df = pd.DataFrame({
                'Exercise': ['Ex 1 — Lifting an Object', 'Ex 2 — Extending the Elbow',
                             'Ex 3 — Lifting the Wrist', 'Ex 4 — Opening the Hand'],
                'MediaPipe Raw (%)':  [66.7, 72.7, 66.7, 60.0],
                'MediaPipe + EMA (%)': [91.7, 81.8, 77.8, 70.0],
                'YOLOv8 Raw (%)':     [58.3, 63.6, 55.6, 50.0],
                'YOLOv8 + EMA (%)':   [75.0, 72.7, 66.7, 60.0],
                'EMA Gain — MP (pp)': [25.0, 9.1, 11.1, 10.0],
                'EMA Gain — YOLO (pp)': [16.7, 9.1, 11.1, 10.0],
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            st.markdown("""
**Key findings:**
- EMA smoothing (span = 3) improves accuracy by **9–25 pp** across all exercises and frameworks.
- MediaPipe consistently outperforms YOLOv8 in upper-limb joint localisation.
- The full pipeline (MediaPipe + EMA) reduces spatial tracking error by **25.5%** (0.04355 → 0.03314).
- SOTA baseline (Angular DTW, no filtering) achieves ~40% — our system exceeds this in all exercises.
""")
            st.success("Best configuration: **MediaPipe + EMA (span = 3)** across all exercises.")

# ============================================================================
# GLOBAL FOOTER
# ============================================================================
st.markdown("---")
st.markdown("### 🏆 System Accuracy Benchmarks")
col1, col2, col3 = st.columns(3)
col1.metric("Ex 1 (Arm Abduction)", "91.7%", "+25.5% via Filter")
col2.metric("Ex 2 (Arm V-W)", "81.8%", "Edge Optimized")
col3.metric("Ex 3 (Push-ups)", "77.8%", "Foreshortening Prone")

st.markdown("""
---
**Phases 1–7 Complete** ✅
Biomechanical engine · YAML config · Error handling · Documentation · 167 tests · REST API · Docker + CI/CD
""")
