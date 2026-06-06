import streamlit as st
import pandas as pd
import numpy as np

#M1 MAC SAFE MATPLOTLIB FIX
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import seaborn as sns
import cv2
import mediapipe as mp
import tempfile
import os

# 1. SETUP & CONFIGURATION
st.set_page_config(page_title="Tele-Rehab Kinematic Dashboard", page_icon="⚕️", layout="wide")
sns.set_theme(style="whitegrid", context="paper")

mp_pose = mp.solutions.pose

# 2. BIOMECHANICAL MATH ENGINE
def calculate_2d_angle(a, b, c):
    """Trigonometry for joint angles."""
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360.0 - angle
    return angle

@st.cache_data 
def extract_kinematic_angles(df, ex_num):
    """Translates (X,Y) coordinates into readable medical angles for CSVs."""
    r_shoulder, r_elbow, r_wrist, r_hip = 12, 14, 16, 24
    angles_df = df.copy()
    shoulder_angles, elbow_angles = [], []
    
    for index, row in angles_df.iterrows():
        try:
            hip = [row[f'Lm{r_hip}_x'], row[f'Lm{r_hip}_y']]
            shoulder = [row[f'Lm{r_shoulder}_x'], row[f'Lm{r_shoulder}_y']]
            elbow = [row[f'Lm{r_elbow}_x'], row[f'Lm{r_elbow}_y']]
            wrist = [row[f'Lm{r_wrist}_x'], row[f'Lm{r_wrist}_y']]
            
            shoulder_angles.append(calculate_2d_angle(hip, shoulder, elbow))
            elbow_angles.append(calculate_2d_angle(shoulder, elbow, wrist))
        except KeyError:
            shoulder_angles.append(np.nan)
            elbow_angles.append(np.nan)
            
    angles_df['Shoulder_Angle'] = shoulder_angles
    angles_df['Elbow_Angle'] = elbow_angles
    
    # Smooth the lines for the UI
    angles_df['Shoulder_Angle'] = angles_df['Shoulder_Angle'].ewm(span=3).mean()
    angles_df['Elbow_Angle'] = angles_df['Elbow_Angle'].ewm(span=3).mean()
    return angles_df

# 3. LIVE & RECORDED VIDEO ENGINE
def draw_clinical_overlay(frame, landmarks, width, height, exercise_num):
    """Handles the math and drawing for a single video frame."""
    shoulder = [landmarks[12].x * width, landmarks[12].y * height]
    elbow = [landmarks[14].x * width, landmarks[14].y * height]
    wrist = [landmarks[16].x * width, landmarks[16].y * height]
    hip = [landmarks[24].x * width, landmarks[24].y * height] 
    
    ui_color = (0, 0, 255) # Red Default
    status_text = "Tracking..."
    
    if landmarks[12].visibility > 0.65 and landmarks[14].visibility > 0.65:
        if exercise_num == 1:
            elbow_angle = calculate_2d_angle(shoulder, elbow, wrist)
            if elbow_angle >= 160.0:
                ui_color = (0, 255, 0); status_text = f"Form: PASS ({elbow_angle:.0f} deg)"
            else:
                ui_color = (0, 0, 255); status_text = f"FAIL: Keep Arm Straight! ({elbow_angle:.0f} deg)"
                
        elif exercise_num == 2:
            shoulder_angle = calculate_2d_angle(hip, shoulder, elbow)
            if shoulder_angle >= 120.0:
                ui_color = (0, 255, 0); status_text = f"Target 'V' Reached! ({shoulder_angle:.0f} deg)"
            elif shoulder_angle <= 90.0:
                ui_color = (0, 165, 255); status_text = f"Target 'W' Reached! ({shoulder_angle:.0f} deg)"
            else:
                ui_color = (0, 255, 255); status_text = f"Transitioning... ({shoulder_angle:.0f} deg)"
                
        elif exercise_num == 3:
            elbow_angle = calculate_2d_angle(shoulder, elbow, wrist)
            if elbow_angle <= 100.0: 
                ui_color = (0, 255, 0); status_text = f"Depth: PASS ({elbow_angle:.0f} deg)"
            else:
                ui_color = (0, 0, 255); status_text = f"FAIL: Go Deeper! ({elbow_angle:.0f} deg)"

        # Draw Skeleton
        s_tup, e_tup = (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1]))
        w_tup, h_tup = (int(wrist[0]), int(wrist[1])), (int(hip[0]), int(hip[1]))
        
        cv2.line(frame, s_tup, e_tup, ui_color, 4)
        cv2.line(frame, e_tup, w_tup, ui_color, 4)
        if exercise_num in [2, 3]:
            cv2.line(frame, h_tup, s_tup, ui_color, 4)
            cv2.circle(frame, h_tup, 5, ui_color, -1)

        for pt in [s_tup, e_tup, w_tup]: cv2.circle(frame, pt, 5, ui_color, -1)
        cv2.putText(frame, status_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_color, 3)
        
    return frame

def process_recorded_video(video_path, exercise_num):
    """Processes uploaded MP4 files."""
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = tfile_out.name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    
    my_bar = st.progress(0, text="Processing Video Frames...")
    total_frames, frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0

    with mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.65) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Lock memory for Apple Silicon
            image_rgb.flags.writeable = False 
            results = pose.process(image_rgb)
            
            #THE READ-ONLY FIX: Unlock memory so OpenCV can draw!
            image_rgb.flags.writeable = True

            if results.pose_landmarks:
                frame = draw_clinical_overlay(frame, results.pose_landmarks.landmark, width, height, exercise_num)

            out.write(frame)
            frame_count += 1
            if total_frames > 0: my_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release(); out.release(); my_bar.empty()
    return output_path

# 4. SIDEBAR NAVIGATION
st.sidebar.markdown("## 🎓 Multimedia University")
st.sidebar.markdown("**FYP 2: Tele-Rehabilitation System**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Select Application Mode", [
    "1. Static Data Analysis (CSV)", 
    "2. Upload Video Analysis (MP4)",
    "3. Live Webcam Analysis 🔴",
    "4. Project Analytics & Stats"
])
st.sidebar.markdown("---")
selected_exercise = st.sidebar.selectbox("Select Clinical Exercise", [1, 2, 3], format_func=lambda x: f"Exercise {x}")

# 5. MAIN DASHBOARD ROUTING
st.title("⚕️ Vision-Based Rehabilitation Assessment")

# --- MODE 1: CSV DATA ANALYSIS ---
if app_mode == "1. Static Data Analysis (CSV)":
    st.markdown("Automated clinical grading using **MediaPipe** and **Dynamic Time Warping (DTW)**.")
    uploaded_csv = st.file_uploader("Upload Patient Coordinate Data (CSV)", type=["csv"])
    
    if uploaded_csv is not None:
        raw_df = pd.read_csv(uploaded_csv)
        with st.spinner('Applying Strict Filter and calculating biomechanical angles...'):
            kinematic_df = extract_kinematic_angles(raw_df, ex_num=selected_exercise)
        
        #THE BLANK GRAPH FIX: Drop NaNs so the line plotting doesn't break
        plot_df = kinematic_df[['Shoulder_Angle', 'Elbow_Angle']].dropna()
            
        st.success("✅ Data Processed Successfully!")
        
        titles = {
            1: ('Right Shoulder Abduction', 'Right Elbow Flexion (Constraint)', 90, 160, 'Clinical Fail Zone'),
            2: ('Arm "V" to "W" Transition', 'Elbow Tracking', 120, 90, 'Target Transition Zone'),
            3: ('Inclined Push-up Alignment', 'Elbow Flexion (Depth)', 150, 100, 'Target Depth Required')
        }
        title_1, title_2, target_line_1, fail_line_2, fail_label = titles[selected_exercise]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plotting with the cleaned data (plot_df)
        sns.lineplot(data=plot_df, x=plot_df.index, y='Shoulder_Angle', ax=ax1, color='#2b7bba', linewidth=2.5)
        ax1.set_title(title_1, fontsize=12, fontweight='bold')
        ax1.axhline(y=target_line_1, color='red', linestyle='--', alpha=0.5, label=f'Target: {target_line_1}°')
        ax1.legend()

        sns.lineplot(data=plot_df, x=plot_df.index, y='Elbow_Angle', ax=ax2, color='#5cb85c', linewidth=2.5)
        ax2.set_title(title_2, fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 200)
        ax2.axhline(y=fail_line_2, color='orange', linestyle='--', alpha=0.7, label=f'{fail_label}: {fail_line_2}°')
        ax2.legend()
        st.pyplot(fig)

# --- MODE 2: UPLOADED VIDEO PROCESSING ---
elif app_mode == "2. Upload Video Analysis (MP4)":
    uploaded_video = st.file_uploader("Upload Raw Patient Video (MP4)", type=["mp4", "mov"])
    if uploaded_video is not None:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile_in.write(uploaded_video.read())
        
        if st.button("▶️ Process Clinical Video"):
            processed_video_path = process_recorded_video(tfile_in.name, selected_exercise)
            st.success("✅ Video Processed! Rendering playback...")
            st.video(open(processed_video_path, 'rb').read())
            os.remove(tfile_in.name); os.remove(processed_video_path)

# --- MODE 3: LIVE WEBCAM ---
elif app_mode == "3. Live Webcam Analysis 🔴":
    st.markdown("### Real-Time Edge Computing Demonstration")
    st.warning("Make sure your terminal has permission to access the Mac Camera!")
    
    start_cam = st.checkbox("Turn On Webcam")
    FRAME_WINDOW = st.image([])
    
    if start_cam:
        camera = cv2.VideoCapture(0) # 0 is the default Mac webcam
        with mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.65) as pose:
            while start_cam:
                ret, frame = camera.read()
                if not ret:
                    st.error("Cannot access webcam.")
                    break
                
                frame = cv2.flip(frame, 1) # Mirror image
                
                width, height = frame.shape[1], frame.shape[0]
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Lock memory for Apple Silicon
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                
                # 🌟 THE READ-ONLY FIX: Unlock memory!
                image_rgb.flags.writeable = True
                
                if results.pose_landmarks:
                    frame_rgb = draw_clinical_overlay(image_rgb, results.pose_landmarks.landmark, width, height, selected_exercise)
                    FRAME_WINDOW.image(frame_rgb)
                else:
                    FRAME_WINDOW.image(image_rgb)
        camera.release()
    else:
        st.info("Click the checkbox to activate the camera.")

# --- MODE 4: PROJECT ANALYTICS ---
elif app_mode == "4. Project Analytics & Stats":
    st.markdown("### FYP 2 Data Validations (REHAB24-6 Dataset)")
    
    tab1, tab2 = st.tabs(["Environmental Robustness", "Ablation Study (Error Reduction)"])
    
    with tab1:
        st.markdown("#### System Performance in Real-World Clinical Noise")
        st.markdown("The pipeline's resilience against real-world clinical variables.")
        
        # THE ANALYTICS FIX: Complete tables from your notebook
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💡 Lighting Conditions**")
            st.markdown("*(0 = Standard, 1 = Artificial)*")
            df_light = pd.DataFrame({'lights_on': [0, 1], 'accuracy_pct': [100.00, 83.33]})
            st.dataframe(df_light, hide_index=True, use_container_width=True)
            
        with col2:
            st.markdown("**🎥 Camera Direction (Occlusion)**")
            st.markdown("*(Angle of the patient)*")
            df_cam = pd.DataFrame({'cam17_orientation': ['front', 'half-profile'], 'accuracy_pct': [87.50, 100.00]})
            st.dataframe(df_cam, hide_index=True, use_container_width=True)
            
        with col3:
            st.markdown("**🚶 Background Clutter**")
            st.markdown("*(Extra person in frame)*")
            df_clutter = pd.DataFrame({'extra_person_in_cam17': [0, 1], 'accuracy_pct': [88.88, 100.00]})
            st.dataframe(df_clutter, hide_index=True, use_container_width=True)
            
        st.info("📌 **Clinical Observation:** The system maintains 100% accuracy with background clutter, validating the Foreground Priority logic.")

    with tab2:
        st.markdown("#### Spatial Tracking Error Reduction via Strict Filter")
        ablation_data = pd.DataFrame({
            'Pipeline Phase': ['Raw MediaPipe', '+ Bounding Box', '+ 12-Point Check', '+ Kinematic EMA (Strict)'],
            'Spatial Error': [0.04355, 0.04452, 0.04452, 0.03314]
        })
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=ablation_data, x='Pipeline Phase', y='Spatial Error', palette=['#d9534f', '#d9534f', '#d9534f', '#5cb85c'], ax=ax)
        ax.set_ylabel('Avg Euclidean Distance')
        plt.xticks(rotation=15)
        st.pyplot(fig)
        st.success("**Outcome:** 25.5% reduction in spatial tracking error.")

# 6. GLOBAL BENCHMARKS FOOTER
st.markdown("---")
st.markdown("### System Accuracy vs 3D ML SOTA (Černek et al. 2024)")
col1, col2, col3 = st.columns(3)
col1.metric("Ex 1 (Arm Abduction)", "91.7%", "+25.5% Gain via Filter")
col2.metric("Ex 2 (Arm VW)", "81.8%", "Edge Computing Optimized")
col3.metric("Ex 3 (Push-ups)", "77.8%", "Subject to Foreshortening")
