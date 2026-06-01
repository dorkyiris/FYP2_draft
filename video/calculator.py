"""
Kinematic calculations on landmark data.
Extends biomechanics with DataFrame-based processing.
"""

import pandas as pd
import numpy as np
import logging
from typing import List

from rehabilitationcore.biomechanics import smooth_signal

logger = logging.getLogger(__name__)


class KinematicCalculator:
    """
    Convert raw landmark data to clinical kinematic metrics.
    Handles DataFrame operations and smoothing.
    """
    
    @staticmethod
    def landmarks_to_dataframe(
        landmarks_sequence: List[List],
        smoothing_method: str = "ema",
        smoothing_span: int = 3,
    ) -> pd.DataFrame:
        """
        Convert landmark sequence to DataFrame with angles.
        
        Args:
            landmarks_sequence: List of landmark lists from extraction
            smoothing_method: 'ema' or 'sma'
            smoothing_span: Window size for smoothing
        
        Returns:
            DataFrame with frame-by-frame coordinate and angle data
        """
        rows = []
        
        for frame_idx, landmarks in enumerate(landmarks_sequence):
            if not landmarks:
                continue
            
            row = {"frame": frame_idx}
            
            # Store all landmark coordinates
            for lm_idx, landmark in enumerate(landmarks):
                if landmark:
                    row[f"Lm{lm_idx}_x"] = landmark.x
                    row[f"Lm{lm_idx}_y"] = landmark.y
                    row[f"Lm{lm_idx}_z"] = landmark.z
                    row[f"Lm{lm_idx}_visibility"] = landmark.visibility
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Created DataFrame with {len(df)} frames")
        
        return df
    
    @staticmethod
    def extract_kinematic_angles(
        df: pd.DataFrame,
        exercise_num: int = 1,
        smoothing_method: str = "ema",
        smoothing_span: int = 3,
    ) -> pd.DataFrame:
        """
        Extract biomechanical angles from landmark data.
        
        Args:
            df: DataFrame with landmark coordinates
            exercise_num: Exercise ID (1, 2, or 3)
            smoothing_method: Smoothing algorithm
            smoothing_span: Smoothing window size
        
        Returns:
            DataFrame with additional Shoulder_Angle and Elbow_Angle columns
        """
        from rehabilitationcore.biomechanics import calculate_2d_angle
        
        # Landmark indices (right side)
        r_shoulder, r_elbow, r_wrist, r_hip = 12, 14, 16, 24
        
        angles_df = df.copy()
        shoulder_angles, elbow_angles = [], []
        
        for index, row in angles_df.iterrows():
            try:
                hip = [row[f'Lm{r_hip}_x'], row[f'Lm{r_hip}_y']]
                shoulder = [row[f'Lm{r_shoulder}_x'], row[f'Lm{r_shoulder}_y']]
                elbow = [row[f'Lm{r_elbow}_x'], row[f'Lm{r_elbow}_y']]
                wrist = [row[f'Lm{r_wrist}_x'], row[f'Lm{r_wrist}_y']]
                
                shoulder_angle = calculate_2d_angle(hip, shoulder, elbow)
                elbow_angle = calculate_2d_angle(shoulder, elbow, wrist)
                
                shoulder_angles.append(shoulder_angle)
                elbow_angles.append(elbow_angle)
            
            except (KeyError, ValueError) as e:
                logger.debug(f"Cannot calculate angles for frame {index}: {e}")
                shoulder_angles.append(np.nan)
                elbow_angles.append(np.nan)
        
        angles_df['Shoulder_Angle'] = shoulder_angles
        angles_df['Elbow_Angle'] = elbow_angles
        
        # Apply smoothing
        angles_df['Shoulder_Angle'] = smooth_signal(
            angles_df['Shoulder_Angle'].tolist(),
            method=smoothing_method,
            span=smoothing_span,
        )
        angles_df['Elbow_Angle'] = smooth_signal(
            angles_df['Elbow_Angle'].tolist(),
            method=smoothing_method,
            span=smoothing_span,
        )
        
        logger.info(
            f"Extracted kinematic angles for exercise {exercise_num}: "
            f"{len(angles_df)} frames with angles"
        )
        
        return angles_df
    
    @staticmethod
    def calculate_error_metrics(angles_df: pd.DataFrame) -> dict:
        """
        Calculate error metrics from angle data.
        
        Args:
            angles_df: DataFrame with angle columns
        
        Returns:
            Dictionary with error metrics
        """
        metrics = {}
        
        for col in ['Shoulder_Angle', 'Elbow_Angle']:
            if col not in angles_df.columns:
                continue
            
            valid_angles = angles_df[col].dropna()
            if len(valid_angles) == 0:
                continue
            
            # Euclidean distance or jitter metric
            if len(valid_angles) > 1:
                diffs = valid_angles.diff().dropna()
                metrics[f"{col}_jitter"] = diffs.abs().mean()
                metrics[f"{col}_stability"] = 1.0 - (metrics[f"{col}_jitter"] / 180.0)
        
        return metrics
