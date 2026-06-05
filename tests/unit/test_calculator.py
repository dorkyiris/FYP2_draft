"""Unit tests for KinematicCalculator."""

import pytest
import numpy as np
import pandas as pd
from typing import List

from rehabilitationcore.models import Landmark
from video.calculator import KinematicCalculator


def _make_landmark_sequence(n_frames: int = 3) -> List[List[Landmark]]:
    """Create a simple sequence of landmark frames (straight arm)."""
    frames = []
    for _ in range(n_frames):
        lms = [Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)]
        lms = list(lms)
        lms[24] = Landmark(x=0.5, y=0.0, z=0.0, visibility=0.9)  # Hip
        lms[12] = Landmark(x=0.5, y=0.3, z=0.0, visibility=0.9)  # Shoulder
        lms[14] = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9)  # Elbow
        lms[16] = Landmark(x=0.5, y=0.7, z=0.0, visibility=0.9)  # Wrist
        frames.append(lms)
    return frames


class TestLandmarksToDataframe:

    def test_returns_dataframe(self):
        seq = _make_landmark_sequence(3)
        df = KinematicCalculator.landmarks_to_dataframe(seq)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_frames(self):
        seq = _make_landmark_sequence(5)
        df = KinematicCalculator.landmarks_to_dataframe(seq)
        assert len(df) == 5

    def test_frame_column_present(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(2))
        assert "frame" in df.columns

    def test_landmark_columns_present(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(1))
        assert "Lm12_x" in df.columns
        assert "Lm12_y" in df.columns
        assert "Lm24_x" in df.columns

    def test_empty_sequence_returns_empty_dataframe(self):
        df = KinematicCalculator.landmarks_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_frame_skipped(self):
        seq = _make_landmark_sequence(2)
        seq.insert(1, [])   # inject empty frame
        df = KinematicCalculator.landmarks_to_dataframe(seq)
        assert len(df) == 2  # empty frame skipped

    def test_coordinates_stored_correctly(self):
        seq = _make_landmark_sequence(1)
        df = KinematicCalculator.landmarks_to_dataframe(seq)
        assert abs(df.iloc[0]["Lm12_x"] - 0.5) < 1e-6
        assert abs(df.iloc[0]["Lm24_y"] - 0.0) < 1e-6


class TestExtractKinematicAngles:

    def test_adds_shoulder_angle_column(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(3))
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        assert "Shoulder_Angle" in angles_df.columns

    def test_adds_elbow_angle_column(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(3))
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        assert "Elbow_Angle" in angles_df.columns

    def test_angles_in_valid_range(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(3))
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        valid = angles_df["Elbow_Angle"].dropna()
        assert (valid >= 0).all() and (valid <= 180).all()

    def test_straight_arm_elbow_near_180(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(2))
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        # Straight arm: hip→shoulder→elbow→wrist all inline → elbow ~180°
        assert angles_df["Elbow_Angle"].iloc[0] > 170

    def test_sma_smoothing_accepted(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(4))
        angles_df = KinematicCalculator.extract_kinematic_angles(df, smoothing_method="sma")
        assert "Elbow_Angle" in angles_df.columns

    def test_missing_landmark_columns_produce_nan(self):
        # DataFrame with no landmark columns
        df = pd.DataFrame({"frame": [0, 1]})
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        assert angles_df["Shoulder_Angle"].isna().all()
        assert angles_df["Elbow_Angle"].isna().all()

    def test_preserves_existing_columns(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(2))
        df["extra_col"] = 99
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        assert "extra_col" in angles_df.columns


class TestCalculateErrorMetrics:

    def test_returns_dict(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(4))
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        metrics = KinematicCalculator.calculate_error_metrics(angles_df)
        assert isinstance(metrics, dict)

    def test_jitter_keys_present(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(4))
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        metrics = KinematicCalculator.calculate_error_metrics(angles_df)
        assert "Shoulder_Angle_jitter" in metrics
        assert "Elbow_Angle_jitter" in metrics

    def test_stability_between_0_and_1(self):
        df = KinematicCalculator.landmarks_to_dataframe(_make_landmark_sequence(4))
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        metrics = KinematicCalculator.calculate_error_metrics(angles_df)
        for key in ["Shoulder_Angle_stability", "Elbow_Angle_stability"]:
            if key in metrics:
                assert 0.0 <= metrics[key] <= 1.0

    def test_empty_dataframe_returns_empty_dict(self):
        metrics = KinematicCalculator.calculate_error_metrics(pd.DataFrame())
        assert metrics == {}

    def test_all_nan_angles_skipped(self):
        df = pd.DataFrame({"Shoulder_Angle": [float("nan"), float("nan")]})
        metrics = KinematicCalculator.calculate_error_metrics(df)
        assert "Shoulder_Angle_jitter" not in metrics
