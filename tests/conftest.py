"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
from typing import List

from rehabilitationcore.models import Landmark


# ---------------------------------------------------------------------------
# Landmark factories
# ---------------------------------------------------------------------------

def make_landmarks(vis: float = 0.9) -> List[Landmark]:
    """33 landmarks all at centre with given visibility."""
    return [Landmark(x=0.5, y=0.5, z=0.0, visibility=vis) for _ in range(33)]


def make_straight_arm_landmarks() -> List[Landmark]:
    """Landmarks positioned for a straight elbow (~180°) and raised shoulder (~90°)."""
    lms = make_landmarks()
    lms = list(lms)
    lms[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)  # Hip
    lms[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)  # Shoulder
    lms[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)  # Elbow (inline)
    lms[16] = Landmark(x=0.5, y=0.7, z=0, visibility=0.9)  # Wrist (inline → ~180° elbow)
    return lms


def make_bent_arm_landmarks() -> List[Landmark]:
    """Landmarks positioned for a ~90° bent elbow and low shoulder angle."""
    lms = make_landmarks()
    lms = list(lms)
    lms[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)
    lms[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)  # Shoulder
    lms[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)  # Elbow
    lms[16] = Landmark(x=0.7, y=0.5, z=0, visibility=0.9)  # Wrist (90° bend)
    return lms


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_arm():
    return make_straight_arm_landmarks()


@pytest.fixture
def bent_arm():
    return make_bent_arm_landmarks()


@pytest.fixture
def low_vis_landmarks():
    return make_landmarks(vis=0.1)


@pytest.fixture
def sample_landmarks_dataframe():
    """DataFrame with pre-computed landmark coordinates (3 frames)."""
    rows = []
    positions = [
        # frame 0 — straight arm
        {24: (0.5, 0.0), 12: (0.5, 0.3), 14: (0.5, 0.5), 16: (0.5, 0.7)},
        # frame 1 — slight variation
        {24: (0.5, 0.0), 12: (0.5, 0.31), 14: (0.5, 0.51), 16: (0.5, 0.71)},
        # frame 2 — bent arm
        {24: (0.5, 0.0), 12: (0.5, 0.3), 14: (0.5, 0.5), 16: (0.7, 0.5)},
    ]
    for frame_idx, lm_pos in enumerate(positions):
        row = {"frame": frame_idx}
        for lm_idx, (x, y) in lm_pos.items():
            row[f"Lm{lm_idx}_x"] = x
            row[f"Lm{lm_idx}_y"] = y
            row[f"Lm{lm_idx}_z"] = 0.0
            row[f"Lm{lm_idx}_visibility"] = 0.9
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def landmark_sequence():
    """Three-frame sequence for sequence analysis tests."""
    return [
        make_straight_arm_landmarks(),
        make_straight_arm_landmarks(),
        make_bent_arm_landmarks(),
    ]


@pytest.fixture
def sample_landmarks_csv(tmp_path):
    """Write sample_landmarks_dataframe to a CSV file."""
    csv_file = tmp_path / "sample_landmarks.csv"
    rows = []
    for i in range(3):
        row = {"frame": i}
        for lm_idx in [12, 14, 16, 24]:
            row[f"Lm{lm_idx}_x"] = 0.5
            row[f"Lm{lm_idx}_y"] = 0.3 + i * 0.01
            row[f"Lm{lm_idx}_z"] = 0.0
            row[f"Lm{lm_idx}_visibility"] = 0.9
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    return csv_file


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "requires_video: marks tests requiring video files")
