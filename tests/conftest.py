"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_landmarks_csv(tmp_path):
    """Create a sample landmarks CSV for testing."""
    csv_file = tmp_path / "sample_landmarks.csv"
    
    # Create minimal CSV with required columns
    csv_content = """frame,Lm12_x,Lm12_y,Lm14_x,Lm14_y,Lm16_x,Lm16_y,Lm24_x,Lm24_y
0,0.5,0.3,0.5,0.5,0.5,0.7,0.5,0.0
1,0.5,0.31,0.5,0.51,0.5,0.71,0.5,0.01
2,0.5,0.32,0.5,0.52,0.5,0.72,0.5,0.02
"""
    
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_video_path(tmp_path):
    """Path where a test video would be (doesn't create actual video)."""
    return str(tmp_path / "test_video.mp4")


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "requires_video: marks tests requiring video files (deselect with '-m \"not requires_video\"')"
    )
