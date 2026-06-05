"""
Unit tests for exercise analysis logic.
"""

import pytest
from typing import List

from rehabilitationcore.models import Landmark, ExerciseStatus
from rehabilitationcore.analyzer import ExerciseAnalyzer
from rehabilitationcore.exercises import EXERCISES


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return ExerciseAnalyzer(min_visibility=0.65)


def create_mock_landmarks(shoulder_angle_high=False, elbow_straight=False, bent_arm=False, low_vis=False) -> List[Landmark]:
    """
    Create mock landmarks for testing.
    Landmarks use MediaPipe indices: 12=shoulder, 14=elbow, 16=wrist, 24=hip.
    """
    vis = 0.2 if low_vis else 0.9
    landmarks = [Landmark(x=0.5, y=0.5, z=0, visibility=vis) for _ in range(33)]

    if shoulder_angle_high:
        # Arm raised so hip-shoulder-elbow angle >= 90°
        landmarks[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)  # Hip (low)
        landmarks[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)  # Shoulder
        landmarks[14] = Landmark(x=0.7, y=0.3, z=0, visibility=0.9)  # Elbow out to side (90°+)
        landmarks[16] = Landmark(x=0.8, y=0.3, z=0, visibility=0.9)  # Wrist

    elif bent_arm:
        # Arm dropped so hip-shoulder-elbow angle < 90°
        landmarks[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)  # Hip
        landmarks[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)  # Shoulder
        landmarks[14] = Landmark(x=0.5, y=0.1, z=0, visibility=0.9)  # Elbow (arm angled toward hip)
        landmarks[16] = Landmark(x=0.5, y=0.4, z=0, visibility=0.9)  # Wrist

    elif elbow_straight:
        # Straight arm: shoulder-elbow-wrist angle >= 160°
        landmarks[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)  # Hip
        landmarks[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)  # Shoulder
        landmarks[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)  # Elbow (inline)
        landmarks[16] = Landmark(x=0.5, y=0.7, z=0, visibility=0.9)  # Wrist (inline = 180°)

    elif not low_vis:
        # Bent elbow: shoulder-elbow-wrist angle ~ 90°
        landmarks[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)
        landmarks[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)  # Shoulder
        landmarks[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)  # Elbow
        landmarks[16] = Landmark(x=0.7, y=0.5, z=0, visibility=0.9)  # Wrist (90° bend)

    return landmarks


class TestExercise1Analysis:
    """Test Exercise 1: Lifting an object (shoulder angle >= 90°)."""

    def test_exercise1_pass_with_arm_raised(self, analyzer):
        """Exercise 1 should PASS when shoulder angle >= 90°."""
        landmarks = create_mock_landmarks(shoulder_angle_high=True)
        result = analyzer.analyze(landmarks, EXERCISES[1])

        assert result.exercise_id == 1
        assert result.exercise_name == "Lifting an object"
        assert result.status == ExerciseStatus.PASS
        assert result.primary_angle >= 90.0

    def test_exercise1_fail_with_arm_down(self, analyzer):
        """Exercise 1 should FAIL when shoulder angle < 90°."""
        landmarks = create_mock_landmarks(bent_arm=True)
        result = analyzer.analyze(landmarks, EXERCISES[1])

        assert result.exercise_id == 1
        assert result.status == ExerciseStatus.FAIL
        assert result.primary_angle < 90.0

    def test_exercise1_low_visibility_tracking_state(self, analyzer):
        """Exercise 1 with low visibility should return TRACKING status."""
        landmarks = create_mock_landmarks(low_vis=True)
        result = analyzer.analyze(landmarks, EXERCISES[1])

        assert result.status == ExerciseStatus.TRACKING
        assert result.confidence <= 0.35


class TestExercise2Analysis:
    """Test Exercise 2: Extending the elbow (elbow angle >= 160°)."""

    def test_exercise2_pass_with_straight_arm(self, analyzer):
        """Exercise 2 should PASS when elbow angle >= 160°."""
        landmarks = create_mock_landmarks(elbow_straight=True)
        result = analyzer.analyze(landmarks, EXERCISES[2])

        assert result.exercise_id == 2
        assert result.exercise_name == "Extending the elbow"
        assert result.status == ExerciseStatus.PASS
        assert result.primary_angle >= 160.0

    def test_exercise2_fail_with_bent_elbow(self, analyzer):
        """Exercise 2 should FAIL when elbow angle < 160°."""
        landmarks = create_mock_landmarks()  # default = ~90° bend
        result = analyzer.analyze(landmarks, EXERCISES[2])

        assert result.exercise_id == 2
        assert result.status == ExerciseStatus.FAIL
        assert result.primary_angle < 160.0

    def test_exercise2_low_visibility_tracking_state(self, analyzer):
        """Exercise 2 with low visibility should return TRACKING status."""
        landmarks = create_mock_landmarks(low_vis=True)
        result = analyzer.analyze(landmarks, EXERCISES[2])

        assert result.status == ExerciseStatus.TRACKING


class TestExercise3And4Tracking:
    """
    Exercises 3 and 4 use wrist/hand_open angle types.
    The current analyzer only computes shoulder and elbow angles,
    so these return TRACKING until the analyzer is extended.
    """

    def test_exercise3_returns_valid_result(self, analyzer):
        """Exercise 3 should return a valid result object."""
        landmarks = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        result = analyzer.analyze(landmarks, EXERCISES[3])

        assert result.exercise_id == 3
        assert result.exercise_name == "Lifting the wrist"
        assert result.status in list(ExerciseStatus)
        assert result.primary_angle is not None

    def test_exercise4_returns_valid_result(self, analyzer):
        """Exercise 4 should return a valid result object."""
        landmarks = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        result = analyzer.analyze(landmarks, EXERCISES[4])

        assert result.exercise_id == 4
        assert result.exercise_name == "Opening the hand"
        assert result.status in list(ExerciseStatus)
        assert result.primary_angle is not None


class TestSequenceAnalysis:
    """Test analyzing sequence of frames."""

    def test_analyze_sequence(self, analyzer):
        """Should process sequence of landmarks."""
        sequences = [
            create_mock_landmarks(elbow_straight=True),
            create_mock_landmarks(elbow_straight=True),
            create_mock_landmarks(),  # bent
        ]

        results = analyzer.analyze_sequence(sequences, EXERCISES[2])

        assert len(results) == 3
        assert results[0].exercise_id == 2
        assert results[0].status == ExerciseStatus.PASS
        assert results[1].status == ExerciseStatus.PASS
        assert results[2].status == ExerciseStatus.FAIL

    def test_sequence_with_frame_numbers(self, analyzer):
        """Frame numbers should be tracked in sequence results."""
        sequences = [
            create_mock_landmarks(elbow_straight=True),
            create_mock_landmarks(),
        ]

        results = analyzer.analyze_sequence(sequences, EXERCISES[2])

        assert results[0].frame_number == 0
        assert results[1].frame_number == 1


class TestResultProperties:
    """Test ExerciseResult properties."""

    def test_result_has_all_fields(self, analyzer):
        """Result should have all expected fields populated."""
        landmarks = create_mock_landmarks(elbow_straight=True)
        result = analyzer.analyze(landmarks, EXERCISES[2], frame_number=5)

        assert result.exercise_id == 2
        assert result.exercise_name is not None
        assert result.status is not None
        assert result.primary_angle is not None
        assert result.feedback is not None
        assert 0.0 <= result.confidence <= 1.0
        assert result.frame_number == 5
        assert isinstance(result.angles, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
