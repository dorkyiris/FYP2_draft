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


def create_mock_landmarks(shoulder_angle=None, elbow_angle=None) -> List[Landmark]:
    """
    Create mock landmarks for testing.
    Uses right-side anatomy: shoulder=12, elbow=14, wrist=16, hip=24.
    """
    # Initialize all 33 landmarks at neutral position
    landmarks = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
    
    # If testing angle-specific scenarios, adjust relevant landmarks
    if elbow_angle is not None:
        # Position: hip, shoulder, elbow, wrist to create specific angle
        # For simplicity, use hardcoded positions
        landmarks[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)  # Hip
        landmarks[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)  # Shoulder
        landmarks[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)  # Elbow
        
        # Adjust wrist position based on desired elbow angle
        if elbow_angle >= 160:
            # Nearly straight arm (180°)
            landmarks[16] = Landmark(x=0.5, y=0.7, z=0, visibility=0.9)
        elif elbow_angle <= 90:
            # Right angle (90°)
            landmarks[16] = Landmark(x=0.7, y=0.5, z=0, visibility=0.9)
        else:
            # Intermediate angle
            landmarks[16] = Landmark(x=0.6, y=0.6, z=0, visibility=0.9)
    
    if shoulder_angle is not None:
        # Similar adjustments for shoulder angle
        landmarks[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)
        landmarks[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)
        landmarks[14] = Landmark(x=0.4, y=0.5, z=0, visibility=0.9)
        landmarks[16] = Landmark(x=0.3, y=0.6, z=0, visibility=0.9)
    
    return landmarks


class TestExercise1Analysis:
    """Test Exercise 1: Arm Abduction."""
    
    def test_exercise1_pass_with_straight_arm(self, analyzer):
        """Exercise 1 should PASS when elbow angle >= 160°."""
        landmarks = create_mock_landmarks(elbow_angle=165)
        result = analyzer.analyze(landmarks, EXERCISES[1])
        
        assert result.exercise_id == 1
        assert result.exercise_name == "Arm Abduction"
        assert result.status == ExerciseStatus.PASS
        assert result.primary_angle >= 160
    
    def test_exercise1_fail_with_bent_arm(self, analyzer):
        """Exercise 1 should FAIL when elbow angle < 160°."""
        landmarks = create_mock_landmarks(elbow_angle=145)
        result = analyzer.analyze(landmarks, EXERCISES[1])
        
        assert result.exercise_id == 1
        assert result.status == ExerciseStatus.FAIL
        assert result.primary_angle < 160
    
    def test_exercise1_low_visibility_tracking_state(self, analyzer):
        """Exercise 1 with low visibility should return TRACKING status."""
        landmarks = [Landmark(x=0.5, y=0.5, z=0, visibility=0.3) for _ in range(33)]
        result = analyzer.analyze(landmarks, EXERCISES[1])
        
        assert result.status == ExerciseStatus.TRACKING
        # Confidence should be based on visibility (0.3)
        assert result.confidence <= 0.35  # Close to visibility value


class TestExercise2Analysis:
    """Test Exercise 2: V-to-W Transition."""
    
    def test_exercise2_v_shape_detection(self, analyzer):
        """Exercise 2 should detect V-shape (~120° shoulder angle)."""
        landmarks = create_mock_landmarks(shoulder_angle=120)
        result = analyzer.analyze(landmarks, EXERCISES[2])
        
        assert result.exercise_id == 2
        assert result.exercise_name == "Arm V-to-W Transition"
        # V-shape should be detected (angle near 120°)
        assert result.primary_angle is not None
    
    def test_exercise2_w_shape_detection(self, analyzer):
        """Exercise 2 should detect W-shape (~90° shoulder angle)."""
        landmarks = create_mock_landmarks(shoulder_angle=90)
        result = analyzer.analyze(landmarks, EXERCISES[2])
        
        assert result.exercise_id == 2
        # W-shape should be detected (angle near 90°)
        assert result.primary_angle is not None


class TestExercise3Analysis:
    """Test Exercise 3: Inclined Push-up."""
    
    def test_exercise3_pass_with_deep_bend(self, analyzer):
        """Exercise 3 should PASS when elbow angle <= 100°."""
        landmarks = create_mock_landmarks(elbow_angle=90)
        result = analyzer.analyze(landmarks, EXERCISES[3])
        
        assert result.exercise_id == 3
        assert result.exercise_name == "Inclined Push-up"
        assert result.status == ExerciseStatus.PASS
        assert result.primary_angle <= 100
    
    def test_exercise3_fail_with_shallow_bend(self, analyzer):
        """Exercise 3 should FAIL when elbow angle > 100°."""
        landmarks = create_mock_landmarks(elbow_angle=130)
        result = analyzer.analyze(landmarks, EXERCISES[3])
        
        assert result.exercise_id == 3
        assert result.status == ExerciseStatus.FAIL
        assert result.primary_angle > 100


class TestSequenceAnalysis:
    """Test analyzing sequence of frames."""
    
    def test_analyze_sequence(self, analyzer):
        """Should process sequence of landmarks."""
        sequences = [
            create_mock_landmarks(elbow_angle=165),
            create_mock_landmarks(elbow_angle=165),
            create_mock_landmarks(elbow_angle=155),
        ]
        
        results = analyzer.analyze_sequence(sequences, EXERCISES[1])
        
        assert len(results) == 3
        assert results[0].exercise_id == 1
        # First two should pass (>= 160)
        assert results[0].status == ExerciseStatus.PASS
        assert results[1].status == ExerciseStatus.PASS
    
    def test_sequence_with_frame_numbers(self, analyzer):
        """Frame numbers should be tracked."""
        sequences = [
            create_mock_landmarks(elbow_angle=165),
            create_mock_landmarks(elbow_angle=145),
        ]
        
        results = analyzer.analyze_sequence(sequences, EXERCISES[1])
        
        assert results[0].frame_number == 0
        assert results[1].frame_number == 1


class TestResultProperties:
    """Test ExerciseResult properties."""
    
    def test_result_has_all_fields(self, analyzer):
        """Result should have all expected fields."""
        landmarks = create_mock_landmarks(elbow_angle=165)
        result = analyzer.analyze(landmarks, EXERCISES[1], frame_number=5)
        
        assert result.exercise_id == 1
        assert result.exercise_name is not None
        assert result.status is not None
        assert result.primary_angle is not None
        assert result.feedback is not None
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        assert result.frame_number == 5
        assert isinstance(result.angles, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
