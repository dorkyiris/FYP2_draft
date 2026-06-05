"""
Integration tests: config → exercises → analyzer full pipeline.
These tests verify the modules work together end-to-end without mocking.
"""

import pytest
from rehabilitationcore.models import ExerciseStatus, Landmark
from rehabilitationcore.exercises import EXERCISES, get_exercise
from rehabilitationcore.analyzer import ExerciseAnalyzer
from rehabilitationcore.errors import ExerciseNotFoundError
from config.loader import ConfigManager
from video.calculator import KinematicCalculator


@pytest.fixture
def analyzer():
    return ExerciseAnalyzer(min_visibility=0.65)


# ---------------------------------------------------------------------------
# Config → EXERCISES registry
# ---------------------------------------------------------------------------

class TestConfigToExercisesIntegration:

    def test_all_yaml_exercises_in_registry(self):
        """Every exercise in exercises.yaml should appear in EXERCISES."""
        cfg = ConfigManager()
        for ex_id in cfg.exercises:
            assert ex_id in EXERCISES

    def test_exercise_names_match_yaml(self):
        cfg = ConfigManager()
        for ex_id, ex_cfg in cfg.exercises.items():
            assert EXERCISES[ex_id].name == ex_cfg["name"]

    def test_thresholds_correctly_mapped(self):
        """Threshold values from YAML should match AngleThreshold objects."""
        cfg = ConfigManager()
        # Exercise 1: shoulder minimum 90°
        t = EXERCISES[1].angle_thresholds["shoulder"]
        assert t.min_value == cfg.get_threshold(1, "shoulder")["value"]

    def test_exercise_2_elbow_threshold_mapped(self):
        t = EXERCISES[2].angle_thresholds["elbow"]
        assert t.min_value == 160.0
        assert t.max_value is None

    def test_landmark_indices_from_yaml(self):
        cfg = ConfigManager()
        for ex_id, ex_cfg in cfg.exercises.items():
            assert EXERCISES[ex_id].landmarks_required == ex_cfg["landmarks"]


# ---------------------------------------------------------------------------
# Exercises → Analyzer full pipeline
# ---------------------------------------------------------------------------

class TestAnalyzerPipelineIntegration:

    def test_exercise1_full_pipeline_pass(self, analyzer):
        """Exercise 1: raised shoulder arm should PASS."""
        lms = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        lms = list(lms)
        lms[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)
        lms[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)
        lms[14] = Landmark(x=0.7, y=0.3, z=0, visibility=0.9)  # arm out (90°+)
        lms[16] = Landmark(x=0.8, y=0.3, z=0, visibility=0.9)

        result = analyzer.analyze(lms, get_exercise(1))

        assert result.exercise_id == 1
        assert result.status == ExerciseStatus.PASS
        assert result.confidence > 0.0

    def test_exercise2_full_pipeline_pass(self, analyzer):
        """Exercise 2: straight elbow should PASS."""
        lms = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        lms = list(lms)
        lms[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)
        lms[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)
        lms[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)
        lms[16] = Landmark(x=0.5, y=0.7, z=0, visibility=0.9)  # inline → ~180°

        result = analyzer.analyze(lms, get_exercise(2))

        assert result.exercise_id == 2
        assert result.status == ExerciseStatus.PASS

    def test_exercise3_returns_result_object(self, analyzer):
        """Exercise 3 (wrist) should return a valid result even if TRACKING."""
        lms = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        result = analyzer.analyze(lms, get_exercise(3))
        assert result.exercise_id == 3
        assert result.status in list(ExerciseStatus)

    def test_exercise4_returns_result_object(self, analyzer):
        """Exercise 4 (hand) should return a valid result even if TRACKING."""
        lms = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        result = analyzer.analyze(lms, get_exercise(4))
        assert result.exercise_id == 4

    def test_bad_exercise_id_raises(self):
        with pytest.raises(ExerciseNotFoundError):
            get_exercise(99)

    def test_confidence_is_mean_landmark_visibility(self, analyzer):
        """Confidence should equal mean visibility of required landmarks."""
        vis = 0.8
        lms = [Landmark(x=0.5, y=0.5, z=0, visibility=vis) for _ in range(33)]
        lms = list(lms)
        lms[24] = Landmark(x=0.5, y=0.0, z=0, visibility=vis)
        lms[12] = Landmark(x=0.5, y=0.3, z=0, visibility=vis)
        lms[14] = Landmark(x=0.7, y=0.3, z=0, visibility=vis)
        lms[16] = Landmark(x=0.8, y=0.3, z=0, visibility=vis)
        result = analyzer.analyze(lms, get_exercise(1))
        assert abs(result.confidence - vis) < 0.01


# ---------------------------------------------------------------------------
# Sequence analysis integration
# ---------------------------------------------------------------------------

class TestSequencePipelineIntegration:

    def _straight(self):
        lms = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        lms = list(lms)
        lms[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)
        lms[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)
        lms[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)
        lms[16] = Landmark(x=0.5, y=0.7, z=0, visibility=0.9)
        return lms

    def _bent(self):
        lms = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        lms = list(lms)
        lms[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)
        lms[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)
        lms[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)
        lms[16] = Landmark(x=0.7, y=0.5, z=0, visibility=0.9)
        return lms

    def test_sequence_length_preserved(self):
        analyzer = ExerciseAnalyzer()
        seq = [self._straight(), self._bent(), self._straight()]
        results = analyzer.analyze_sequence(seq, get_exercise(2))
        assert len(results) == 3

    def test_sequence_frame_numbers_sequential(self):
        analyzer = ExerciseAnalyzer()
        seq = [self._straight(), self._bent()]
        results = analyzer.analyze_sequence(seq, get_exercise(2))
        assert results[0].frame_number == 0
        assert results[1].frame_number == 1

    def test_sequence_mixed_pass_fail(self):
        analyzer = ExerciseAnalyzer()
        seq = [self._straight(), self._bent()]
        results = analyzer.analyze_sequence(seq, get_exercise(2))
        assert results[0].status == ExerciseStatus.PASS
        assert results[1].status == ExerciseStatus.FAIL


# ---------------------------------------------------------------------------
# KinematicCalculator integration
# ---------------------------------------------------------------------------

class TestCalculatorPipelineIntegration:

    def _make_seq(self, n=4):
        frames = []
        for _ in range(n):
            lms = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
            lms = list(lms)
            lms[24] = Landmark(x=0.5, y=0.0, z=0, visibility=0.9)
            lms[12] = Landmark(x=0.5, y=0.3, z=0, visibility=0.9)
            lms[14] = Landmark(x=0.5, y=0.5, z=0, visibility=0.9)
            lms[16] = Landmark(x=0.5, y=0.7, z=0, visibility=0.9)
            frames.append(lms)
        return frames

    def test_full_calculator_pipeline(self):
        """landmarks → DataFrame → angles → metrics all succeed."""
        seq = self._make_seq(5)
        df = KinematicCalculator.landmarks_to_dataframe(seq)
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        metrics = KinematicCalculator.calculate_error_metrics(angles_df)

        assert "Elbow_Angle" in angles_df.columns
        assert "Shoulder_Angle" in angles_df.columns
        assert isinstance(metrics, dict)

    def test_calculator_output_feeds_analyzer(self):
        """DataFrames produced by KinematicCalculator should be consistent
        with expected angle ranges."""
        seq = self._make_seq(3)
        df = KinematicCalculator.landmarks_to_dataframe(seq)
        angles_df = KinematicCalculator.extract_kinematic_angles(df)
        # Straight arm: elbow angle should be near 180°
        assert angles_df["Elbow_Angle"].mean() > 160
