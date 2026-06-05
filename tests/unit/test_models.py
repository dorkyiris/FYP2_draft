"""Unit tests for data model classes."""

import pytest
from rehabilitationcore.models import (
    Landmark,
    AngleThreshold,
    ExerciseStatus,
    ExerciseDefinition,
    ExerciseResult,
)


class TestLandmarkModel:

    def test_to_tuple_returns_xy(self):
        lm = Landmark(x=0.3, y=0.7, z=0.1, visibility=0.9)
        assert lm.to_tuple() == (0.3, 0.7)

    def test_is_visible_above_threshold(self):
        lm = Landmark(x=0.5, y=0.5, z=0, visibility=0.8)
        assert lm.is_visible(0.65) is True

    def test_is_visible_below_threshold(self):
        lm = Landmark(x=0.5, y=0.5, z=0, visibility=0.3)
        assert lm.is_visible(0.65) is False

    def test_is_visible_at_exact_threshold(self):
        lm = Landmark(x=0.5, y=0.5, z=0, visibility=0.65)
        assert lm.is_visible(0.65) is True

    def test_default_visibility(self):
        lm = Landmark(x=0.0, y=0.0)
        assert lm.visibility == 1.0

    def test_frozen(self):
        lm = Landmark(x=0.5, y=0.5)
        with pytest.raises(Exception):
            lm.x = 0.9  # frozen dataclass should raise


class TestAngleThresholdEvaluate:

    def test_minimum_pass(self):
        t = AngleThreshold(name="elbow", min_value=160.0)
        assert t.evaluate(165.0) == ExerciseStatus.PASS

    def test_minimum_fail(self):
        t = AngleThreshold(name="elbow", min_value=160.0)
        assert t.evaluate(150.0) == ExerciseStatus.FAIL

    def test_minimum_exact_boundary(self):
        t = AngleThreshold(name="elbow", min_value=160.0)
        assert t.evaluate(160.0) == ExerciseStatus.PASS

    def test_maximum_pass(self):
        t = AngleThreshold(name="elbow", max_value=100.0)
        assert t.evaluate(90.0) == ExerciseStatus.PASS

    def test_maximum_fail(self):
        t = AngleThreshold(name="elbow", max_value=100.0)
        assert t.evaluate(110.0) == ExerciseStatus.FAIL

    def test_maximum_exact_boundary(self):
        t = AngleThreshold(name="elbow", max_value=100.0)
        assert t.evaluate(100.0) == ExerciseStatus.PASS

    def test_range_within_returns_pass(self):
        # target=105, within 15° means PASS
        t = AngleThreshold(name="shoulder", min_value=85.0, max_value=125.0, target_value=105.0)
        assert t.evaluate(105.0) == ExerciseStatus.PASS

    def test_range_near_target_pass(self):
        t = AngleThreshold(name="shoulder", min_value=85.0, max_value=125.0, target_value=105.0)
        assert t.evaluate(110.0) == ExerciseStatus.PASS  # within 15° of target

    def test_range_far_from_target_transitioning(self):
        t = AngleThreshold(name="shoulder", min_value=85.0, max_value=125.0, target_value=105.0)
        assert t.evaluate(88.0) == ExerciseStatus.TRANSITIONING  # 17° from target

    def test_range_outside_min_fails(self):
        t = AngleThreshold(name="shoulder", min_value=85.0, max_value=125.0, target_value=105.0)
        assert t.evaluate(80.0) == ExerciseStatus.FAIL

    def test_range_outside_max_fails(self):
        t = AngleThreshold(name="shoulder", min_value=85.0, max_value=125.0, target_value=105.0)
        assert t.evaluate(130.0) == ExerciseStatus.FAIL

    def test_no_constraints_always_pass(self):
        t = AngleThreshold(name="free")
        assert t.evaluate(0.0) == ExerciseStatus.PASS
        assert t.evaluate(90.0) == ExerciseStatus.PASS
        assert t.evaluate(180.0) == ExerciseStatus.PASS


class TestExerciseResult:

    def test_angles_defaults_to_empty_dict(self):
        result = ExerciseResult(
            exercise_id=1,
            exercise_name="Test",
            status=ExerciseStatus.PASS,
            primary_angle=90.0,
        )
        assert result.angles == {}

    def test_result_frozen(self):
        result = ExerciseResult(
            exercise_id=1,
            exercise_name="Test",
            status=ExerciseStatus.PASS,
            primary_angle=90.0,
        )
        with pytest.raises(Exception):
            result.exercise_id = 2
