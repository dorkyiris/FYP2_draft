"""Tests for Phase 3: error handling and logging."""

import logging
import pytest
from pathlib import Path

from rehabilitationcore.errors import (
    RehabSystemError,
    ConfigError,
    ExerciseNotFoundError,
    LandmarkError,
    AnalysisError,
)
from rehabilitationcore.logging_config import get_logger, configure_logging
from config.loader import ConfigManager


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:

    def test_exercise_not_found_is_config_error(self):
        assert issubclass(ExerciseNotFoundError, ConfigError)

    def test_config_error_is_rehab_error(self):
        assert issubclass(ConfigError, RehabSystemError)

    def test_landmark_error_is_rehab_error(self):
        assert issubclass(LandmarkError, RehabSystemError)

    def test_analysis_error_is_rehab_error(self):
        assert issubclass(AnalysisError, RehabSystemError)

    def test_all_errors_are_exceptions(self):
        assert issubclass(RehabSystemError, Exception)


# ---------------------------------------------------------------------------
# ExerciseNotFoundError
# ---------------------------------------------------------------------------

class TestExerciseNotFoundError:

    def test_message_contains_id(self):
        err = ExerciseNotFoundError(99)
        assert "99" in str(err)

    def test_message_lists_available(self):
        err = ExerciseNotFoundError(99, available=[1, 2, 3])
        assert "1" in str(err)
        assert "2" in str(err)

    def test_attributes_set(self):
        err = ExerciseNotFoundError(42, available=[1, 2])
        assert err.exercise_id == 42
        assert err.available == [1, 2]

    def test_no_available_list(self):
        err = ExerciseNotFoundError(5)
        assert err.available == []
        assert "5" in str(err)


# ---------------------------------------------------------------------------
# LandmarkError
# ---------------------------------------------------------------------------

class TestLandmarkError:

    def test_message_stored(self):
        err = LandmarkError("visibility too low", landmark_idx=12, visibility=0.3)
        assert "visibility too low" in str(err)

    def test_attributes_set(self):
        err = LandmarkError("bad landmark", landmark_idx=14, visibility=0.1)
        assert err.landmark_idx == 14
        assert err.visibility == 0.1

    def test_optional_attributes_default_none(self):
        err = LandmarkError("generic error")
        assert err.landmark_idx is None
        assert err.visibility is None


# ---------------------------------------------------------------------------
# ConfigManager error paths
# ---------------------------------------------------------------------------

class TestConfigManagerErrors:

    def test_raises_exercise_not_found_for_bad_id(self):
        config = ConfigManager()
        with pytest.raises(ExerciseNotFoundError) as exc_info:
            config.get_exercise(999)
        assert exc_info.value.exercise_id == 999

    def test_error_includes_available_ids(self):
        config = ConfigManager()
        with pytest.raises(ExerciseNotFoundError) as exc_info:
            config.get_exercise(999)
        assert exc_info.value.available == [1, 2, 3, 4]

    def test_raises_config_error_for_missing_yaml(self, tmp_path):
        with pytest.raises(ConfigError):
            ConfigManager(config_dir=str(tmp_path))

    def test_raises_config_error_for_malformed_yaml(self, tmp_path):
        bad_yaml = tmp_path / "exercises.yaml"
        bad_yaml.write_text("exercises: :::bad:::yaml:::")
        with pytest.raises(ConfigError, match="Failed to parse"):
            ConfigManager(config_dir=str(tmp_path))

    def test_raises_value_error_for_missing_threshold(self):
        config = ConfigManager()
        with pytest.raises(ValueError):
            config.get_threshold(1, "nonexistent_angle")


# ---------------------------------------------------------------------------
# rehabilitationcore.exercises
# ---------------------------------------------------------------------------

class TestExercisesModule:

    def test_get_exercise_raises_exercise_not_found_error(self):
        from rehabilitationcore.exercises import get_exercise
        with pytest.raises(ExerciseNotFoundError) as exc:
            get_exercise(999)
        assert exc.value.exercise_id == 999

    def test_get_exercise_error_includes_available(self):
        from rehabilitationcore.exercises import get_exercise
        with pytest.raises(ExerciseNotFoundError) as exc:
            get_exercise(999)
        assert sorted(exc.value.available) == [1, 2, 3, 4]

    def test_list_exercises_returns_all_four(self):
        from rehabilitationcore.exercises import list_exercises
        result = list_exercises()
        assert len(result) == 4

    def test_list_exercises_sorted_by_id(self):
        from rehabilitationcore.exercises import list_exercises
        result = list_exercises()
        ids = [item[0] for item in result]
        assert ids == sorted(ids)

    def test_list_exercises_contains_names(self):
        from rehabilitationcore.exercises import list_exercises
        result = list_exercises()
        names = [item[1] for item in result]
        assert "Lifting an object" in names
        assert "Extending the elbow" in names


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:

    def test_get_logger_returns_logger(self):
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)

    def test_get_logger_namespaced(self):
        log = get_logger("biomechanics")
        assert log.name == "rehabilitation.biomechanics"

    def test_configure_logging_idempotent(self):
        configure_logging()
        configure_logging()  # calling twice should not add duplicate handlers
        root = logging.getLogger("rehabilitation")
        assert len(root.handlers) <= 1

    def test_analyzer_logs_on_low_visibility(self, caplog):
        from rehabilitationcore.models import Landmark
        from rehabilitationcore.analyzer import ExerciseAnalyzer
        from rehabilitationcore.exercises import EXERCISES

        analyzer = ExerciseAnalyzer(min_visibility=0.65)
        landmarks = [Landmark(x=0.5, y=0.5, z=0, visibility=0.1) for _ in range(33)]

        with caplog.at_level(logging.WARNING, logger="rehabilitation"):
            analyzer.analyze(landmarks, EXERCISES[1])

        assert any("validation failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Coverage gap: exercises.py maximum/range threshold branches
# ---------------------------------------------------------------------------

class TestExercisesThresholdBranches:
    """Build exercises from a custom YAML to cover maximum and range branches."""

    def _write_yaml(self, tmp_path, content: str):
        f = tmp_path / "exercises.yaml"
        f.write_text(content)
        return tmp_path

    def test_maximum_threshold_type_covered(self, tmp_path):
        """The 'maximum' branch in _build_exercises_from_config should be hit."""
        self._write_yaml(tmp_path, """
exercises:
  1:
    name: "Push-up depth"
    description: "Elbow max test"
    landmarks: [12, 14, 16, 24]
    thresholds:
      elbow:
        type: "maximum"
        value: 100.0
        feedback_pass: "✅ Deep enough"
        feedback_fail: "❌ Go deeper"
""")
        cfg = ConfigManager(config_dir=str(tmp_path))
        assert cfg.exercises[1]["thresholds"]["elbow"]["type"] == "maximum"

        from rehabilitationcore.exercises import _build_exercises_from_config
        import unittest.mock as mock
        with mock.patch("rehabilitationcore.exercises.ConfigManager") as MockCfg:
            MockCfg.return_value.exercises = cfg.exercises
            result = _build_exercises_from_config()
        assert result[1].angle_thresholds["elbow"].max_value == 100.0
        assert result[1].angle_thresholds["elbow"].min_value is None

    def test_range_threshold_type_covered(self, tmp_path):
        """The 'range' branch in _build_exercises_from_config should be hit."""
        self._write_yaml(tmp_path, """
exercises:
  1:
    name: "V-to-W"
    description: "Shoulder range test"
    landmarks: [12, 14, 16, 24]
    thresholds:
      shoulder:
        type: "range"
        min: 85.0
        max: 125.0
        target: 105.0
        feedback_pass: "✅ Target"
        feedback_fail: "❌ Off target"
""")
        cfg = ConfigManager(config_dir=str(tmp_path))

        from rehabilitationcore.exercises import _build_exercises_from_config
        import unittest.mock as mock
        with mock.patch("rehabilitationcore.exercises.ConfigManager") as MockCfg:
            MockCfg.return_value.exercises = cfg.exercises
            result = _build_exercises_from_config()
        t = result[1].angle_thresholds["shoulder"]
        assert t.min_value == 85.0
        assert t.max_value == 125.0
        assert t.target_value == 105.0


# ---------------------------------------------------------------------------
# Coverage gap: config/loader.py system.yaml YAML parse error
# ---------------------------------------------------------------------------

class TestSystemYamlErrors:

    def test_malformed_system_yaml_raises_config_error(self, tmp_path):
        """Malformed system.yaml should raise ConfigError."""
        (tmp_path / "exercises.yaml").write_text("exercises:\n  1:\n    name: Test\n    landmarks: [12]\n    thresholds:\n      elbow:\n        type: minimum\n        value: 90\n")
        (tmp_path / "system.yaml").write_text("key: [unclosed bracket")
        with pytest.raises(ConfigError, match="Failed to parse system.yaml"):
            ConfigManager(config_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Coverage gap: analyzer.py line 94 — empty angles dict
# ---------------------------------------------------------------------------

class TestAnalyzerEmptyAngles:

    def test_empty_angles_returns_tracking(self, monkeypatch):
        """Force _calculate_angles to return {} to cover the empty-angles guard."""
        from rehabilitationcore.analyzer import ExerciseAnalyzer
        from rehabilitationcore.exercises import EXERCISES
        from rehabilitationcore.models import Landmark, ExerciseStatus

        analyzer = ExerciseAnalyzer(min_visibility=0.0)  # 0.0 so validation always passes
        monkeypatch.setattr(analyzer, "_calculate_angles", lambda *_: {})

        lms = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9) for _ in range(33)]
        result = analyzer.analyze(lms, EXERCISES[1])
        assert result.status == ExerciseStatus.TRACKING
