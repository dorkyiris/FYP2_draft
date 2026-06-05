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

    def test_raises_value_error_for_missing_threshold(self):
        config = ConfigManager()
        with pytest.raises(ValueError):
            config.get_threshold(1, "nonexistent_angle")


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
