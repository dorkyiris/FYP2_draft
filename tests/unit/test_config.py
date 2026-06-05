"""Tests for configuration management system (Phase 2)."""

import pytest
from config.loader import ConfigManager
from rehabilitationcore.exercises import EXERCISES


class TestConfigManager:
    """Test ConfigManager loading and accessing configurations."""

    def test_config_manager_loads_exercises(self):
        """ConfigManager should load all 4 exercises from YAML."""
        config = ConfigManager()
        assert len(config.exercises) == 4

    def test_config_has_all_four_exercises(self):
        """Config should include all 4 clinical exercises."""
        config = ConfigManager()
        assert 1 in config.exercises
        assert 2 in config.exercises
        assert 3 in config.exercises
        assert 4 in config.exercises

    def test_get_exercise_by_id(self):
        """Should retrieve exercise configuration by ID."""
        config = ConfigManager()
        ex1 = config.get_exercise(1)
        assert ex1["name"] == "Lifting an object"
        assert "thresholds" in ex1
        assert "landmarks" in ex1

    def test_get_exercise_invalid_id(self):
        """Should raise ValueError for invalid exercise ID."""
        config = ConfigManager()
        with pytest.raises(ValueError):
            config.get_exercise(999)

    def test_get_threshold_by_angle(self):
        """Should retrieve threshold configuration for specific angle."""
        config = ConfigManager()
        threshold = config.get_threshold(1, "shoulder")
        assert threshold["type"] == "minimum"
        assert threshold["value"] == 90.0

    def test_get_threshold_invalid_angle(self):
        """Should raise ValueError for invalid angle."""
        config = ConfigManager()
        with pytest.raises(ValueError):
            config.get_threshold(1, "nonexistent_angle")

    def test_exercise_definitions_loaded_from_config(self):
        """EXERCISES registry should be built from config with all 4 exercises."""
        assert 1 in EXERCISES
        assert 2 in EXERCISES
        assert 3 in EXERCISES
        assert 4 in EXERCISES
        assert 5 not in EXERCISES

    def test_exercise_1_lifting_object(self):
        """Exercise 1 should be Lifting an object with shoulder threshold."""
        ex = EXERCISES[1]
        assert ex.name == "Lifting an object"
        assert ex.exercise_id == 1
        assert "shoulder" in ex.angle_thresholds
        assert ex.angle_thresholds["shoulder"].min_value == 90.0
        assert ex.angle_thresholds["shoulder"].max_value is None

    def test_exercise_2_extending_elbow(self):
        """Exercise 2 should be Extending the elbow with elbow threshold."""
        ex = EXERCISES[2]
        assert ex.name == "Extending the elbow"
        assert ex.exercise_id == 2
        assert "elbow" in ex.angle_thresholds
        assert ex.angle_thresholds["elbow"].min_value == 160.0
        assert ex.angle_thresholds["elbow"].max_value is None

    def test_exercise_3_lifting_wrist(self):
        """Exercise 3 should be Lifting the wrist with wrist threshold."""
        ex = EXERCISES[3]
        assert ex.name == "Lifting the wrist"
        assert ex.exercise_id == 3
        assert "wrist" in ex.angle_thresholds
        assert ex.angle_thresholds["wrist"].min_value == 15.0

    def test_exercise_4_opening_hand(self):
        """Exercise 4 should be Opening the hand with hand_open threshold."""
        ex = EXERCISES[4]
        assert ex.name == "Opening the hand"
        assert ex.exercise_id == 4
        assert "hand_open" in ex.angle_thresholds
        assert ex.angle_thresholds["hand_open"].min_value == 45.0

    def test_exercise_landmarks_loaded(self):
        """Exercises should load landmark indices from config."""
        assert EXERCISES[1].landmarks_required == [12, 14, 16, 24]
        assert EXERCISES[3].landmarks_required == [14, 16, 18, 20]
        assert EXERCISES[4].landmarks_required == [16, 18, 20, 22]

    def test_threshold_mapping_minimum_type(self):
        """Threshold type 'minimum' should map to min_value only."""
        threshold = EXERCISES[1].angle_thresholds["shoulder"]
        assert threshold.min_value == 90.0
        assert threshold.max_value is None

    def test_threshold_mapping_elbow_minimum(self):
        """Exercise 2 elbow threshold should have min_value only."""
        threshold = EXERCISES[2].angle_thresholds["elbow"]
        assert threshold.min_value == 160.0
        assert threshold.max_value is None

    def test_feedback_messages_loaded(self):
        """Feedback messages should be loaded from config."""
        threshold = EXERCISES[1].angle_thresholds["shoulder"]
        assert threshold.feedback_pass == "✅ Lift Height: PASS"
        assert threshold.feedback_fail == "❌ FAIL: Lift Higher!"

    def test_exercise_descriptions_loaded(self):
        """Exercise descriptions should be loaded from config."""
        assert EXERCISES[1].description == "Shoulder flexion and elbow tracking while lifting"
        assert EXERCISES[3].description == "Wrist extension/flexion tracking"
