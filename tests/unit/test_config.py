"""Tests for configuration management system (Phase 2)."""

import pytest
from config.loader import ConfigManager
from rehabilitationcore.exercises import EXERCISES


class TestConfigManager:
    """Test ConfigManager loading and accessing configurations."""
    
    def test_config_manager_loads_exercises(self):
        """ConfigManager should load exercises from YAML."""
        config = ConfigManager()
        assert len(config.exercises) > 0
    
    def test_config_has_original_exercises(self):
        """Config should include original 3 exercises for backward compatibility."""
        config = ConfigManager()
        assert 1 in config.exercises
        assert 2 in config.exercises
        assert 3 in config.exercises
    
    def test_config_has_new_clinical_exercises(self):
        """Config should include new 4 clinical exercises."""
        config = ConfigManager()
        assert 4 in config.exercises
        assert 5 in config.exercises
        assert 6 in config.exercises
        assert 7 in config.exercises
    
    def test_get_exercise_by_id(self):
        """Should retrieve exercise configuration by ID."""
        config = ConfigManager()
        ex1 = config.get_exercise(1)
        assert ex1["name"] == "Arm Abduction"
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
        threshold = config.get_threshold(1, "elbow")
        assert threshold["type"] == "minimum"
        assert threshold["value"] == 160.0
    
    def test_get_threshold_invalid_angle(self):
        """Should raise ValueError for invalid angle."""
        config = ConfigManager()
        with pytest.raises(ValueError):
            config.get_threshold(1, "nonexistent_angle")
    
    def test_exercise_definitions_loaded_from_config(self):
        """EXERCISES registry should be built from config."""
        # Original exercises
        assert 1 in EXERCISES
        assert 2 in EXERCISES
        assert 3 in EXERCISES
        
        # New exercises
        assert 4 in EXERCISES
        assert 5 in EXERCISES
        assert 6 in EXERCISES
        assert 7 in EXERCISES
    
    def test_exercise_1_backward_compatible(self):
        """Exercise 1 should maintain backward compatibility."""
        ex = EXERCISES[1]
        assert ex.name == "Arm Abduction"
        assert ex.exercise_id == 1
        assert "elbow" in ex.angle_thresholds
        assert ex.angle_thresholds["elbow"].min_value == 160.0
    
    def test_exercise_2_backward_compatible(self):
        """Exercise 2 should maintain backward compatibility."""
        ex = EXERCISES[2]
        assert ex.name == "Arm V-to-W Transition"
        assert ex.exercise_id == 2
        assert "shoulder" in ex.angle_thresholds
        assert ex.angle_thresholds["shoulder"].min_value == 85.0
        assert ex.angle_thresholds["shoulder"].max_value == 125.0
    
    def test_exercise_3_backward_compatible(self):
        """Exercise 3 should maintain backward compatibility."""
        ex = EXERCISES[3]
        assert ex.name == "Inclined Push-up"
        assert ex.exercise_id == 3
        assert "elbow" in ex.angle_thresholds
        assert ex.angle_thresholds["elbow"].max_value == 100.0
    
    def test_new_exercise_4_clinical(self):
        """New Exercise 4 should be clinical lift exercise."""
        ex = EXERCISES[4]
        assert ex.name == "Lifting an object"
        assert ex.description == "Shoulder flexion and elbow tracking while lifting"
        assert "shoulder" in ex.angle_thresholds
        assert ex.angle_thresholds["shoulder"].min_value == 90.0
    
    def test_new_exercise_5_clinical(self):
        """New Exercise 5 should be elbow extension exercise."""
        ex = EXERCISES[5]
        assert ex.name == "Extending the elbow"
        assert "elbow" in ex.angle_thresholds
        assert ex.angle_thresholds["elbow"].min_value == 160.0
    
    def test_new_exercise_6_clinical(self):
        """New Exercise 6 should be wrist lift exercise."""
        ex = EXERCISES[6]
        assert ex.name == "Lifting the wrist"
        assert "wrist" in ex.angle_thresholds
        assert ex.angle_thresholds["wrist"].min_value == 15.0
    
    def test_new_exercise_7_clinical(self):
        """New Exercise 7 should be hand opening exercise."""
        ex = EXERCISES[7]
        assert ex.name == "Opening the hand"
        assert "hand_open" in ex.angle_thresholds
        assert ex.angle_thresholds["hand_open"].min_value == 45.0
    
    def test_exercise_landmarks_loaded(self):
        """Exercises should load landmark indices from config."""
        ex = EXERCISES[1]
        assert ex.landmarks_required == [12, 14, 16, 24]
    
    def test_threshold_mapping_minimum_type(self):
        """Threshold type 'minimum' should map to min_value."""
        ex = EXERCISES[4]
        threshold = ex.angle_thresholds["shoulder"]
        assert threshold.min_value == 90.0
        assert threshold.max_value is None
    
    def test_threshold_mapping_maximum_type(self):
        """Threshold type 'maximum' should map to max_value."""
        ex = EXERCISES[3]
        threshold = ex.angle_thresholds["elbow"]
        assert threshold.max_value == 100.0
        assert threshold.min_value is None
    
    def test_threshold_mapping_range_type(self):
        """Threshold type 'range' should map to both min and max."""
        ex = EXERCISES[2]
        threshold = ex.angle_thresholds["shoulder"]
        assert threshold.min_value == 85.0
        assert threshold.max_value == 125.0
        assert threshold.target_value == 105.0
    
    def test_feedback_messages_loaded(self):
        """Feedback messages should be loaded from config."""
        ex = EXERCISES[1]
        threshold = ex.angle_thresholds["elbow"]
        assert threshold.feedback_pass == "✅ Form: PASS (Arm Straight)"
        assert threshold.feedback_fail == "❌ FAIL: Keep Arm Straight!"
