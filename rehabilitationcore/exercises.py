"""
Exercise definitions for rehabilitation system.
Exercises are dynamically loaded from YAML configuration.
Each exercise specifies angles, thresholds, and clinical parameters.
"""

from typing import Dict, Any
from .models import (
    ExerciseDefinition,
    AngleThreshold,
    ExerciseStatus,
)
from config.loader import ConfigManager


# MediaPipe pose landmark indices (right side for consistency)
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_detector
LANDMARK_INDICES = {
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
    "right_hip": 24,
    "left_hip": 23,
}


def _build_exercises_from_config() -> Dict[int, ExerciseDefinition]:
    """
    Dynamically load all exercise definitions from YAML configuration.
    Maps YAML threshold parameters directly to AngleThreshold dataclass.
    """
    config_manager = ConfigManager()
    exercises_dict = {}
    
    for ex_id, ex_config in config_manager.exercises.items():
        thresholds = {}
        
        # Build thresholds from YAML configuration
        for angle_name, threshold_config in ex_config.get("thresholds", {}).items():
            threshold_type = threshold_config.get("type", "minimum")
            threshold_value = threshold_config.get("value")
            
            # Map threshold type to min/max values
            min_val = None
            max_val = None
            
            if threshold_type == "minimum":
                min_val = threshold_value
            elif threshold_type == "maximum":
                max_val = threshold_value
            elif threshold_type == "range":
                min_val = threshold_config.get("min")
                max_val = threshold_config.get("max")
            
            thresholds[angle_name] = AngleThreshold(
                name=angle_name,
                min_value=min_val,
                max_value=max_val,
                target_value=threshold_config.get("target"),
                feedback_pass=threshold_config.get("feedback_pass"),
                feedback_fail=threshold_config.get("feedback_fail"),
            )
        
        # Create ExerciseDefinition from config
        exercises_dict[ex_id] = ExerciseDefinition(
            exercise_id=ex_id,
            name=ex_config.get("name", f"Exercise {ex_id}"),
            description=ex_config.get("description", ""),
            landmarks_required=ex_config.get("landmarks", []),
            primary_angles=list(thresholds.keys()),
            angle_thresholds=thresholds,
            feedback_rules={},
        )
    
    return exercises_dict


# Global registry of exercises loaded from config
EXERCISES: Dict[int, ExerciseDefinition] = _build_exercises_from_config()


def get_exercise(exercise_id: int) -> ExerciseDefinition:
    """Get exercise definition by ID."""
    if exercise_id not in EXERCISES:
        raise ValueError(
            f"Unknown exercise ID: {exercise_id}. "
            f"Available: {list(EXERCISES.keys())}"
        )
    return EXERCISES[exercise_id]


def list_exercises():
    """List all available exercises."""
    return [
        (ex_id, ex.name, ex.description)
        for ex_id, ex in sorted(EXERCISES.items())
    ]
