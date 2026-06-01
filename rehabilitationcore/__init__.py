"""
Rehabilitation Core: Pure biomechanical engine for exercise analysis.
Separates clinical logic from UI/visualization.
"""

from .models import (
    Landmark,
    ExerciseDefinition,
    ExerciseResult,
    AngleThreshold,
    ExerciseStatus,
)
from .biomechanics import calculate_2d_angle
from .exercises import EXERCISES, get_exercise, list_exercises
from .analyzer import ExerciseAnalyzer

__version__ = "1.0.0"
__all__ = [
    "Landmark",
    "ExerciseDefinition",
    "ExerciseResult",
    "AngleThreshold",
    "ExerciseStatus",
    "ExerciseAnalyzer",
    "calculate_2d_angle",
    "EXERCISES",
    "get_exercise",
    "list_exercises",
]
