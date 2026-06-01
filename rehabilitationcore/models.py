"""
Data models for rehabilitation system.
Immutable, serializable objects for type safety and clarity.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class ExerciseStatus(str, Enum):
    """Clinical exercise status outcomes."""
    PASS = "PASS"
    FAIL = "FAIL"
    TRANSITIONING = "TRANSITIONING"
    TRACKING = "TRACKING"  # When detection quality is uncertain


@dataclass(frozen=True)
class Landmark:
    """Single pose landmark from MediaPipe or extracted data."""
    x: float  # Normalized 0-1 or pixel coordinates
    y: float  # Normalized 0-1 or pixel coordinates
    z: float = 0.0  # Depth (optional, for 3D)
    visibility: float = 1.0  # Confidence 0-1
    
    def to_tuple(self) -> tuple:
        """Convert to (x, y) tuple for calculations."""
        return (self.x, self.y)
    
    def is_visible(self, threshold: float = 0.65) -> bool:
        """Check if landmark meets visibility threshold."""
        return self.visibility >= threshold


@dataclass(frozen=True)
class AngleThreshold:
    """Threshold configuration for a single angle measurement."""
    name: str  # e.g., "elbow_extension"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None  # For transitioning states
    feedback_pass: str = "✅ PASS"
    feedback_fail: str = "❌ FAIL"
    feedback_transitioning: str = "⏳ Transitioning"
    
    def evaluate(self, angle: float) -> ExerciseStatus:
        """Determine status based on angle value."""
        if self.min_value is not None and angle < self.min_value:
            return ExerciseStatus.FAIL
        if self.max_value is not None and angle > self.max_value:
            return ExerciseStatus.FAIL
        
        # Transitioning: between target and threshold
        if self.target_value is not None:
            if abs(angle - self.target_value) > 15:
                return ExerciseStatus.TRANSITIONING
        
        return ExerciseStatus.PASS


@dataclass(frozen=True)
class ExerciseDefinition:
    """Clinical exercise specification and parameters."""
    exercise_id: int  # 1, 2, 3, etc.
    name: str  # e.g., "Arm Abduction"
    description: str
    landmarks_required: List[int]  # MediaPipe landmark indices [12, 14, 16, 24]
    primary_angles: List[str]  # e.g., ["shoulder", "elbow"]
    angle_thresholds: Dict[str, AngleThreshold]  # angle_name -> threshold config
    feedback_rules: Dict[ExerciseStatus, str]  # Override default feedback


@dataclass(frozen=True)
class ExerciseResult:
    """Immutable result of exercise analysis."""
    exercise_id: int
    exercise_name: str
    status: ExerciseStatus
    primary_angle: float  # Main angle being evaluated
    secondary_angle: Optional[float] = None
    angles: Dict[str, float] = None  # All calculated angles
    feedback: str = ""  # Clinical feedback message
    confidence: float = 1.0  # Detection confidence
    frame_number: Optional[int] = None  # Which frame this came from
    
    def __post_init__(self):
        if self.angles is None:
            object.__setattr__(self, "angles", {})
