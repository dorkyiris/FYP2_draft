"""
Exercise analyzer: Analyze biomechanics and grade exercise performance.
"""

from typing import List, Dict, Optional
import logging

from .models import Landmark, ExerciseDefinition, ExerciseResult, ExerciseStatus
from .biomechanics import calculate_2d_angle, validate_landmark_chain

from .logging_config import get_logger

logger = get_logger("analyzer")


class ExerciseAnalyzer:
    """Analyze exercise performance from pose landmarks."""
    
    def __init__(self, min_visibility: float = 0.65):
        """Initialize analyzer with landmark visibility threshold."""
        self.min_visibility = min_visibility
    
    def _calculate_angles(
        self,
        landmarks: List[Landmark],
        landmark_indices: Dict[str, int],
    ) -> Dict[str, float]:
        """Calculate shoulder and elbow angles from landmarks."""
        angles = {}
        
        def get_landmark(idx: int) -> Optional[Landmark]:
            return landmarks[idx] if 0 <= idx < len(landmarks) else None
        
        # Shoulder angle (hip-shoulder-elbow)
        hip = get_landmark(landmark_indices.get("hip", 24))
        shoulder = get_landmark(landmark_indices.get("shoulder", 12))
        elbow = get_landmark(landmark_indices.get("elbow", 14))
        
        if all([hip, shoulder, elbow]) and all(
            lm.is_visible(self.min_visibility) for lm in [hip, shoulder, elbow]
        ):
            angles["shoulder"] = calculate_2d_angle(
                hip.to_tuple(), shoulder.to_tuple(), elbow.to_tuple()
            )
        
        # Elbow angle (shoulder-elbow-wrist)
        wrist = get_landmark(landmark_indices.get("wrist", 16))
        if all([shoulder, elbow, wrist]) and all(
            lm.is_visible(self.min_visibility) for lm in [shoulder, elbow, wrist]
        ):
            angles["elbow"] = calculate_2d_angle(
                shoulder.to_tuple(), elbow.to_tuple(), wrist.to_tuple()
            )
        
        return angles
    
    def analyze(
        self,
        landmarks: List[Landmark],
        exercise: ExerciseDefinition,
        frame_number: Optional[int] = None,
    ) -> ExerciseResult:
        """Analyze single frame of exercise."""
        # Validate landmarks
        is_valid, error_msg = validate_landmark_chain(
            landmarks, exercise.landmarks_required, self.min_visibility
        )
        
        if not is_valid:
            logger.warning(
                "Exercise %d (%s): landmark validation failed — %s",
                exercise.exercise_id, exercise.name, error_msg,
            )
            return ExerciseResult(
                exercise_id=exercise.exercise_id,
                exercise_name=exercise.name,
                status=ExerciseStatus.TRACKING,
                primary_angle=0.0,
                feedback=f"🔍 Tracking... {error_msg}",
                confidence=0.0,
                frame_number=frame_number,
            )
        
        # Calculate angles
        landmark_indices = {
            "hip": exercise.landmarks_required[3],
            "shoulder": exercise.landmarks_required[0],
            "elbow": exercise.landmarks_required[1],
            "wrist": exercise.landmarks_required[2],
        }
        angles = self._calculate_angles(landmarks, landmark_indices)
        
        if not angles:
            return ExerciseResult(
                exercise_id=exercise.exercise_id,
                exercise_name=exercise.name,
                status=ExerciseStatus.TRACKING,
                primary_angle=0.0,
                feedback="🔍 Insufficient landmark visibility",
                confidence=0.0,
                frame_number=frame_number,
            )
        
        # Get primary angle
        primary_angle_name = exercise.primary_angles[0]
        if primary_angle_name not in angles:
            logger.warning(
                "Exercise %d (%s): angle '%s' not computable — unsupported landmark set",
                exercise.exercise_id, exercise.name, primary_angle_name,
            )
            return ExerciseResult(
                exercise_id=exercise.exercise_id,
                exercise_name=exercise.name,
                status=ExerciseStatus.TRACKING,
                primary_angle=0.0,
                feedback=f"🔍 Cannot calculate {primary_angle_name} angle",
                confidence=0.0,
                frame_number=frame_number,
            )
        
        primary_angle = angles[primary_angle_name]
        
        # Evaluate threshold
        threshold = exercise.angle_thresholds[primary_angle_name]
        status = threshold.evaluate(primary_angle)
        
        # Generate feedback
        feedback = exercise.feedback_rules.get(status, f"{status.value}: {primary_angle:.1f}°")
        feedback_with_angle = f"{feedback} ({primary_angle:.1f}°)"
        
        # Calculate confidence
        key_landmarks = [landmarks[idx] for idx in exercise.landmarks_required]
        confidence = sum(lm.visibility for lm in key_landmarks) / len(key_landmarks)
        
        return ExerciseResult(
            exercise_id=exercise.exercise_id,
            exercise_name=exercise.name,
            status=status,
            primary_angle=primary_angle,
            secondary_angle=angles.get("shoulder" if primary_angle_name == "elbow" else "elbow"),
            angles=angles,
            feedback=feedback_with_angle,
            confidence=confidence,
            frame_number=frame_number,
        )
    
    def analyze_sequence(
        self,
        landmark_sequence: List[List[Landmark]],
        exercise: ExerciseDefinition,
    ) -> List[ExerciseResult]:
        """Analyze multiple frames from a video."""
        return [
            self.analyze(landmarks, exercise, frame_number=frame_num)
            for frame_num, landmarks in enumerate(landmark_sequence)
        ]
