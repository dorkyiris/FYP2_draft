"""
Exercise analyzer using biomechanical engine.
Orchestrates landmark extraction, angle calculations, and grading.
"""

from typing import List, Tuple, Dict, Optional
import logging

from .models import Landmark, ExerciseDefinition, ExerciseResult, ExerciseStatus
from .biomechanics import calculate_2d_angle, validate_landmark_chain

logger = logging.getLogger(__name__)


class BiomechanicalAnalyzer:
    """
    Core analyzer for exercise biomechanics.
    Uses pure calculations to grade exercise performance.
    """
    
    def __init__(self, min_visibility: float = 0.65):
        """
        Initialize analyzer.
        
        Args:
            min_visibility: Minimum landmark visibility threshold (0-1)
        """
        self.min_visibility = min_visibility
    
    def calculate_angles(
        self,
        landmarks: List[Landmark],
        landmark_indices: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Calculate relevant angles from landmarks.
        
        Args:
            landmarks: List of pose landmarks
            landmark_indices: Mapping of body part names to landmark indices
        
        Returns:
            Dictionary of angle_name -> angle_value
        """
        angles = {}
        
        # Safely get landmark by index
        def get_landmark(idx: int) -> Optional[Landmark]:
            if 0 <= idx < len(landmarks):
                return landmarks[idx]
            return None
        
        # Calculate shoulder angle (hip-shoulder-elbow)
        hip = get_landmark(landmark_indices.get("hip", 24))
        shoulder = get_landmark(landmark_indices.get("shoulder", 12))
        elbow = get_landmark(landmark_indices.get("elbow", 14))
        
        if all([hip, shoulder, elbow]) and all(
            lm.is_visible(self.min_visibility) for lm in [hip, shoulder, elbow]
        ):
            shoulder_angle = calculate_2d_angle(
                hip.to_tuple(),
                shoulder.to_tuple(),
                elbow.to_tuple(),
            )
            angles["shoulder"] = shoulder_angle
        
        # Calculate elbow angle (shoulder-elbow-wrist)
        wrist = get_landmark(landmark_indices.get("wrist", 16))
        if all([shoulder, elbow, wrist]) and all(
            lm.is_visible(self.min_visibility) for lm in [shoulder, elbow, wrist]
        ):
            elbow_angle = calculate_2d_angle(
                shoulder.to_tuple(),
                elbow.to_tuple(),
                wrist.to_tuple(),
            )
            angles["elbow"] = elbow_angle
        
        return angles
    
    def analyze_exercise(
        self,
        landmarks: List[Landmark],
        exercise: ExerciseDefinition,
        frame_number: Optional[int] = None,
    ) -> ExerciseResult:
        """
        Analyze exercise performance based on landmarks.
        
        Args:
            landmarks: List of pose landmarks from MediaPipe
            exercise: Exercise definition with thresholds
            frame_number: Optional frame number for tracking
        
        Returns:
            ExerciseResult with status and feedback
        """
        # Validate landmarks
        is_valid, error_msg = validate_landmark_chain(
            landmarks,
            exercise.landmarks_required,
            self.min_visibility,
        )
        
        if not is_valid:
            logger.debug(f"Exercise {exercise.exercise_id}: {error_msg}")
            return ExerciseResult(
                exercise_id=exercise.exercise_id,
                exercise_name=exercise.name,
                status=ExerciseStatus.TRACKING,
                primary_angle=0.0,
                feedback=f"🔍 Tracking... {error_msg}",
                confidence=0.0,
                frame_number=frame_number,
            )
        
        # Calculate all angles
        landmark_indices = {
            "hip": exercise.landmarks_required[3],
            "shoulder": exercise.landmarks_required[0],
            "elbow": exercise.landmarks_required[1],
            "wrist": exercise.landmarks_required[2],
        }
        
        angles = self.calculate_angles(landmarks, landmark_indices)
        
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
        
        # Get primary angle (first one in the list)
        primary_angle_name = exercise.primary_angles[0]
        if primary_angle_name not in angles:
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
        
        # Evaluate using thresholds
        threshold = exercise.angle_thresholds[primary_angle_name]
        status = threshold.evaluate(primary_angle)
        
        # Generate feedback
        if status in exercise.feedback_rules:
            feedback = exercise.feedback_rules[status]
        else:
            feedback = f"{status.value}: {primary_angle:.1f}°"
        
        # Add angle info to feedback
        feedback_with_angle = f"{feedback} ({primary_angle:.1f}°)"
        
        # Calculate confidence based on visibility of key landmarks
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


class ExerciseAnalyzer:
    """
    High-level exercise analyzer combining biomechanical calculations with exercise definitions.
    Stateless, can process multiple exercises and frames.
    """
    
    def __init__(self, min_visibility: float = 0.65):
        """Initialize analyzer."""
        self.biomech = BiomechanicalAnalyzer(min_visibility)
        self.min_visibility = min_visibility
    
    def analyze(
        self,
        landmarks: List[Landmark],
        exercise: ExerciseDefinition,
        frame_number: Optional[int] = None,
    ) -> ExerciseResult:
        """Analyze single frame of exercise."""
        return self.biomech.analyze_exercise(landmarks, exercise, frame_number)
    
    def analyze_sequence(
        self,
        landmark_sequence: List[List[Landmark]],
        exercise: ExerciseDefinition,
    ) -> List[ExerciseResult]:
        """
        Analyze sequence of frames (e.g., from a video).
        
        Args:
            landmark_sequence: List of landmark frames
            exercise: Exercise definition
        
        Returns:
            List of ExerciseResults, one per frame
        """
        results = []
        for frame_num, landmarks in enumerate(landmark_sequence):
            result = self.analyze(landmarks, exercise, frame_number=frame_num)
            results.append(result)
        return results
