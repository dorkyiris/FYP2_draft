"""
Video rendering: Draw clinical overlays on video frames.
"""

import cv2
import numpy as np
import logging
from typing import Tuple

from rehabilitationcore.models import Landmark, ExerciseResult, ExerciseStatus

logger = logging.getLogger(__name__)


class VideoRenderer:
    """Render clinical annotations on video frames."""
    
    COLORS = {
        ExerciseStatus.PASS: (0, 255, 0),
        ExerciseStatus.FAIL: (0, 0, 255),
        ExerciseStatus.TRANSITIONING: (0, 165, 255),
        ExerciseStatus.TRACKING: (255, 0, 0),
    }
    
    @staticmethod
    def draw_clinical_overlay(
        frame: np.ndarray,
        landmarks: List[Landmark],
        exercise_result: ExerciseResult,
        exercise_num: int,
    ) -> np.ndarray:
        """
        Draw comprehensive clinical overlay on frame.
        
        Args:
            frame: OpenCV frame
            landmarks: Pose landmarks
            exercise_result: Analysis result
            exercise_num: Exercise ID for specific logic
        
        Returns:
            Annotated frame
        """
        height, width = frame.shape[:2]
        
        # Landmark indices
        r_shoulder, r_elbow, r_wrist, r_hip = 12, 14, 16, 24
        
        # Get coordinates
        def get_coords(idx) -> Tuple[int, int]:
            if idx >= len(landmarks):
                return (0, 0)
            lm = landmarks[idx]
            return (int(lm.x * width), int(lm.y * height))
        
        shoulder = get_coords(r_shoulder)
        elbow = get_coords(r_elbow)
        wrist = get_coords(r_wrist)
        hip = get_coords(r_hip)
        
        # Check visibility
        vis_shoulder = (
            r_shoulder < len(landmarks)
            and landmarks[r_shoulder].visibility > 0.65
        )
        vis_elbow = r_elbow < len(landmarks) and landmarks[r_elbow].visibility > 0.65
        vis_wrist = r_wrist < len(landmarks) and landmarks[r_wrist].visibility > 0.65
        vis_hip = r_hip < len(landmarks) and landmarks[r_hip].visibility > 0.65
        
        # Get color based on status
        color = VideoRenderer.COLORS.get(exercise_result.status, (0, 255, 255))
        
        # Draw skeleton if visible
        if all([vis_shoulder, vis_elbow, vis_wrist]):
            cv2.line(frame, shoulder, elbow, color, 4)
            cv2.line(frame, elbow, wrist, color, 4)
        
        if all([vis_hip, vis_shoulder]) and exercise_num in [2, 3]:
            cv2.line(frame, hip, shoulder, color, 4)
        
        # Draw joint circles
        for pt in [shoulder, elbow, wrist]:
            cv2.circle(frame, pt, 5, color, -1)
        
        if exercise_num in [2, 3]:
            cv2.circle(frame, hip, 5, color, -1)
        
        # Draw status text and angle
        text = exercise_result.feedback
        y_pos = 50
        cv2.putText(
            frame,
            text,
            (40, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )
        
        # Draw confidence
        confidence_text = f"Confidence: {exercise_result.confidence:.1%}"
        cv2.putText(
            frame,
            confidence_text,
            (40, y_pos + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )
        
        return frame
    
    @staticmethod
    def create_output_video(
        input_path: str,
        output_path: str,
        frame_processor,
    ) -> None:
        """
        Process video and create output with annotations.
        
        Args:
            input_path: Input video file
            output_path: Output video file
            frame_processor: Callable that takes (frame) and returns annotated frame
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise IOError(f"Cannot create video writer: {output_path}")
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame = frame_processor(frame)
                out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.debug(
                        f"Wrote {frame_count}/{total_frames} frames "
                        f"({100.0 * frame_count / total_frames:.1f}%)"
                    )
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Output video created: {output_path} ({frame_count} frames)")
