"""
Video rendering: Draw clinical overlays on video frames.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

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
        color = VideoRenderer.COLORS.get(exercise_result.status, (0, 255, 255))

        def get_coords(idx: int) -> Tuple[int, int]:
            if idx >= len(landmarks):
                return (0, 0)
            lm = landmarks[idx]
            return (int(lm.x * width), int(lm.y * height))

        def vis(idx: int) -> bool:
            return idx < len(landmarks) and landmarks[idx].visibility > 0.65

        # Draw exercise-specific skeleton
        if exercise_num in (1, 2):
            # Exercises 1 & 2: shoulder-elbow-wrist chain + hip reference
            pts = {k: get_coords(i) for k, i in
                   [("hip", 24), ("shoulder", 12), ("elbow", 14), ("wrist", 16)]}
            if vis(12) and vis(14) and vis(16):
                cv2.line(frame, pts["shoulder"], pts["elbow"], color, 4)
                cv2.line(frame, pts["elbow"], pts["wrist"], color, 4)
            if vis(24) and vis(12):
                cv2.line(frame, pts["hip"], pts["shoulder"], color, 4)
                cv2.circle(frame, pts["hip"], 5, color, -1)
            for k in ("shoulder", "elbow", "wrist"):
                if vis({"shoulder": 12, "elbow": 14, "wrist": 16}[k]):
                    cv2.circle(frame, pts[k], 8, color, -1)

        elif exercise_num == 3:
            # Exercise 3 – Lifting the wrist: elbow → wrist → pinky chain
            pts = {k: get_coords(i) for k, i in
                   [("elbow", 14), ("wrist", 16), ("pinky", 18)]}
            if vis(14) and vis(16):
                cv2.line(frame, pts["elbow"], pts["wrist"], color, 4)
            if vis(16) and vis(18):
                cv2.line(frame, pts["wrist"], pts["pinky"], color, 4)
            for k, i in [("elbow", 14), ("wrist", 16), ("pinky", 18)]:
                if vis(i):
                    cv2.circle(frame, pts[k], 8, color, -1)

        elif exercise_num == 4:
            # Exercise 4 – Opening the hand: thumb/index/pinky fan from wrist
            pts = {k: get_coords(i) for k, i in
                   [("wrist", 16), ("pinky", 18), ("index", 20), ("thumb", 22)]}
            for finger_idx, idx in [("pinky", 18), ("index", 20), ("thumb", 22)]:
                if vis(16) and vis(idx):
                    cv2.line(frame, pts["wrist"], pts[finger_idx], color, 4)
                if vis(idx):
                    cv2.circle(frame, pts[finger_idx], 8, color, -1)
            if vis(16):
                cv2.circle(frame, pts["wrist"], 8, color, -1)

        # Status text
        cv2.putText(frame, exercise_result.feedback,
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, f"Confidence: {exercise_result.confidence:.1%}",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

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
