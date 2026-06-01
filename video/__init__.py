"""
Video processing: Pose extraction from MediaPipe.
Decoupled from UI and biomechanical calculations.
"""

import cv2
import mediapipe as mp
import logging
from typing import List, Tuple, Optional
import numpy as np

from rehabilitationcore.models import Landmark

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose


class PoseExtractionPipeline:
    """
    Extract pose landmarks from video frames using MediaPipe.
    Handles resource management (context managers).
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
        use_smoothing: bool = True,
    ):
        """
        Initialize pose extraction pipeline.
        
        Args:
            min_detection_confidence: MediaPipe detection confidence
            min_tracking_confidence: MediaPipe tracking confidence
            use_smoothing: Whether to apply MediaPipe's built-in smoothing
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_smoothing = use_smoothing
        self.pose = None
    
    def __enter__(self):
        """Context manager entry."""
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=self.use_smoothing,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.pose:
            self.pose.close()
            self.pose = None
    
    def extract_frame(self, frame: np.ndarray) -> Optional[List[Landmark]]:
        """
        Extract pose landmarks from a single frame.
        
        Args:
            frame: OpenCV frame (BGR format)
        
        Returns:
            List of Landmark objects, or None if extraction failed
        """
        if self.pose is None:
            raise RuntimeError(
                "Pose extractor not initialized. Use with context manager: "
                "with PoseExtractionPipeline() as pipeline: ..."
            )
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        try:
            results = self.pose.process(image_rgb)
        except Exception as e:
            logger.error(f"MediaPipe processing failed: {e}")
            return None
        finally:
            image_rgb.flags.writeable = True
        
        if results.pose_landmarks is None:
            return None
        
        # Convert MediaPipe landmarks to our Landmark objects
        landmarks = []
        for mp_landmark in results.pose_landmarks.landmark:
            landmark = Landmark(
                x=mp_landmark.x,
                y=mp_landmark.y,
                z=mp_landmark.z,
                visibility=mp_landmark.visibility,
            )
            landmarks.append(landmark)
        
        return landmarks
    
    def extract_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> List[List[Landmark]]:
        """
        Extract landmarks from all frames in a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None for all)
        
        Returns:
            List of landmark lists, one per frame
        
        Raises:
            FileNotFoundError: If video file not found
            IOError: If video cannot be opened
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Extracting landmarks from {total_frames} frames")
        
        all_landmarks = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = self.extract_frame(frame)
                all_landmarks.append(landmarks if landmarks else [])
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
                
                if frame_count % 30 == 0:
                    logger.debug(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Extraction complete: {frame_count} frames processed")
        return all_landmarks
    
    def get_video_properties(self, video_path: str) -> dict:
        """Get video properties (FPS, resolution, total frames)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        try:
            props = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            return props
        finally:
            cap.release()
