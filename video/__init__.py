"""
Video processing: Pose extraction from MediaPipe.
Decoupled from UI and biomechanical calculations.
"""

import cv2
import logging
import pathlib
import urllib.request
from typing import List, Optional
import numpy as np

from rehabilitationcore.models import Landmark

logger = logging.getLogger(__name__)

_MODEL_PATH = pathlib.Path(__file__).parent.parent / "assets" / "pose_landmarker_lite.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# mediapipe imported lazily inside __enter__ to avoid TensorFlow loading at import time


def _patch_tf_stub():
    """Inject a minimal tensorflow stub so mediapipe loads cleanly on broken TF installs."""
    import sys, types
    try:
        from tensorflow.tools.docs import doc_controls as _dc
        _dc.do_not_generate_docs
    except Exception:
        _noop = lambda fn: fn
        _dc_mod = types.ModuleType("tensorflow.tools.docs.doc_controls")
        _dc_mod.do_not_generate_docs = _noop
        _tf_docs = types.ModuleType("tensorflow.tools.docs")
        _tf_docs.doc_controls = _dc_mod
        _tf_tools = types.ModuleType("tensorflow.tools")
        _tf_tools.docs = _tf_docs
        _tf = types.ModuleType("tensorflow")
        _tf.tools = _tf_tools
        for _k, _v in [
            ("tensorflow",                         _tf),
            ("tensorflow.tools",                   _tf_tools),
            ("tensorflow.tools.docs",              _tf_docs),
            ("tensorflow.tools.docs.doc_controls", _dc_mod),
        ]:
            sys.modules.setdefault(_k, _v)


class PoseExtractionPipeline:
    """
    Extract pose landmarks from video frames using MediaPipe Tasks API.
    Uses PoseLandmarker (IMAGE mode) — compatible with mediapipe >= 0.10.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
        use_smoothing: bool = True,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        self.use_smoothing            = use_smoothing
        self._landmarker              = None
        self._mp                      = None

    def __enter__(self):
        """Context manager entry — loads MediaPipe Tasks API here to keep import-time clean."""
        _patch_tf_stub()

        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode,
        )

        if not _MODEL_PATH.exists():
            logger.info("Downloading pose_landmarker_lite.task …")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

        opts = PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(opts)
        self._mp         = mp
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def extract_frame(self, frame: np.ndarray) -> Optional[List[Landmark]]:
        """
        Extract pose landmarks from a single BGR frame.

        Returns:
            List of 33 Landmark objects, or None if no pose detected.
        """
        if self._landmarker is None:
            raise RuntimeError(
                "Pose extractor not initialised. "
                "Use as context manager: with PoseExtractionPipeline() as p: …"
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect(mp_image)
        except Exception as e:
            logger.error(f"MediaPipe detection failed: {e}")
            return None

        if not result.pose_landmarks:
            return None

        return [
            Landmark(
                x=lmk.x,
                y=lmk.y,
                z=lmk.z,
                visibility=getattr(lmk, "visibility", 1.0),
            )
            for lmk in result.pose_landmarks[0]
        ]

    def extract_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> List[List[Landmark]]:
        """
        Extract landmarks from all frames in a video file.

        Returns:
            List of landmark lists, one per frame (empty list if no pose on that frame).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Extracting landmarks from {total_frames} frames")

        all_landmarks = []
        frame_count   = 0

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
            return {
                "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps":          cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
        finally:
            cap.release()
