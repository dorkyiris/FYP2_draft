"""Custom exceptions for the rehabilitation system."""


class RehabSystemError(Exception):
    """Base exception for all rehabilitation system errors."""
    pass


class ConfigError(RehabSystemError):
    """Configuration file loading or validation failure."""
    pass


class ExerciseNotFoundError(ConfigError):
    """Requested exercise ID does not exist in configuration."""

    def __init__(self, exercise_id: int, available: list = None):
        self.exercise_id = exercise_id
        self.available = available or []
        available_str = f" Available IDs: {self.available}" if self.available else ""
        super().__init__(f"Exercise {exercise_id} not found.{available_str}")


class LandmarkError(RehabSystemError):
    """Landmark visibility or chain validation failure."""

    def __init__(self, message: str, landmark_idx: int = None, visibility: float = None):
        self.landmark_idx = landmark_idx
        self.visibility = visibility
        super().__init__(message)


class AnalysisError(RehabSystemError):
    """Unexpected failure in the exercise analysis pipeline."""
    pass
