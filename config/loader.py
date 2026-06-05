"""Configuration loader for YAML-based exercise and system settings."""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any

from rehabilitationcore.errors import ConfigError, ExerciseNotFoundError

logger = logging.getLogger("rehabilitation.config")


class ConfigManager:
    """Load and manage system configuration from YAML files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.exercises = self._load_exercises()
        self.system = self._load_system()

    def _load_exercises(self) -> Dict[int, Dict[str, Any]]:
        path = self.config_dir / "exercises.yaml"
        if not path.exists():
            raise ConfigError(f"exercises.yaml not found at {path}")

        try:
            with open(path) as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse exercises.yaml: {e}") from e

        exercises = config.get("exercises", {})
        logger.debug("Loaded %d exercises from %s", len(exercises), path)
        return exercises

    def _load_system(self) -> Dict[str, Any]:
        path = self.config_dir / "system.yaml"
        if not path.exists():
            logger.warning("system.yaml not found at %s — using defaults", path)
            return {}

        try:
            with open(path) as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse system.yaml: {e}") from e

        logger.debug("Loaded system config from %s", path)
        return config

    def get_exercise(self, exercise_id: int) -> Dict[str, Any]:
        """Return exercise config dict, raise ExerciseNotFoundError if missing."""
        if exercise_id not in self.exercises:
            raise ExerciseNotFoundError(
                exercise_id, available=sorted(self.exercises.keys())
            )
        return self.exercises[exercise_id]

    def get_threshold(self, exercise_id: int, angle_name: str) -> Dict[str, Any]:
        """Return threshold config for a specific angle, raise ValueError if missing."""
        exercise = self.get_exercise(exercise_id)
        thresholds = exercise.get("thresholds", {})
        if angle_name not in thresholds:
            raise ValueError(
                f"No threshold for '{angle_name}' in exercise {exercise_id}. "
                f"Available: {list(thresholds.keys())}"
            )
        return thresholds[angle_name]
