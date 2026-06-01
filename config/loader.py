"""Configuration loader for YAML-based exercise and system settings."""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Load and manage system configuration from YAML files."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize ConfigManager and load configuration files.
        
        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = Path(config_dir)
        self.exercises = self._load_exercises()
        self.system = self._load_system()
    
    def _load_exercises(self) -> Dict[int, Dict[str, Any]]:
        """Load exercise configurations from exercises.yaml."""
        path = self.config_dir / "exercises.yaml"
        if not path.exists():
            return {}
        
        with open(path) as f:
            config = yaml.safe_load(f) or {}
        return config.get("exercises", {})
    
    def _load_system(self) -> Dict[str, Any]:
        """Load system configurations from system.yaml."""
        path = self.config_dir / "system.yaml"
        if not path.exists():
            return {}
        
        with open(path) as f:
            config = yaml.safe_load(f) or {}
        return config
    
    def get_exercise(self, exercise_id: int) -> Dict[str, Any]:
        """Get exercise configuration by ID.
        
        Args:
            exercise_id: ID of the exercise
            
        Returns:
            Dictionary with exercise configuration
            
        Raises:
            ValueError: If exercise not found
        """
        if exercise_id not in self.exercises:
            raise ValueError(f"Exercise {exercise_id} not found in configuration")
        return self.exercises[exercise_id]
    
    def get_threshold(self, exercise_id: int, angle_name: str) -> Dict[str, Any]:
        """Get threshold configuration for a specific angle in an exercise.
        
        Args:
            exercise_id: ID of the exercise
            angle_name: Name of the angle (e.g., "elbow", "shoulder")
            
        Returns:
            Dictionary with threshold configuration
            
        Raises:
            ValueError: If exercise or threshold not found
        """
        exercise = self.get_exercise(exercise_id)
        thresholds = exercise.get("thresholds", {})
        if angle_name not in thresholds:
            raise ValueError(f"No threshold for {angle_name} in exercise {exercise_id}")
        return thresholds[angle_name]
