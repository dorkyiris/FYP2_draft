# Phase 2 Kickoff: Configuration Management

**Status:** Ready to Start  
**Estimated Effort:** 8-12 hours  
**Priority:** High (blocks API development in Phase 3)

---

## 📋 What Phase 2 Will Do

Move exercise thresholds and system parameters from hard-coded values to configuration files. This enables:

- ✅ Changing exercise parameters without code changes
- ✅ Version-controlling parameter changes
- ✅ A/B testing different thresholds
- ✅ Fast iteration on clinical validation
- ✅ Configuration-as-code for reproducibility

---

## 🎯 Phase 2 Scope

### Task 1: Create Exercise Configuration Files

**File:** `config/exercises.yaml`

```yaml
exercises:
  1:
    name: "Arm Abduction"
    description: "Right shoulder abduction with elbow constraint"
    landmarks: [12, 14, 16, 24]
    thresholds:
      elbow:
        type: "minimum"
        value: 160.0
        feedback_pass: "✅ Form: PASS (Arm Straight)"
        feedback_fail: "❌ FAIL: Keep Arm Straight!"
  
  2:
    name: "Arm V-to-W Transition"
    landmarks: [12, 14, 16, 24]
    thresholds:
      shoulder:
        type: "range"
        min: 85.0
        max: 125.0
        target: 105.0
        feedback_pass: "✅ Target Transition Achieved"
  
  3:
    name: "Inclined Push-up"
    landmarks: [12, 14, 16, 24]
    thresholds:
      elbow:
        type: "maximum"
        value: 100.0
        feedback_pass: "✅ Depth: PASS"
        feedback_fail: "❌ FAIL: Go Deeper!"
```

### Task 2: Create System Configuration File

**File:** `config/system.yaml`

```yaml
mediapipe:
  min_detection_confidence: 0.65
  min_tracking_confidence: 0.65
  model_complexity: 1

smoothing:
  method: "ema"  # or "sma"
  span: 3

visualization:
  colors:
    pass: [0, 255, 0]           # Green (BGR)
    fail: [0, 0, 255]           # Red
    transitioning: [0, 165, 255]  # Orange
    tracking: [255, 0, 0]       # Blue
  
  font_size: 1.2
  line_thickness: 4
  joint_radius: 5

logging:
  level: "INFO"
  file: "logs/app.log"
  max_bytes: 10485760  # 10MB
  backup_count: 5
```

### Task 3: Build Configuration Loader

**File:** `config/loader.py`

```python
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """Load and manage system configuration."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.exercises = self._load_exercises()
        self.system = self._load_system()
    
    def _load_exercises(self) -> Dict[int, Dict[str, Any]]:
        """Load exercise configurations."""
        path = self.config_dir / "exercises.yaml"
        with open(path) as f:
            config = yaml.safe_load(f)
        return config.get("exercises", {})
    
    def _load_system(self) -> Dict[str, Any]:
        """Load system configurations."""
        path = self.config_dir / "system.yaml"
        with open(path) as f:
            config = yaml.safe_load(f)
        return config
    
    def get_exercise(self, exercise_id: int) -> Dict[str, Any]:
        """Get exercise configuration."""
        if exercise_id not in self.exercises:
            raise ValueError(f"Unknown exercise: {exercise_id}")
        return self.exercises[exercise_id]
    
    def get_threshold(self, exercise_id: int, angle_name: str):
        """Get threshold for specific angle."""
        exercise = self.get_exercise(exercise_id)
        thresholds = exercise.get("thresholds", {})
        if angle_name not in thresholds:
            raise ValueError(f"No threshold for {angle_name} in exercise {exercise_id}")
        return thresholds[angle_name]
```

### Task 4: Integrate Configuration Into Core

**Modify:** `rehabilitationcore/exercises.py`

```python
from config.loader import ConfigManager

# Load configuration
config = ConfigManager()

# Build exercise definitions from config
EXERCISES = {}
for ex_id, ex_config in config.exercises.items():
    thresholds = {}
    for angle_name, threshold_config in ex_config.get("thresholds", {}).items():
        thresholds[angle_name] = AngleThreshold(
            name=angle_name,
            min_value=threshold_config.get("min"),
            max_value=threshold_config.get("max"),
            target_value=threshold_config.get("target"),
            feedback_pass=threshold_config.get("feedback_pass"),
            feedback_fail=threshold_config.get("feedback_fail"),
        )
    
    EXERCISES[ex_id] = ExerciseDefinition(
        exercise_id=ex_id,
        name=ex_config.get("name"),
        description=ex_config.get("description"),
        landmarks_required=ex_config.get("landmarks", []),
        primary_angles=list(thresholds.keys()),
        angle_thresholds=thresholds,
        feedback_rules={...},
    )
```

### Task 5: Add Configuration Tests

**File:** `tests/unit/test_config.py`

```python
import pytest
from config.loader import ConfigManager

def test_load_exercises():
    """Configuration loads all 3 exercises."""
    config = ConfigManager()
    assert len(config.exercises) >= 3
    assert 1 in config.exercises
    assert 2 in config.exercises
    assert 3 in config.exercises

def test_exercise_has_required_fields():
    """Each exercise has required fields."""
    config = ConfigManager()
    for ex_id, ex in config.exercises.items():
        assert "name" in ex
        assert "landmarks" in ex
        assert "thresholds" in ex

def test_threshold_validation():
    """Get threshold for exercise and angle."""
    config = ConfigManager()
    threshold = config.get_threshold(1, "elbow")
    assert threshold is not None
```

### Task 6: Update Documentation

**Modify:** `PHASE_1_QUICKSTART.md`

Add section: "Configuring Exercises Without Code Changes"

```markdown
## Configuration Management (Phase 2)

### Change Exercise Threshold
Edit `config/exercises.yaml`:
```yaml
exercises:
  1:
    thresholds:
      elbow:
        value: 155.0  # Changed from 160.0
```
Run tests to verify: `pytest tests/unit/ -v`

### Add New Optimization
Edit `config/system.yaml`:
```yaml
smoothing:
  span: 5  # Increased smoothing
```
Restart app - no code changes needed!
```

---

## 📝 Implementation Checklist

- [ ] Create `config/` directory
- [ ] Write `config/exercises.yaml` with all 3 exercises
- [ ] Write `config/system.yaml` with all parameters
- [ ] Create `config/loader.py` with `ConfigManager` class
- [ ] Create `config/__init__.py`
- [ ] Modify `rehabilitationcore/exercises.py` to use config
- [ ] Update `rehabilitationcore/analyzer.py` to use system config
- [ ] Add tests in `tests/unit/test_config.py`
- [ ] Run full test suite: `pytest tests/unit/ -v`
- [ ] Update documentation
- [ ] Verify no regressions (all 30 tests still pass)

---

## ✅ Phase 2 Success Criteria

- [x] Configuration files created and valid YAML
- [x] ConfigManager class loads without errors
- [x] All 3 exercises load from config
- [x] Thresholds applied correctly in analyzer
- [x] New tests pass (target: 40+ total tests)
- [x] No regressions (30 Phase 1 tests still pass)
- [x] Can change threshold without code changes
- [x] Documentation updated

---

## 🚀 After Phase 2

Ready for:

- **Phase 3:** Error Handling & Logging
- **Phase 4:** Documentation
- **Phase 5:** CI/CD pipelines
- **API Development:** REST endpoints using new modules
- **Clinical Validation:** Easy threshold adjustment

---

## 📚 Reference Files

**Start with these Phase 1 files:**
- `rehabilitationcore/exercises.py` - Current hardcoded values
- `rehabilitationcore/analyzer.py` - Where thresholds are used
- `tests/unit/test_analyzer.py` - Test patterns to follow

**Phase 2 deliverables:**
- `config/exercises.yaml` - Exercise definitions
- `config/system.yaml` - System parameters
- `config/loader.py` - Configuration loading
- `tests/unit/test_config.py` - Configuration tests

---

## 💡 Tips

1. Start with one exercise in YAML, verify it loads and works
2. Gradually convert remaining exercises
3. Run tests after each major change
4. Keep original `rehabilitationcore/exercises.py` values in YAML
5. Document why each parameter exists

---

**Phase 2 Status:** Ready to implement  
**Estimated Time:** 8-12 hours  
**Target Completion:** Next session
