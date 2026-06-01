"""
Exercise definitions for rehabilitation system.
Each exercise specifies angles, thresholds, and clinical parameters.
"""

from .models import (
    ExerciseDefinition,
    AngleThreshold,
    ExerciseStatus,
)

# MediaPipe pose landmark indices (right side for consistency)
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_detector
LANDMARK_INDICES = {
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
    "right_hip": 24,
    "left_hip": 23,
}


# ============================================================================
# EXERCISE 1: ARM ABDUCTION (Shoulder Raise with Arm Straight)
# ============================================================================
EXERCISE_1_DEFINITION = ExerciseDefinition(
    exercise_id=1,
    name="Arm Abduction",
    description="Right shoulder abduction with elbow constraint (keep arm straight)",
    landmarks_required=[
        LANDMARK_INDICES["right_shoulder"],
        LANDMARK_INDICES["right_elbow"],
        LANDMARK_INDICES["right_wrist"],
        LANDMARK_INDICES["right_hip"],
    ],
    primary_angles=["elbow"],
    angle_thresholds={
        "elbow": AngleThreshold(
            name="elbow_flexion",
            min_value=160.0,  # Must keep arm straight (nearly 180°)
            feedback_pass="✅ Form: PASS (Arm Straight)",
            feedback_fail="❌ FAIL: Keep Arm Straight!",
        ),
    },
    feedback_rules={
        ExerciseStatus.PASS: "✅ Excellent form! Arm abduction achieved.",
        ExerciseStatus.FAIL: "❌ Elbow bent too much. Keep arm straight at ~160°.",
    },
)


# ============================================================================
# EXERCISE 2: ARM V-TO-W TRANSITION (Shoulder Flexion 120° to 90°)
# ============================================================================
EXERCISE_2_DEFINITION = ExerciseDefinition(
    exercise_id=2,
    name="Arm V-to-W Transition",
    description="Shoulder flexion transition from V-shape (120°) to W-shape (90°)",
    landmarks_required=[
        LANDMARK_INDICES["right_shoulder"],
        LANDMARK_INDICES["right_elbow"],
        LANDMARK_INDICES["right_wrist"],
        LANDMARK_INDICES["right_hip"],
    ],
    primary_angles=["shoulder"],
    angle_thresholds={
        "shoulder": AngleThreshold(
            name="shoulder_flexion",
            min_value=85.0,  # W-shape minimum
            max_value=125.0,  # V-shape maximum
            target_value=105.0,  # Middle transition zone
            feedback_pass="✅ Target Transition Achieved",
            feedback_fail="⏳ Keep transitioning between V and W",
            feedback_transitioning="⏳ Transitioning between shapes",
        ),
    },
    feedback_rules={
        ExerciseStatus.PASS: "✅ Perfect V-to-W transition!",
        ExerciseStatus.TRANSITIONING: "⏳ Keep moving between V (120°) and W (90°)",
        ExerciseStatus.FAIL: "❌ Out of target range. Maintain 85-125° shoulder angle.",
    },
)


# ============================================================================
# EXERCISE 3: INCLINED PUSH-UP (Elbow Flexion Depth)
# ============================================================================
EXERCISE_3_DEFINITION = ExerciseDefinition(
    exercise_id=3,
    name="Inclined Push-up",
    description="Inclined push-up with elbow flexion depth requirement",
    landmarks_required=[
        LANDMARK_INDICES["right_shoulder"],
        LANDMARK_INDICES["right_elbow"],
        LANDMARK_INDICES["right_wrist"],
        LANDMARK_INDICES["right_hip"],
    ],
    primary_angles=["elbow"],
    angle_thresholds={
        "elbow": AngleThreshold(
            name="elbow_flexion_depth",
            max_value=100.0,  # Must achieve sufficient depth (100° or less)
            feedback_pass="✅ Depth: PASS",
            feedback_fail="❌ FAIL: Go Deeper!",
        ),
    },
    feedback_rules={
        ExerciseStatus.PASS: "✅ Great depth! Maintaining proper form.",
        ExerciseStatus.FAIL: "❌ Not deep enough. Bend elbow more (aim for <100°)",
    },
)


# Registry of all exercises
EXERCISES = {
    1: EXERCISE_1_DEFINITION,
    2: EXERCISE_2_DEFINITION,
    3: EXERCISE_3_DEFINITION,
}


def get_exercise(exercise_id: int) -> ExerciseDefinition:
    """Get exercise definition by ID."""
    if exercise_id not in EXERCISES:
        raise ValueError(
            f"Unknown exercise ID: {exercise_id}. "
            f"Available: {list(EXERCISES.keys())}"
        )
    return EXERCISES[exercise_id]


def list_exercises():
    """List all available exercises."""
    return [
        (ex_id, ex.name, ex.description)
        for ex_id, ex in sorted(EXERCISES.items())
    ]
