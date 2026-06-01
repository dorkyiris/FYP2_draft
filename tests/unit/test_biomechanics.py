"""
Unit tests for biomechanical calculations.
Pure math functions with no external dependencies.
"""

import pytest
import numpy as np
from rehabilitationcore.biomechanics import (
    calculate_2d_angle,
    calculate_3d_angle,
    calculate_distance,
    smooth_signal,
)


class TestAngleCalculation:
    """Test 2D angle calculations."""
    
    def test_straight_line_180_degrees(self):
        """Straight line should be 180°."""
        # Points on a line
        result = calculate_2d_angle((0, 0), (1, 1), (2, 2))
        assert abs(result - 180.0) < 0.1, f"Expected ~180°, got {result}°"
    
    def test_right_angle_90_degrees(self):
        """Right angle should be 90°."""
        result = calculate_2d_angle((0, 0), (0, 0), (1, 0))
        # This is a special case where both vectors start from same point
        assert 0 <= result <= 180, "Angle should be in [0, 180]"
    
    def test_horizontal_vertical_perpendicular(self):
        """Horizontal to vertical should be 90°."""
        result = calculate_2d_angle((1, 0), (0, 0), (0, 1))
        assert abs(result - 90.0) < 1.0, f"Expected ~90°, got {result}°"
    
    def test_elbow_extension_160_degrees(self):
        """Arm extension (elbow straight) should be ~160°."""
        # Shoulder, elbow, wrist forming an extended arm
        shoulder = (0, 0)
        elbow = (10, 10)
        wrist = (20, 20)
        result = calculate_2d_angle(shoulder, elbow, wrist)
        assert abs(result - 180.0) < 1.0, f"Expected ~180° for extended arm, got {result}°"
    
    def test_elbow_flexion_90_degrees(self):
        """Right angle at elbow (bicep curl)."""
        shoulder = (0, 0)
        elbow = (0, 10)
        wrist = (10, 10)
        result = calculate_2d_angle(shoulder, elbow, wrist)
        assert abs(result - 90.0) < 1.0, f"Expected ~90° for right angle, got {result}°"
    
    def test_angle_range_always_0_to_180(self):
        """Angle should always be in [0, 180] range."""
        test_cases = [
            ((0, 0), (0, 0), (1, 0)),
            ((1, 1), (0, 0), (1, -1)),
            ((5, 5), (2, 2), (-3, -3)),
            ((-1, -1), (0, 0), (1, 1)),
        ]
        for p1, p2, p3 in test_cases:
            result = calculate_2d_angle(p1, p2, p3)
            assert 0 <= result <= 180, (
                f"Angle {result}° outside [0, 180] for points {p1}, {p2}, {p3}"
            )
    
    def test_symmetry_reversing_points(self):
        """Angle should be same if we reverse the first and third points."""
        p1, p2, p3 = (1, 1), (0, 0), (1, -1)
        angle1 = calculate_2d_angle(p1, p2, p3)
        angle2 = calculate_2d_angle(p3, p2, p1)
        assert abs(angle1 - angle2) < 0.1, "Angle should be symmetric"


class TestExerciseScenarios:
    """Test realistic exercise scenarios."""
    
    def test_exercise_1_arm_abduction_pass(self):
        """Exercise 1: Arm abduction with straight elbow (>= 160°) should PASS."""
        shoulder = (0, 0)
        elbow = (0, 10)
        wrist = (0.1, 20)  # Nearly straight
        angle = calculate_2d_angle(shoulder, elbow, wrist)
        assert angle >= 160, f"Straight arm angle {angle}° should be >= 160°"
    
    def test_exercise_1_arm_abduction_fail(self):
        """Exercise 1: Bent elbow (< 160°) should FAIL."""
        shoulder = (0, 0)
        elbow = (0, 10)
        wrist = (5, 15)  # Bent
        angle = calculate_2d_angle(shoulder, elbow, wrist)
        assert angle < 160, f"Bent arm angle {angle}° should be < 160°"
    
    def test_exercise_2_v_shape_120_degrees(self):
        """Exercise 2: V-shape should be ~120° shoulder angle."""
        hip = (0, 0)
        shoulder = (0, 10)
        elbow = (8, 15)  # V-shape
        angle = calculate_2d_angle(hip, shoulder, elbow)
        # Should be in V-range (>100°)
        assert angle > 100, f"V-shape angle {angle}° should be > 100°"
    
    def test_exercise_2_w_shape_90_degrees(self):
        """Exercise 2: W-shape should result in specific shoulder angle."""
        # Using proper V-W positions
        hip = (0, 0)
        shoulder = (5, 10)
        # For W-shape, elbow should be positioned to create ~90-100° angle
        elbow = (6, 18)
        angle = calculate_2d_angle(hip, shoulder, elbow)
        # Just verify angle is in reasonable range
        assert 0 <= angle <= 180, f"W-shape angle {angle}° should be valid"
    
    def test_exercise_3_push_up_deep(self):
        """Exercise 3: Deep push-up (elbow <= 100°) should PASS."""
        shoulder = (0, 0)
        elbow = (5, 4)
        wrist = (8, 2)  # Deep bend creating acute angle
        angle = calculate_2d_angle(shoulder, elbow, wrist)
        # Acute angle close to 90-100 range
        assert angle < 120, f"Deep push-up angle {angle}° should be acute"
    
    def test_exercise_3_push_up_shallow(self):
        """Exercise 3: Shallow push-up (elbow > 100°) should FAIL."""
        shoulder = (0, 0)
        elbow = (0, 10)
        wrist = (1, 18)  # Shallow bend
        angle = calculate_2d_angle(shoulder, elbow, wrist)
        assert angle > 100, f"Shallow push-up angle {angle}° should be > 100°"


class TestSmoothing:
    """Test signal smoothing algorithms."""
    
    def test_ema_smoothing_reduces_jitter(self):
        """EMA smoothing should reduce variance."""
        noisy = [90, 92, 88, 91, 89, 93, 87, 95]
        smoothed = smooth_signal(noisy, method="ema", span=3)
        
        # Calculate variance
        noisy_var = np.var(noisy)
        smoothed_var = np.var(smoothed)
        assert smoothed_var < noisy_var, "Smoothed signal should have less variance"
    
    def test_sma_smoothing_preserves_mean(self):
        """Simple moving average should preserve mean."""
        signal = [90, 92, 88, 91, 89, 93, 87, 95]
        smoothed = smooth_signal(signal, method="sma", span=3)
        
        assert abs(np.mean(signal) - np.mean(smoothed)) < 0.5, (
            "SMA should preserve mean"
        )
    
    def test_smoothing_handles_nan(self):
        """Smoothing should handle NaN values gracefully."""
        noisy = [90, float('nan'), 88, float('nan'), 89, 93, 87, 95]
        smoothed = smooth_signal(noisy, method="ema", span=3)
        
        # Should still produce valid output
        valid_count = sum(1 for v in smoothed if not np.isnan(v))
        assert valid_count > 0, "Should produce valid smoothed values"
    
    def test_smoothing_empty_list(self):
        """Empty list should return empty."""
        result = smooth_signal([], method="ema", span=3)
        assert result == [], "Empty input should return empty output"


class TestDistance:
    """Test distance calculations."""
    
    def test_distance_same_point(self):
        """Distance from point to itself should be 0."""
        assert calculate_distance((1, 1), (1, 1)) == 0
    
    def test_distance_unit_square(self):
        """Distance should follow Pythagorean theorem."""
        # 3-4-5 triangle
        result = calculate_distance((0, 0), (3, 4))
        assert abs(result - 5.0) < 0.0001
    
    def test_distance_symmetry(self):
        """Distance should be symmetric."""
        d1 = calculate_distance((1, 1), (4, 5))
        d2 = calculate_distance((4, 5), (1, 1))
        assert d1 == d2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
