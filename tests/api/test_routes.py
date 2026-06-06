"""API route tests using FastAPI TestClient."""

import os
import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def keyed_client():
    """Client with REHAB_API_KEY env var set."""
    os.environ["REHAB_API_KEY"] = "test-secret"
    c = TestClient(create_app())
    yield c
    del os.environ["REHAB_API_KEY"]


# ---------------------------------------------------------------------------
# Helper — 33 high-visibility landmarks at centre
# ---------------------------------------------------------------------------

def _landmarks(n: int = 33, vis: float = 0.9):
    return [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": vis} for _ in range(n)]


def _straight_arm_landmarks():
    lms = _landmarks()
    lms[24] = {"x": 0.5, "y": 0.0, "z": 0.0, "visibility": 0.9}   # Hip
    lms[12] = {"x": 0.5, "y": 0.3, "z": 0.0, "visibility": 0.9}   # Shoulder
    lms[14] = {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}   # Elbow
    lms[16] = {"x": 0.5, "y": 0.7, "z": 0.0, "visibility": 0.9}   # Wrist
    return lms


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_exercises_loaded(self, client):
        data = client.get("/health").json()
        assert data["exercises_loaded"] == 4

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


# ---------------------------------------------------------------------------
# Exercises list
# ---------------------------------------------------------------------------

class TestExercisesList:

    def test_list_returns_200(self, client):
        r = client.get("/exercises")
        assert r.status_code == 200

    def test_list_returns_four_exercises(self, client):
        data = client.get("/exercises").json()
        assert len(data) == 4

    def test_list_sorted_by_id(self, client):
        data = client.get("/exercises").json()
        ids = [ex["exercise_id"] for ex in data]
        assert ids == sorted(ids)

    def test_exercise_shape(self, client):
        data = client.get("/exercises").json()
        ex = data[0]
        assert "exercise_id" in ex
        assert "name" in ex
        assert "landmarks_required" in ex
        assert "primary_angles" in ex


# ---------------------------------------------------------------------------
# Single exercise
# ---------------------------------------------------------------------------

class TestExerciseDetail:

    def test_get_exercise_1(self, client):
        r = client.get("/exercises/1")
        assert r.status_code == 200
        assert r.json()["exercise_id"] == 1

    def test_get_exercise_name(self, client):
        r = client.get("/exercises/1")
        assert r.json()["name"] == "Lifting an object"

    def test_get_unknown_exercise_returns_404(self, client):
        r = client.get("/exercises/999")
        assert r.status_code == 404

    def test_404_detail_mentions_id(self, client):
        r = client.get("/exercises/999")
        assert "999" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Analyze single frame
# ---------------------------------------------------------------------------

class TestAnalyzeFrame:

    def test_analyze_returns_200(self, client):
        payload = {"exercise_id": 1, "landmarks": _straight_arm_landmarks()}
        r = client.post("/analyze", json=payload)
        assert r.status_code == 200

    def test_analyze_response_shape(self, client):
        payload = {"exercise_id": 2, "landmarks": _straight_arm_landmarks()}
        data = client.post("/analyze", json=payload).json()
        assert "status" in data
        assert "primary_angle" in data
        assert "feedback" in data
        assert "confidence" in data

    def test_analyze_pass_on_straight_arm_ex2(self, client):
        payload = {"exercise_id": 2, "landmarks": _straight_arm_landmarks()}
        data = client.post("/analyze", json=payload).json()
        assert data["status"] == "PASS"

    def test_analyze_unknown_exercise_404(self, client):
        payload = {"exercise_id": 99, "landmarks": _landmarks()}
        r = client.post("/analyze", json=payload)
        assert r.status_code == 404

    def test_analyze_propagates_frame_number(self, client):
        payload = {"exercise_id": 1, "landmarks": _straight_arm_landmarks(), "frame_number": 7}
        data = client.post("/analyze", json=payload).json()
        assert data["frame_number"] == 7

    def test_analyze_low_visibility_returns_tracking(self, client):
        payload = {"exercise_id": 1, "landmarks": _landmarks(vis=0.1)}
        data = client.post("/analyze", json=payload).json()
        assert data["status"] == "TRACKING"


# ---------------------------------------------------------------------------
# Analyze sequence
# ---------------------------------------------------------------------------

class TestAnalyzeSequence:

    def test_sequence_returns_200(self, client):
        payload = {
            "exercise_id": 1,
            "frames": [_straight_arm_landmarks(), _straight_arm_landmarks()],
        }
        r = client.post("/analyze-sequence", json=payload)
        assert r.status_code == 200

    def test_sequence_total_frames(self, client):
        frames = [_straight_arm_landmarks()] * 3
        payload = {"exercise_id": 2, "frames": frames}
        data = client.post("/analyze-sequence", json=payload).json()
        assert data["total_frames"] == 3

    def test_sequence_pass_count(self, client):
        frames = [_straight_arm_landmarks()] * 3
        payload = {"exercise_id": 2, "frames": frames}
        data = client.post("/analyze-sequence", json=payload).json()
        assert data["pass_count"] == 3
        assert data["fail_count"] == 0

    def test_sequence_results_length(self, client):
        frames = [_straight_arm_landmarks()] * 2
        payload = {"exercise_id": 1, "frames": frames}
        data = client.post("/analyze-sequence", json=payload).json()
        assert len(data["results"]) == 2

    def test_sequence_unknown_exercise_404(self, client):
        payload = {"exercise_id": 99, "frames": [_landmarks()]}
        r = client.post("/analyze-sequence", json=payload)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Auth (keyed client)
# ---------------------------------------------------------------------------

class TestAuth:

    def test_valid_key_accepted(self, keyed_client):
        r = keyed_client.get("/health", headers={"X-API-Key": "test-secret"})
        assert r.status_code == 200

    def test_wrong_key_rejected(self, keyed_client):
        r = keyed_client.get("/health", headers={"X-API-Key": "wrong"})
        assert r.status_code == 403

    def test_missing_key_rejected(self, keyed_client):
        r = keyed_client.get("/health")
        assert r.status_code == 403
