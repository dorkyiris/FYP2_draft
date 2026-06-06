"""POST /analyze and POST /analyze-sequence endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from api.auth import get_api_key
from api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    SequenceRequest,
    SequenceResponse,
)
from rehabilitationcore.exercises import EXERCISES
from rehabilitationcore.analyzer import ExerciseAnalyzer
from rehabilitationcore.models import Landmark
from monitoring.metrics import get_metrics

router = APIRouter(tags=["analysis"])
_analyzer = ExerciseAnalyzer()


def _request_landmarks(raw: list) -> List[Landmark]:
    return [Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) for lm in raw]


def _exercise_or_404(exercise_id: int):
    if exercise_id not in EXERCISES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exercise {exercise_id} not found. Available: {sorted(EXERCISES)}",
        )
    return EXERCISES[exercise_id]


def _result_to_response(result, frame_number=None) -> AnalyzeResponse:
    return AnalyzeResponse(
        exercise_id=result.exercise_id,
        exercise_name=result.exercise_name,
        status=result.status.value,
        primary_angle=result.primary_angle,
        secondary_angle=result.secondary_angle,
        feedback=result.feedback,
        confidence=result.confidence,
        frame_number=result.frame_number,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_frame(req: AnalyzeRequest, _: str | None = Depends(get_api_key)):
    exercise = _exercise_or_404(req.exercise_id)
    landmarks = _request_landmarks(req.landmarks)
    result = _analyzer.analyze(landmarks, exercise, frame_number=req.frame_number)
    get_metrics().record_analysis(req.exercise_id, result.status.value, result.confidence)
    return _result_to_response(result)


@router.post("/analyze-sequence", response_model=SequenceResponse)
def analyze_sequence(req: SequenceRequest, _: str | None = Depends(get_api_key)):
    exercise = _exercise_or_404(req.exercise_id)
    landmark_sequences = [_request_landmarks(frame) for frame in req.frames]
    results = _analyzer.analyze_sequence(landmark_sequences, exercise)

    responses = [_result_to_response(r) for r in results]
    pass_count = sum(1 for r in results if r.status.value == "PASS")
    fail_count = sum(1 for r in results if r.status.value == "FAIL")

    return SequenceResponse(
        exercise_id=exercise.exercise_id,
        exercise_name=exercise.name,
        results=responses,
        total_frames=len(results),
        pass_count=pass_count,
        fail_count=fail_count,
    )
