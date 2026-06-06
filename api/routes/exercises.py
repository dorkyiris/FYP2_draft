"""GET /exercises and GET /exercises/{id} endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from api.auth import get_api_key
from api.models import ExerciseResponse
from rehabilitationcore.exercises import EXERCISES
from rehabilitationcore.errors import ExerciseNotFoundError

router = APIRouter(prefix="/exercises", tags=["exercises"])


def _serialize(exercise_id: int) -> ExerciseResponse:
    ex = EXERCISES[exercise_id]
    return ExerciseResponse(
        exercise_id=exercise_id,
        name=ex.name,
        description=ex.description,
        landmarks_required=list(ex.landmarks_required),
        primary_angles=list(ex.angle_thresholds.keys()),
    )


@router.get("", response_model=List[ExerciseResponse])
def list_exercises(_: str | None = Depends(get_api_key)):
    return [_serialize(eid) for eid in sorted(EXERCISES)]


@router.get("/{exercise_id}", response_model=ExerciseResponse)
def get_exercise(exercise_id: int, _: str | None = Depends(get_api_key)):
    if exercise_id not in EXERCISES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exercise {exercise_id} not found. Available: {sorted(EXERCISES)}",
        )
    return _serialize(exercise_id)
