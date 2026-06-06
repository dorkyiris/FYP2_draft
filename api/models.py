"""Pydantic request/response schemas for the REST API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class LandmarkInput(BaseModel):
    x: float = Field(..., ge=0.0, le=2.0)   # >1.0 allowed for cropped frames
    y: float = Field(..., ge=0.0, le=2.0)
    z: float = 0.0
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    exercise_id: int = Field(..., ge=1, description="Clinical exercise ID (1–4)")
    landmarks: List[LandmarkInput] = Field(
        ..., min_length=1, description="33 MediaPipe pose landmarks"
    )
    frame_number: Optional[int] = Field(default=None, ge=0)


class AnalyzeResponse(BaseModel):
    exercise_id: int
    exercise_name: str
    status: str                          # PASS | FAIL | TRANSITIONING | TRACKING
    primary_angle: float
    secondary_angle: Optional[float] = None
    feedback: str
    confidence: float
    frame_number: Optional[int] = None


class SequenceRequest(BaseModel):
    exercise_id: int = Field(..., ge=1)
    frames: List[List[LandmarkInput]] = Field(..., min_length=1)


class SequenceResponse(BaseModel):
    exercise_id: int
    exercise_name: str
    results: List[AnalyzeResponse]
    total_frames: int
    pass_count: int
    fail_count: int


class ExerciseResponse(BaseModel):
    exercise_id: int
    name: str
    description: str
    landmarks_required: List[int]
    primary_angles: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    exercises_loaded: int
