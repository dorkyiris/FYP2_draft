"""GET /health endpoint."""

from fastapi import APIRouter, Depends
from api.auth import get_api_key
from api.models import HealthResponse
from rehabilitationcore.exercises import EXERCISES

router = APIRouter()

_VERSION = "2.0.0"


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check(_: str | None = Depends(get_api_key)):
    return HealthResponse(
        status="ok",
        version=_VERSION,
        exercises_loaded=len(EXERCISES),
    )
