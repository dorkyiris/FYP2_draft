"""FastAPI application factory."""

from fastapi import FastAPI
from api.routes import health, exercises, analyze


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tele-Rehabilitation API",
        description="Vision-based stroke rehabilitation exercise analysis",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(health.router)
    app.include_router(exercises.router)
    app.include_router(analyze.router)

    return app


app = create_app()
