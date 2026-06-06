"""FastAPI application factory."""

import time
from fastapi import FastAPI, Request, Response
from api.routes import health, exercises, analyze
from monitoring.metrics import get_metrics, metrics_text


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tele-Rehabilitation API",
        description="Vision-based stroke rehabilitation exercise analysis",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.middleware("http")
    async def _record_metrics(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start
        get_metrics().record_request(
            endpoint=request.url.path,
            status_code=response.status_code,
            latency_s=latency,
        )
        return response

    @app.get("/metrics", tags=["system"], include_in_schema=False)
    def prometheus_metrics():
        return Response(content=metrics_text(), media_type="text/plain; version=0.0.4")

    app.include_router(health.router)
    app.include_router(exercises.router)
    app.include_router(analyze.router)

    return app


app = create_app()
