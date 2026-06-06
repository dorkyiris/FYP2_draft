# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System packages needed to compile OpenCV and MediaPipe wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime-only system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY rehabilitationcore/ rehabilitationcore/
COPY config/           config/
COPY api/              api/
COPY monitoring/       monitoring/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Expose API port
EXPOSE 8000

# Health check (polls /health every 30 s, 3 retries before unhealthy)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python - <<'EOF'
import urllib.request, sys
try:
    urllib.request.urlopen("http://localhost:8000/health", timeout=4)
except Exception:
    sys.exit(1)
EOF

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
