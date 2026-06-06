# Deployment Guide

Tele-Rehabilitation API — containerised deployment reference.

---

## Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2 (or `docker compose` plugin)
- Python 3.10+ (for running load tests locally)

---

## Quick start with Docker Compose

```bash
# 1. Build and start
docker compose up --build -d

# 2. Verify the API is healthy
curl http://localhost:8000/health

# 3. List exercises
curl http://localhost:8000/exercises

# 4. Tail logs
docker compose logs -f api
```

Stop and clean up:

```bash
docker compose down
docker compose down -v   # also removes the log volume
```

---

## Manual Docker commands

```bash
# Build
docker build -t rehab-api:latest .

# Run (no auth)
docker run -p 8000:8000 rehab-api:latest

# Run with API-key auth enabled
docker run -p 8000:8000 \
  -e REHAB_API_KEY=my-secret-key \
  rehab-api:latest

# Shell into a running container
docker exec -it rehab-api /bin/bash
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `REHAB_API_KEY` | *(empty)* | When set, all requests must include `X-API-Key: <value>`. Leave blank to disable auth. |
| `REHAB_LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/exercises` | List all exercises |
| `GET` | `/exercises/{id}` | Exercise details |
| `POST` | `/analyze` | Analyse a single frame |
| `POST` | `/analyze-sequence` | Analyse a sequence of frames |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |

### Example: single-frame analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "exercise_id": 1,
    "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95}]
  }'
```

---

## Monitoring

The `/metrics` endpoint returns Prometheus exposition format. Scrape it with any Prometheus-compatible tool.

Key metrics:

| Metric | Type | Description |
|---|---|---|
| `rehab_uptime_seconds` | gauge | Process uptime |
| `rehab_http_requests_total` | counter | Requests by endpoint and status code |
| `rehab_request_duration_seconds` | histogram | Latency per endpoint |
| `rehab_analysis_results_total` | counter | PASS / FAIL / TRACKING counts |
| `rehab_exercise_requests_total` | counter | Requests per exercise ID |
| `rehab_mean_confidence_ratio` | gauge | Running mean landmark confidence |

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: rehab-api
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
```

---

## Load testing

```bash
# Default: 200 requests, 10 concurrent workers
python scripts/load_test.py

# Custom parameters
python scripts/load_test.py --requests 1000 --concurrency 20 --url http://localhost:8000

# With API key
python scripts/load_test.py --api-key my-secret-key
```

The script exits non-zero if more than 1% of requests fail.

---

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. **Test** — runs the full test suite with coverage (85% threshold enforced).
2. **Build** — builds the Docker image to verify the Dockerfile is correct.

To add image push to a registry, extend the `build` job:

```yaml
- name: Log in to registry
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}

- name: Build and push
  uses: docker/build-push-action@v5
  with:
    push: true
    tags: ghcr.io/<org>/rehab-api:${{ github.sha }}
```

---

## Scaling

| Scenario | Recommendation |
|---|---|
| Single machine | Increase `--workers` in `CMD` (Dockerfile) to match CPU count |
| Multiple machines | Place a reverse proxy (nginx, Caddy) in front and run one container per node |
| Kubernetes | Use the Deployment + Service pattern; liveness probe hits `/health` |

MediaPipe model weights are loaded once per worker process. For a 2-worker setup expect ~400 MB RAM usage at rest.
