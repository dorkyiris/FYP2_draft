# FYP 2: Vision-Based Tele-Rehabilitation System

Upper-limb stroke rehabilitation assessment using MediaPipe pose estimation and biomechanical angle analysis.

**Live demo:** https://fyp2draft-tele7zpcsvsxbydhrovtrn.streamlit.app

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app_refactored.py        # dashboard (webcam or video file)
uvicorn api.app:app --reload           # REST API at http://localhost:8000
```

---

## What It Does

The system analyses four clinical upper-limb exercises from video or a live webcam:

| # | Exercise | Pass Condition |
|---|---|---|
| 1 | Lifting an object | Shoulder angle ≥ 90° |
| 2 | Extending the elbow | Elbow angle ≥ 160° |
| 3 | Lifting the wrist | Wrist angle ≥ 15° |
| 4 | Opening the hand | Hand open ≥ 45° |

For each frame the system extracts 33 body landmarks via MediaPipe, computes joint angles, evaluates clinical thresholds, and overlays real-time feedback on the video.

---

## Project Structure

```
FYP2/
├── app_refactored.py          Streamlit dashboard (4 modes)
├── requirements.txt           Pinned dependencies
│
├── config/
│   ├── exercises.yaml         Clinical thresholds (edit without code changes)
│   ├── system.yaml            MediaPipe / smoothing / visualisation settings
│   └── loader.py              ConfigManager
│
├── rehabilitationcore/        Core engine — no UI/video dependencies
│   ├── models.py              Landmark, ExerciseDefinition, ExerciseResult
│   ├── biomechanics.py        2-D angle calculation, landmark validation
│   ├── exercises.py           Exercise registry (loaded from YAML)
│   ├── analyzer.py            ExerciseAnalyzer (single frame + sequence)
│   ├── errors.py              Custom exception hierarchy
│   └── logging_config.py      Structured logging with rotation
│
├── video/                     MediaPipe extraction, OpenCV rendering
│
├── api/                       FastAPI REST API
│   ├── app.py                 Application factory + metrics middleware
│   ├── auth.py                Optional API-key auth (REHAB_API_KEY)
│   ├── models.py              Pydantic request/response schemas
│   └── routes/
│       ├── health.py          GET /health
│       ├── exercises.py       GET /exercises, GET /exercises/{id}
│       └── analyze.py         POST /analyze, POST /analyze-sequence
│
├── monitoring/
│   └── metrics.py             Prometheus-compatible metrics (GET /metrics)
│
├── tests/
│   ├── unit/                  167 unit tests
│   ├── integration/           Config → analyzer pipeline tests
│   ├── api/                   FastAPI route tests (TestClient)
│   └── fixtures/              Shared test data
│
├── scripts/
│   └── load_test.py           Concurrent load tester (stdlib only)
│
├── docs/                      Developer and user guides
│
├── Dockerfile                 Multi-stage build (builder + runtime)
├── docker-compose.yml         One-command local deployment
├── .dockerignore
├── .github/workflows/ci.yml   GitHub Actions: test + Docker build
└── DEPLOYMENT.md              Docker, env vars, monitoring, scaling
```

---

## Phase-by-Phase Build History

### Phase 1 — Modularisation

Extracted the original 303-line `app.py` monolith into a layered architecture:

- `rehabilitationcore/` — pure Python core with no UI or I/O dependencies
- `video/` — MediaPipe and OpenCV wrappers isolated from the logic
- `app_refactored.py` — thin Streamlit front-end that calls the core
- First 30 unit tests written covering biomechanics and angle evaluation

### Phase 2 — Configuration Management

Moved all clinical thresholds and system settings out of code and into YAML:

- `config/exercises.yaml` — exercise definitions, angle thresholds, feedback strings
- `config/system.yaml` — MediaPipe confidence levels, smoothing parameters, visualisation colours
- `config/loader.py` — `ConfigManager` class with typed accessors and validation
- Exercise parameters can now be changed without touching a single line of Python

### Phase 3 — Error Handling and Logging

Made the system production-safe by adding structured error handling and logging:

- `rehabilitationcore/errors.py` — custom exception hierarchy (`RehabError` → `ConfigError`, `LandmarkError`, `AnalysisError`)
- `rehabilitationcore/logging_config.py` — rotating file + console logging, configurable level
- Graceful degradation on invalid landmarks, missing YAML keys, and MediaPipe failures
- Error path coverage added to the test suite (test count grew to 50+)

### Phase 4 — Documentation

Wrote comprehensive developer and user-facing documentation:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module graph, data-flow diagram, design decisions
- [docs/API.md](docs/API.md) — public classes and functions reference with type signatures
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — adding exercises, testing patterns, file structure
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — running the app, mode descriptions, troubleshooting

### Phase 5 — Testing Infrastructure

Built a comprehensive, layered test suite targeting 85%+ coverage:

- `tests/unit/` — fast isolated tests for models, biomechanics, config, errors, analyzer, calculator
- `tests/integration/` — end-to-end config → analyzer pipeline tests with real YAML
- `tests/api/` — FastAPI route tests using `TestClient` (no running server needed)
- `tests/fixtures/` — shared landmark datasets and video-frame factories
- Coverage gate enforced via `pyproject.toml` (`fail_under = 85`)
- 167 tests, all passing in under 2 seconds

### Phase 6 — REST API

Added a production-quality FastAPI layer so external clients (mobile apps, dashboards) can call the engine over HTTP:

- `POST /analyze` — analyse a single frame of landmarks
- `POST /analyze-sequence` — analyse a full video sequence
- `GET /exercises` / `GET /exercises/{id}` — exercise catalogue
- `GET /health` — liveness check
- Optional API-key authentication via `REHAB_API_KEY` env var
- Pydantic v2 request/response validation with descriptive error messages
- Full OpenAPI docs at `/docs` and `/redoc`

### Phase 7 — Deployment and Observability

Containerised the system and added production operations tooling:

- **Dockerfile** — two-stage build (builder → slim runtime); non-root user; health check built in
- **docker-compose.yml** — one-command local stack with log volume and configurable env vars
- **Prometheus metrics** (`GET /metrics`) — request counters, latency histograms, PASS/FAIL rates, confidence gauge; no external library dependency
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — runs all tests with coverage on every push; builds the Docker image to verify it compiles
- **Load test** (`scripts/load_test.py`) — concurrent stress test using stdlib only; reports throughput, P95 latency, and exits non-zero on >1% error rate
- **DEPLOYMENT.md** — Docker commands, environment variables, Prometheus scrape config, scaling guidance

---

## Tests

```bash
pytest tests/ -v                                    # all 167 tests
pytest tests/ --cov=rehabilitationcore --cov=api    # with coverage report
python scripts/load_test.py                         # load test (API must be running)
```

---

## Docker

```bash
docker compose up --build -d        # start
curl http://localhost:8000/health   # verify
curl http://localhost:8000/metrics  # Prometheus metrics
docker compose down                 # stop
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment reference.

---

## Changing Exercise Thresholds

Edit `config/exercises.yaml` — no code changes needed. Restart the app or API after saving.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module graph, data flow, design decisions |
| [docs/API.md](docs/API.md) | Public classes and functions reference |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Adding exercises, testing patterns, file structure |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Running the app, modes, troubleshooting |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker, env vars, monitoring, scaling |

---

## Implementation Status

| Phase | Description | Status |
|---|---|---|
| 1 | Modularise monolith → testable core | Done |
| 2 | YAML configuration management | Done |
| 3 | Error handling and logging | Done |
| 4 | Documentation | Done |
| 5 | Expanded test infrastructure (167 tests, 85%+ coverage) | Done |
| 6 | FastAPI REST API with auth and validation | Done |
| 7 | Docker, CI/CD, Prometheus metrics, load testing | Done |

---

**Dataset:** Nandana et al. 2026 — 491 upper-limb stroke rehabilitation videos, 4 exercises, 10 volunteers; 3D-CNN baseline achieves **40% test accuracy** (Table 7, Data in Brief, doi:10.1016/j.dib.2026.112819).  
**This work:** EMA filter reduces spatial tracking error by **25.5%** (0.04355 → 0.03314 normalised units).
