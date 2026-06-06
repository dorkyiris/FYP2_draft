# Rehabilitation System: Phases 2-7 Roadmap

**CLAUDE.md Principles Applied Throughout**
- Think Before Coding: Explicit assumptions and tradeoffs stated
- Simplicity First: Minimum code, no speculative features
- Surgical Changes: Touch only what's needed
- Goal-Driven: Clear success criteria per phase

---

## Phase 2: Configuration Management (12 hours)

**Goal:** Move exercise thresholds from code → YAML files  
**Blocks:** Phase 3, API development, clinical validation  
**Success:** Parameters changeable without code changes

### Tasks
1. Create `config/exercises.yaml` (all 3 exercises + thresholds)
2. Create `config/system.yaml` (MediaPipe, smoothing, visualization)
3. Build `config/loader.py` (ConfigManager class)
4. Integrate into `rehabilitationcore/exercises.py`
5. Add unit tests (`tests/unit/test_config.py`)
6. Verify: 40+ tests pass, no regressions

### Files Changed
- **New:** config/, tests/unit/test_config.py
- **Modified:** rehabilitationcore/exercises.py, rehabilitationcore/analyzer.py

### Verification
```bash
pytest tests/unit/ -v          # 40+ tests passing
streamlit run app_refactored.py # UI functional
python -c "from config import ConfigManager"  # Imports work
```

---

## Phase 3: Error Handling & Logging (10 hours)

**Goal:** Robust error recovery, production-ready logging  
**Blocks:** API deployment, monitoring  
**Success:** Graceful failures, traceable logs

### Tasks
1. Add error handling layer:
   - `rehabilitationcore/errors.py` (custom exceptions)
   - Try/catch patterns in analyzer, video pipeline
2. Implement logging:
   - `rehabilitationcore/logging.py` (log configuration)
   - File + console logging via config
   - Rotating logs (10MB, 5 backups)
3. Handle edge cases:
   - Invalid landmarks
   - Missing config files
   - Corrupted video files
   - MediaPipe failures
4. Add tests for error paths
5. Update documentation

### Files Changed
- **New:** rehabilitationcore/errors.py, rehabilitationcore/logging.py
- **Modified:** analyzer.py, video/__init__.py, video/calculator.py, tests/

### Verification
```bash
pytest tests/unit/test_errors.py -v  # Error handling tests pass
# Log file created with proper rotation
python -c "from rehabilitationcore.logging import get_logger"
```

---

## Phase 4: Documentation (8 hours)

**Goal:** Comprehensive dev/user guides  
**Blocks:** Open source release  
**Success:** Clear onboarding, API examples

### Tasks
1. API Documentation:
   - `docs/API.md` (all public classes/functions)
   - Code examples for each module
   - Configuration reference
2. Developer Guide:
   - `docs/DEVELOPMENT.md` (architecture, dataflow)
   - Extending exercises (step-by-step)
   - Testing patterns
3. User Guide:
   - `docs/USER_GUIDE.md` (how to run, modes explained)
   - Troubleshooting
4. Architecture Diagrams:
   - Data flow diagram
   - Module dependency graph
   - System components
5. Update README.md

### Files Created
- docs/API.md
- docs/DEVELOPMENT.md
- docs/USER_GUIDE.md
- docs/ARCHITECTURE.md
- Updated README.md

### Verification
```bash
# All examples compile and run
python docs/examples/*.py
# No broken links in markdown
```

---

## Phase 5: Testing Infrastructure (12 hours)

**Goal:** Comprehensive test coverage, CI/CD ready  
**Blocks:** Merge protection, automated deployment  
**Success:** 85%+ coverage, all tests under 5s

### Tasks
1. Expand unit tests:
   - Integration tests (config → analyzer → results)
   - End-to-end tests (video → landmarks → analysis)
   - Edge case coverage
2. Test fixtures & factories:
   - Mock video frames
   - Landmark datasets
   - Reusable test utilities
3. Coverage reporting:
   - `pytest-cov` configuration
   - Coverage badges
4. Performance tests:
   - Video processing speed
   - Analyzer latency
   - Memory usage tracking
5. Create `tests/` structure:
   - unit/ (existing)
   - integration/
   - e2e/
   - fixtures/

### Files Changed
- **New:** tests/integration/, tests/e2e/, tests/fixtures/, tests/conftest_expanded.py
- **Modified:** tests/unit/ (expanded tests)

### Verification
```bash
pytest --cov=rehabilitationcore --cov=video --cov=config  # 85%+ coverage
pytest -v --durations=10  # All tests < 5s total
```

---

## Phase 6: API Development - REST Endpoints (16 hours)

**Goal:** Production REST API for external clients  
**Blocks:** Mobile/web integrations  
**Success:** OpenAPI docs, working /analyze endpoint

### Tasks
1. Framework setup (Flask or FastAPI):
   - Choose based on simplicity (Flask) vs performance (FastAPI)
   - Project structure: `api/`, `api/routes/`
2. Core endpoints:
   - `POST /analyze` - analyze single frame
   - `POST /analyze-sequence` - analyze video
   - `GET /exercises` - list available exercises
   - `GET /exercises/{id}` - exercise details
   - `GET /config` - current configuration
3. Request/response models:
   - Validation (pydantic for FastAPI)
   - Serialization (JSON landmarks, results)
   - Error responses
4. Authentication (basic):
   - API key validation
   - Rate limiting
5. Deployment readiness:
   - Docker configuration
   - Environment variables
   - Health check endpoint
6. API tests:
   - Unit tests for routes
   - Integration tests with real analyzer

### Files Created
- api/app.py (main application)
- api/routes/analyze.py
- api/routes/exercises.py
- api/routes/config.py
- api/models.py (request/response schemas)
- api/auth.py
- api/middleware.py
- tests/api/
- Dockerfile
- requirements.txt

### Verification
```bash
# API runs
python api/app.py
# Endpoints respond
curl http://localhost:5000/exercises
curl -X POST http://localhost:5000/analyze -d '{...}'
# Tests pass
pytest tests/api/ -v
```

---

## Phase 7: Deployment & Optimization (14 hours)

**Goal:** Production-ready, fast, scalable  
**Blocks:** Enterprise usage  
**Success:** Sub-100ms latency, containerized, monitored

### Tasks
1. Performance optimization:
   - Profile code (cProfile)
   - Optimize hot paths (angle calculations)
   - Cache configuration/models
   - Lazy loading of MediaPipe
2. Containerization:
   - Dockerfile (optimized layers)
   - docker-compose.yml (with logging, volumes)
   - .dockerignore
3. Monitoring & Metrics:
   - Application metrics (request counts, latencies)
   - Business metrics (pass rate, confidence distribution)
   - Prometheus-compatible format
4. CI/CD Pipeline:
   - GitHub Actions workflow
   - Test on commit
   - Build Docker image
   - Push to registry
5. Documentation for deployment:
   - Docker commands
   - Environment setup
   - Scaling guidelines
6. Load testing:
   - Simulate concurrent requests
   - Measure throughput
   - Identify bottlenecks

### Files Created
- Dockerfile (multi-stage)
- docker-compose.yml
- .github/workflows/ci.yml (GitHub Actions)
- monitoring/metrics.py
- scripts/load_test.py
- DEPLOYMENT.md

### Verification
```bash
# Docker runs
docker build -t rehab:latest .
docker run rehab:latest
# Metrics work
curl http://localhost:5000/metrics
# CI passes
git push  # GitHub Actions runs automatically
# Load test passes
python scripts/load_test.py --requests 1000
```

---

## Phase Summary Table

| Phase | Focus | Duration | Key Deliverable | Tests |
|-------|-------|----------|-----------------|-------|
| 1 ✅ | Modularization | 20h | Core engine | 30 |
| 2 | Configuration | 12h | YAML + Loader | 40+ |
| 3 | Error Handling | 10h | Logging framework | 50+ |
| 4 | Documentation | 8h | Dev + User guides | 50+ |
| 5 | Testing | 12h | 85%+ coverage | 70+ |
| 6 | API | 16h | REST endpoints | 85+ |
| 7 | Deployment | 14h | Docker + CI/CD | 85+ |
| **Total** | **Full Stack** | **92h** | **Production Ready** | **85+** |

---

## Architecture: From Monolith to Microservices

```
┌─ Monolith (Original)
│  app.py (303 lines, everything mixed)
│
├─ Phase 1: Modular Core ✅
│  ├─ rehabilitationcore/ (logic, no UI)
│  ├─ video/ (processing)
│  ├─ tests/ (30 unit tests)
│  └─ app_refactored.py (UI using modules)
│
├─ Phase 2: Configuration-Driven
│  ├─ config/ (YAML + loader)
│  └─ Parameter changes without code
│
├─ Phase 3: Production-Grade
│  ├─ Error handling layer
│  ├─ Structured logging
│  └─ Edge case coverage
│
├─ Phase 4: Well-Documented
│  ├─ API documentation
│  ├─ Architecture guides
│  ├─ User manuals
│  └─ Troubleshooting
│
├─ Phase 5: Well-Tested
│  ├─ Unit tests (core logic)
│  ├─ Integration tests (config → analyzer)
│  ├─ E2E tests (video → results)
│  └─ Performance tests
│
├─ Phase 6: API-First
│  ├─ REST endpoints
│  ├─ Request validation
│  ├─ Response serialization
│  └─ Authentication
│
└─ Phase 7: Deployment-Ready
   ├─ Containerization (Docker)
   ├─ Orchestration (docker-compose)
   ├─ CI/CD (GitHub Actions)
   ├─ Monitoring (Prometheus)
   └─ Load tested
```

---

## CLAUDE.md Principles Applied

### 1. Think Before Coding
- ✅ Phases have explicit goals and success criteria
- ✅ Assumptions stated (e.g., "REST API for external clients")
- ✅ Tradeoffs noted (e.g., Flask vs FastAPI)

### 2. Simplicity First
- ✅ Each phase has minimum necessary scope
- ✅ No speculative features (only what's needed)
- ✅ Builds incrementally on previous phases

### 3. Surgical Changes
- ✅ Each phase focuses on one concern
- ✅ Clear file organization (new folders, modified files listed)
- ✅ No "nice-to-have" improvements mixed in

### 4. Goal-Driven Execution
- ✅ Every phase has success criteria
- ✅ Verification commands specified
- ✅ Tests define completion

---

## Execution Order (Sequential Dependency)

```
Phase 1 ✅ (Complete)
    ↓
Phase 2 (Configuration) ← Start here next
    ↓
Phase 3 (Error Handling)
    ↓
Phase 4 (Documentation)
    ↓
Phase 5 (Testing Infrastructure)
    ↓
Phase 6 (API Development)
    ↓
Phase 7 (Deployment & Optimization)
    ↓
🚀 Production-Ready System
```

**Estimated Timeline:** 10-12 weeks (92 hours over 12 weeks @ 8h/week)

---

**Document Generated:** 2026-06-01  
**Status:** Ready for Phase 2 Implementation
