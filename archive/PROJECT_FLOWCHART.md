# Rehabilitation System: Complete Project Flowchart

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                       REHABILITATION EXERCISE SYSTEM                           ║
║                                    (FYP2)                                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│ INPUT SOURCES                                                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   CSV Data   │    │  Video File  │    │ Live Camera  │    │ API Request  │  │
│  │   (Phase 1)  │    │  (Phase 1)   │    │ (Phase 1)    │    │ (Phase 6)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │                    │         │
│         └───────────────────┼────────────────────┼────────────────────┘         │
│                             │                    │                             │
└─────────────────────────────┼────────────────────┼─────────────────────────────┘
                              │                    │
                              ▼                    ▼
                    ┌──────────────────┐  ┌──────────────────┐
                    │  Video Capture   │  │  Landmark Data   │
                    │  (OpenCV)        │  │  (from JSON/API) │
                    └──────────────────┘  └──────────────────┘
                              │                    │
                              └────────┬───────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────┐
        │  VIDEO/FRAME PROCESSING (video/ module)                │
        ├──────────────────────────────────────────────────────────┤
        │                                                          │
        │  ┌────────────────────────────────────────────────┐     │
        │  │ PoseExtractionPipeline                         │     │
        │  │ ├─ extract_video(video.mp4)                    │     │
        │  │ │  └─ → List[List[Landmark]]                  │     │
        │  │ ├─ extract_frame(frame)                        │     │
        │  │ │  └─ → List[Landmark]                        │     │
        │  │ └─ MediaPipe Pose Detection                    │     │
        │  │    └─ Returns 33 pose landmarks per frame      │     │
        │  └────────────────────────────────────────────────┘     │
        │                         │                                │
        │                         ▼                                │
        │  ┌────────────────────────────────────────────────┐     │
        │  │ KinematicCalculator                            │     │
        │  │ ├─ landmarks_to_dataframe()                    │     │
        │  │ │  └─ Pandas DataFrame with x,y,z,visibility  │     │
        │  │ └─ extract_kinematic_angles()                  │     │
        │  │    └─ Shoulder & elbow angles                 │     │
        │  └────────────────────────────────────────────────┘     │
        │                         │                                │
        └─────────────────────────┼────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────────────┐
        │  CONFIGURATION LAYER (Phase 2: config/)                 │
        ├──────────────────────────────────────────────────────────┤
        │                                                          │
        │  ┌──────────────────┐  ┌──────────────────┐             │
        │  │ exercises.yaml   │  │  system.yaml     │             │
        │  │ ├─ Exercise 1    │  │ ├─ MediaPipe     │             │
        │  │ ├─ Exercise 2    │  │ ├─ Smoothing     │             │
        │  │ └─ Exercise 3    │  │ ├─ Visualization │             │
        │  │ └─ Thresholds    │  │ └─ Logging       │             │
        │  └──────────────────┘  └──────────────────┘             │
        │           │                    │                        │
        │           └────────┬───────────┘                        │
        │                    │                                    │
        │            ConfigManager                               │
        │            └─ Loads YAML, validates, caches            │
        │                                                          │
        └──────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────────┐
        │  CORE ENGINE (rehabilitationcore/)                      │
        ├──────────────────────────────────────────────────────────┤
        │                                                          │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ Models (models.py)                               │   │
        │  │ ├─ Landmark (x, y, z, visibility)               │   │
        │  │ ├─ ExerciseDefinition (thresholds, rules)       │   │
        │  │ ├─ ExerciseResult (status, angle, feedback)     │   │
        │  │ ├─ AngleThreshold (min, max, evaluate)          │   │
        │  │ └─ ExerciseStatus enum (PASS, FAIL, TRACKING)   │   │
        │  └──────────────────────────────────────────────────┘   │
        │                         │                                │
        │                         ▼                                │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ Biomechanics (biomechanics.py)                   │   │
        │  │ ├─ calculate_2d_angle(p1, p2, p3)               │   │
        │  │ ├─ calculate_distance(p1, p2)                   │   │
        │  │ ├─ smooth_signal(values, method, span)          │   │
        │  │ └─ validate_landmark_chain(landmarks, indices)  │   │
        │  │                                                  │   │
        │  │ Pure functions, fully tested, no side effects    │   │
        │  └──────────────────────────────────────────────────┘   │
        │                         │                                │
        │                         ▼                                │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ Exercise Definitions (exercises.py)              │   │
        │  │ └─ EXERCISES registry                            │   │
        │  │    ├─ Exercise 1: Arm Abduction                  │   │
        │  │    │  └─ Elbow angle ≥ 160°                     │   │
        │  │    ├─ Exercise 2: V-to-W Transition             │   │
        │  │    │  └─ Shoulder 85-125°                       │   │
        │  │    └─ Exercise 3: Inclined Push-up              │   │
        │  │       └─ Elbow angle ≤ 100°                     │   │
        │  └──────────────────────────────────────────────────┘   │
        │                         │                                │
        │                         ▼                                │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ ExerciseAnalyzer (analyzer.py)                   │   │
        │  │ ├─ analyze(landmarks, exercise)                 │   │
        │  │ │  └─ Single frame analysis                      │   │
        │  │ │     1. Validate landmarks                      │   │
        │  │ │     2. Calculate angles                        │   │
        │  │ │     3. Evaluate thresholds                     │   │
        │  │ │     4. Generate feedback                       │   │
        │  │ │     5. Return ExerciseResult                   │   │
        │  │ │                                                │   │
        │  │ ├─ analyze_sequence(landmark_frames, exercise)   │   │
        │  │ │  └─ Multiple frames (video)                    │   │
        │  │ │     └─ Returns List[ExerciseResult]            │   │
        │  │ │                                                │   │
        │  │ └─ Private: _calculate_angles()                  │   │
        │  │    └─ Orchestrates angle calculations            │   │
        │  └──────────────────────────────────────────────────┘   │
        │                                                          │
        │  Phase 3: Error Handling & Logging                      │
        │  ├─ errors.py (custom exceptions)                       │
        │  │  └─ InvalidLandmarkError, ConfigError, etc.         │
        │  └─ logging.py (structured logging)                     │
        │     └─ File + console, rotating logs                   │
        │                                                          │
        └──────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼ ExerciseResult
                                   │
        ┌──────────────────────────────────────────────────────────┐
        │  RENDERING & PRESENTATION (video/renderer.py)           │
        ├──────────────────────────────────────────────────────────┤
        │                                                          │
        │  VideoRenderer.draw_clinical_overlay()                  │
        │  ├─ Input: Frame + Landmarks + ExerciseResult          │
        │  ├─ Steps:                                             │
        │  │  1. Draw skeleton (shoulder-elbow-wrist)            │
        │  │  2. Draw joint circles (colored by status)          │
        │  │  3. Add feedback text                               │
        │  │  4. Add confidence score                            │
        │  └─ Output: Annotated frame                            │
        │                                                          │
        │  VideoRenderer.create_output_video()                    │
        │  └─ Batch process video frames                         │
        │                                                          │
        └──────────────────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
        ┌─────────────────────────┐    ┌─────────────────────────┐
        │  UI LAYER               │    │  API LAYER (Phase 6)    │
        │  (Streamlit)            │    │  (Flask/FastAPI)        │
        ├─────────────────────────┤    ├─────────────────────────┤
        │                         │    │                         │
        │ app_refactored.py       │    │ api/routes/analyze.py   │
        │ ├─ Mode 1: CSV Analysis │    │ ├─ POST /analyze        │
        │ ├─ Mode 2: Video Upload │    │ ├─ POST /analyze-seq    │
        │ ├─ Mode 3: Live Webcam  │    │ ├─ GET /exercises       │
        │ └─ Mode 4: Dashboard    │    │ └─ GET /config          │
        │                         │    │                         │
        │ Uses:                   │    │ Uses:                   │
        │ ├─ ExerciseAnalyzer     │    │ ├─ ExerciseAnalyzer     │
        │ ├─ VideoRenderer        │    │ ├─ Request Validation   │
        │ ├─ PoseExtraction       │    │ ├─ Error Handling       │
        │ └─ ConfigManager        │    │ ├─ Authentication       │
        │                         │    │ └─ Rate Limiting        │
        │                         │    │                         │
        └──────────────┬──────────┘    └────────────┬────────────┘
                       │                           │
                       ▼                           ▼
        ┌──────────────────────┐    ┌────────────────────────────┐
        │  OUTPUTS             │    │  API RESPONSES             │
        ├──────────────────────┤    ├────────────────────────────┤
        │                      │    │                            │
        │ ✓ Annotated frames   │    │ JSON Response:             │
        │ ✓ Video files        │    │ {                          │
        │ ✓ CSV exports        │    │   "status": "PASS",        │
        │ ✓ Analytics charts   │    │   "angle": 165.3,          │
        │ ✓ Real-time feedback │    │   "feedback": "...",       │
        │ ✓ Confidence scores  │    │   "confidence": 0.92       │
        │                      │    │ }                          │
        │                      │    │                            │
        └──────────────────────┘    └────────────────────────────┘
                       │                           │
                       └──────────────┬────────────┘
                                      │
                                      ▼
        ┌──────────────────────────────────────────────────────────┐
        │  DEPLOYMENT & MONITORING (Phase 7)                      │
        ├──────────────────────────────────────────────────────────┤
        │                                                          │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ Docker Containerization                          │   │
        │  │ ├─ Dockerfile (multi-stage)                      │   │
        │  │ ├─ docker-compose.yml                           │   │
        │  │ └─ .dockerignore                                │   │
        │  └──────────────────────────────────────────────────┘   │
        │                                                          │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ CI/CD Pipeline (GitHub Actions)                  │   │
        │  │ ├─ Tests on commit                               │   │
        │  │ ├─ Build Docker image                            │   │
        │  │ ├─ Push to registry                              │   │
        │  │ └─ Deploy to staging/prod                        │   │
        │  └──────────────────────────────────────────────────┘   │
        │                                                          │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ Monitoring & Metrics                             │   │
        │  │ ├─ Request latency                               │   │
        │  │ ├─ Throughput (requests/sec)                     │   │
        │  │ ├─ Error rates                                   │   │
        │  │ ├─ Model performance (pass rate)                 │   │
        │  │ └─ System health (CPU, memory)                   │   │
        │  └──────────────────────────────────────────────────┘   │
        │                                                          │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ Logging Infrastructure                           │   │
        │  │ ├─ Application logs (rotating files)             │   │
        │  │ ├─ Error logs                                    │   │
        │  │ ├─ Audit logs (API calls)                        │   │
        │  │ └─ Aggregation (ELK stack, Splunk, etc.)         │   │
        │  └──────────────────────────────────────────────────┘   │
        │                                                          │
        └──────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────────┐
        │  🚀 PRODUCTION-READY SYSTEM                             │
        │                                                          │
        │  ✓ Modular, testable core                              │
        │  ✓ Configuration-driven parameters                     │
        │  ✓ Error handling & logging                            │
        │  ✓ Well-documented (API, dev, user guides)             │
        │  ✓ Comprehensive tests (85%+ coverage)                 │
        │  ✓ REST API for integrations                           │
        │  ✓ Containerized & automated deployment                │
        │  ✓ Monitored and observable                            │
        │                                                          │
        └──────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram: Single Frame Analysis

```
Input Frame (RGB image, 1920x1080)
           │
           ▼
┌──────────────────────────────┐
│ MediaPipe Pose Detection     │
│ (PoseExtractionPipeline)     │
│ - Confidence: 0.65+          │
│ - Output: 33 landmarks       │
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Landmark Validation          │
│ - Check visibility > 0.65    │
│ - Verify required indices    │
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Angle Calculation            │
│ - P1 (hip/shoulder)          │
│ - P2 (shoulder/elbow)        │
│ - P3 (elbow/wrist)           │
│ → calculate_2d_angle()       │
│ → Result: 0-180°             │
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Threshold Evaluation         │
│ - Load from config           │
│ - Compare angle vs threshold │
│ - Determine status:          │
│   • PASS (>= min)            │
│   • FAIL (< min)             │
│   • TRACKING (low visibility)│
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Feedback Generation          │
│ - Get rule from config       │
│ - Include angle value        │
│ - Calculate confidence       │
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ ExerciseResult               │
│ {                            │
│   status: PASS               │
│   primary_angle: 165.3°      │
│   feedback: "✅ Form: PASS"  │
│   confidence: 0.92           │
│ }                            │
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Render Overlay               │
│ - Draw skeleton              │
│ - Color by status (green)    │
│ - Add text feedback          │
│ - Add confidence %           │
└──────────────────────────────┘
           │
           ▼
   Annotated Frame Output
   (Display or save)
```

---

## Module Dependency Graph (All Phases)

```
rehabilitationcore/
├── models.py
│   └── No dependencies
│
├── biomechanics.py
│   └── → models.py, numpy
│
├── exercises.py
│   └── → models.py
│
├── analyzer.py (Phase 1 - Consolidated)
│   └── → models.py, biomechanics.py, exercises.py
│
├── errors.py (Phase 3)
│   └── No dependencies
│
├── logging.py (Phase 3)
│   └── → config/
│
└── __init__.py
    └── Public API exports

video/
├── __init__.py (PoseExtractionPipeline)
│   └── → models.py, mediapipe, cv2, logging
│
├── calculator.py
│   └── → models.py, biomechanics.py, pandas, numpy
│
├── renderer.py (Phase 1 - Simplified)
│   └── → models.py, cv2, numpy
│
└── (no circular dependencies)

config/ (Phase 2)
├── __init__.py
├── loader.py (ConfigManager)
│   └── → yaml, pathlib
│
├── exercises.yaml
│   └── Exercise definitions (data)
│
└── system.yaml
    └── System parameters (data)

tests/ (Phase 5)
├── unit/
│   ├── test_biomechanics.py
│   ├── test_analyzer.py
│   ├── test_config.py (Phase 2)
│   ├── test_errors.py (Phase 3)
│   └── conftest.py (fixtures)
│
├── integration/
│   ├── test_config_to_analyzer.py
│   └── test_video_to_analysis.py
│
└── e2e/
    └── test_full_pipeline.py

api/ (Phase 6)
├── app.py (Flask/FastAPI)
│   └── → rehabilitationcore, config, error handling
│
├── routes/
│   ├── analyze.py (POST /analyze)
│   ├── exercises.py (GET /exercises)
│   └── config.py (GET /config)
│
├── models.py (Request/Response schemas)
├── auth.py (API key validation)
├── middleware.py (logging, error handling)
└── tests/

monitoring/ (Phase 7)
├── metrics.py (Prometheus metrics)
└── (optional: OpenTelemetry, etc.)

docs/ (Phase 4)
├── API.md
├── DEVELOPMENT.md
├── USER_GUIDE.md
├── ARCHITECTURE.md
└── examples/

deployment/ (Phase 7)
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
└── scripts/
    └── load_test.py
```

---

## Quality Gates by Phase

```
Phase 1: Modularization ✅
├─ Tests: 30/30 passing ✓
├─ Coverage: 100% (biomechanics, analyzer) ✓
├─ No circular dependencies ✓
└─ Code review: Clean, surgical changes ✓

Phase 2: Configuration
├─ Tests: 40+ passing (incl. config tests)
├─ Can change threshold without code change
├─ YAML validates against schema
└─ All Phase 1 tests still pass (regression check)

Phase 3: Error Handling
├─ Tests: 50+ passing (incl. error tests)
├─ All exception paths covered
├─ Logging structured and testable
└─ Graceful failures under edge cases

Phase 4: Documentation
├─ API docs complete (all public functions)
├─ Examples compile and run
├─ No broken markdown links
└─ Troubleshooting guide covers common issues

Phase 5: Testing Infrastructure
├─ Tests: 70+ passing (unit + integration + e2e)
├─ Coverage: 85%+
├─ All tests run in < 5 seconds
└─ Performance benchmarks established

Phase 6: API Development
├─ Tests: 85+ passing (API integration tests)
├─ All endpoints respond correctly
├─ Request validation working
├─ OpenAPI docs generated
└─ Docker image builds successfully

Phase 7: Deployment & Optimization
├─ Tests: 85+ passing (no regressions)
├─ Load test: 1000 req/min sustained
├─ Latency: < 100ms per request
├─ CI/CD passes on all commits
└─ Monitoring metrics visible in dashboard
```

---

## Success Metrics by Phase

| Phase | Metric | Target | Verification |
|-------|--------|--------|--------------|
| 1 ✅ | Test coverage | 100% math | `pytest --cov` |
| 1 ✅ | Module count | 4+ | `ls -la rehabilitationcore/` |
| 2 | Config files | 2 (exercises, system) | `config/` exists + valid YAML |
| 2 | Parameter flexibility | Thresholds changeable | Modify `.yaml`, no code changes |
| 3 | Error coverage | 100% error paths | `pytest tests/unit/test_errors.py` |
| 3 | Log file rotation | Working | Check `logs/` directory |
| 4 | Doc completeness | 4+ guides | `ls -la docs/` |
| 4 | Example quality | All runnable | `python docs/examples/*.py` |
| 5 | Test count | 70+ | `pytest --collect-only` |
| 5 | Coverage | 85%+ | `pytest --cov` report |
| 5 | Speed | < 5s | `pytest --durations=10` |
| 6 | API endpoints | 5+ working | `curl` tests pass |
| 6 | Response schema | Valid JSON | Pydantic validation |
| 6 | Docker image | Buildable | `docker build` succeeds |
| 7 | Load throughput | 1000 req/min | `load_test.py` passes |
| 7 | Latency p99 | < 200ms | Prometheus metrics |
| 7 | CI/CD | Automated | GitHub Actions runs |

---

## CLAUDE.md Alignment Checklist

### Principle 1: Think Before Coding ✅
- [x] Phases have explicit goals
- [x] Success criteria stated upfront
- [x] Tradeoffs documented (e.g., Flask vs FastAPI)
- [x] Assumptions listed per phase
- [x] Flowchart clarifies scope boundaries

### Principle 2: Simplicity First ✅
- [x] Each phase does one thing well
- [x] No speculative features
- [x] Builds incrementally on Phase 1
- [x] Configuration is YAML (readable, not code)
- [x] API is simple CRUD + analyze endpoints

### Principle 3: Surgical Changes ✅
- [x] Each phase modifies specific files
- [x] No "nice-to-have" scope creep
- [x] Tests define boundaries
- [x] Each phase isolated from others
- [x] Backward compatibility maintained

### Principle 4: Goal-Driven Execution ✅
- [x] Success criteria for each phase
- [x] Verification commands specified
- [x] Tests are the definition of done
- [x] Dependencies mapped (sequential)
- [x] Quality gates documented

---

**Total Project Timeline:** 92 hours (12 weeks @ 8h/week)  
**Target Completion:** 12 weeks from Phase 2 start  
**Status:** Ready to proceed with Phase 2 ✓

Generated: 2026-06-01
