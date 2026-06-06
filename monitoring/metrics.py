"""
Prometheus-compatible metrics for the Tele-Rehabilitation API.

Emits text in the Prometheus exposition format (no external dependency).
Mount the /metrics route in the FastAPI app to expose these to a scraper.
"""

import threading
import time
from collections import defaultdict
from typing import Dict


class MetricsCollector:
    """Thread-safe store for application and business metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time: float = time.time()

        # Request counters  {endpoint: {status_code: count}}
        self._request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Latency buckets (seconds) — cumulative histogram
        self._latency_buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, float("inf")]
        self._latency_counts: Dict[str, list] = defaultdict(
            lambda: [0] * len(self._latency_buckets)
        )
        self._latency_sum: Dict[str, float] = defaultdict(float)
        self._latency_total: Dict[str, int] = defaultdict(int)

        # Business metrics
        self._analysis_results: Dict[str, int] = defaultdict(int)  # {PASS|FAIL|TRACKING: count}
        self._exercise_requests: Dict[int, int] = defaultdict(int)  # {exercise_id: count}
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0

    # ── Request tracking ──────────────────────────────────────────────────────

    def record_request(self, endpoint: str, status_code: int, latency_s: float) -> None:
        with self._lock:
            self._request_counts[endpoint][str(status_code)] += 1
            self._latency_sum[endpoint] += latency_s
            self._latency_total[endpoint] += 1
            buckets = self._latency_counts[endpoint]
            for i, bound in enumerate(self._latency_buckets):
                if latency_s <= bound:
                    buckets[i] += 1

    # ── Business metrics ──────────────────────────────────────────────────────

    def record_analysis(self, exercise_id: int, status: str, confidence: float) -> None:
        with self._lock:
            self._analysis_results[status] += 1
            self._exercise_requests[exercise_id] += 1
            self._confidence_sum += confidence
            self._confidence_count += 1

    # ── Prometheus text format ─────────────────────────────────────────────────

    def render(self) -> str:
        lines: list[str] = []
        uptime = time.time() - self._start_time

        with self._lock:
            # Uptime
            lines += [
                "# HELP rehab_uptime_seconds Seconds since the API process started",
                "# TYPE rehab_uptime_seconds gauge",
                f"rehab_uptime_seconds {uptime:.2f}",
                "",
            ]

            # Request counters
            lines += [
                "# HELP rehab_http_requests_total Total HTTP requests by endpoint and status",
                "# TYPE rehab_http_requests_total counter",
            ]
            for endpoint, codes in self._request_counts.items():
                for code, count in codes.items():
                    lines.append(
                        f'rehab_http_requests_total{{endpoint="{endpoint}",status="{code}"}} {count}'
                    )
            lines.append("")

            # Latency histograms
            lines += [
                "# HELP rehab_request_duration_seconds Request latency histogram",
                "# TYPE rehab_request_duration_seconds histogram",
            ]
            for endpoint, buckets in self._latency_counts.items():
                cumulative = 0
                for i, bound in enumerate(self._latency_buckets):
                    cumulative += buckets[i]
                    le = "+Inf" if bound == float("inf") else str(bound)
                    lines.append(
                        f'rehab_request_duration_seconds_bucket{{endpoint="{endpoint}",le="{le}"}} {cumulative}'
                    )
                lines.append(
                    f'rehab_request_duration_seconds_sum{{endpoint="{endpoint}"}} {self._latency_sum[endpoint]:.6f}'
                )
                lines.append(
                    f'rehab_request_duration_seconds_count{{endpoint="{endpoint}"}} {self._latency_total[endpoint]}'
                )
            lines.append("")

            # Analysis result distribution
            lines += [
                "# HELP rehab_analysis_results_total Exercise analysis outcomes",
                "# TYPE rehab_analysis_results_total counter",
            ]
            for status, count in self._analysis_results.items():
                lines.append(f'rehab_analysis_results_total{{status="{status}"}} {count}')
            lines.append("")

            # Per-exercise request counts
            lines += [
                "# HELP rehab_exercise_requests_total Requests per exercise ID",
                "# TYPE rehab_exercise_requests_total counter",
            ]
            for ex_id, count in self._exercise_requests.items():
                lines.append(f'rehab_exercise_requests_total{{exercise_id="{ex_id}"}} {count}')
            lines.append("")

            # Mean confidence
            mean_conf = (
                self._confidence_sum / self._confidence_count
                if self._confidence_count > 0
                else 0.0
            )
            lines += [
                "# HELP rehab_mean_confidence_ratio Running mean landmark visibility confidence",
                "# TYPE rehab_mean_confidence_ratio gauge",
                f"rehab_mean_confidence_ratio {mean_conf:.4f}",
                "",
            ]

        return "\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────

_collector = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _collector


def metrics_text() -> str:
    """Return current metrics in Prometheus exposition format."""
    return _collector.render()
