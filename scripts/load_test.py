#!/usr/bin/env python3
"""
Load test for the Tele-Rehabilitation REST API.

Usage:
    python scripts/load_test.py                        # defaults
    python scripts/load_test.py --requests 500 --concurrency 20
    python scripts/load_test.py --url http://myhost:8000
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urllib_request
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Minimal valid payload: 33 landmarks for exercise 1 (shoulder lift)
# ---------------------------------------------------------------------------
_LANDMARK = {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95}
_ANALYZE_PAYLOAD = json.dumps(
    {"exercise_id": 1, "landmarks": [_LANDMARK] * 33}
).encode()


def _post(base_url: str, api_key: str | None) -> tuple[int, float]:
    """Send one POST /analyze. Returns (status_code, latency_s)."""
    req = urllib_request.Request(
        url=f"{base_url}/analyze",
        data=_ANALYZE_PAYLOAD,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        req.add_header("X-API-Key", api_key)

    t0 = time.perf_counter()
    try:
        with urllib_request.urlopen(req, timeout=10) as resp:
            resp.read()
            return resp.status, time.perf_counter() - t0
    except URLError as exc:
        latency = time.perf_counter() - t0
        code = getattr(getattr(exc, "reason", None), "errno", 0) or 0
        return code, latency


def run(base_url: str, total: int, concurrency: int, api_key: str | None) -> None:
    print(f"Load test → {base_url}/analyze")
    print(f"  {total} requests  |  {concurrency} concurrent workers\n")

    results: list[tuple[int, float]] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_post, base_url, api_key) for _ in range(total)]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % max(1, total // 10) == 0:
                print(f"  {i}/{total} done …")

    wall_elapsed = time.perf_counter() - wall_start

    # ── Stats ────────────────────────────────────────────────────────────────
    latencies = [r[1] for r in results]
    status_counts: dict[int, int] = {}
    for code, _ in results:
        status_counts[code] = status_counts.get(code, 0) + 1

    success = status_counts.get(200, 0)
    failure = total - success

    latencies_200 = [r[1] for r in results if r[0] == 200]

    print("\n── Results ─────────────────────────────────────────────────────")
    print(f"  Total requests  : {total}")
    print(f"  Successful (2xx): {success}")
    print(f"  Failed          : {failure}")
    print(f"  Status codes    : {dict(sorted(status_counts.items()))}")
    print(f"  Wall time       : {wall_elapsed:.2f} s")
    print(f"  Throughput      : {total / wall_elapsed:.1f} req/s")

    if latencies_200:
        print(f"\n── Latency (successful requests) ───────────────────────────────")
        print(f"  Min   : {min(latencies_200)*1000:.1f} ms")
        print(f"  Mean  : {statistics.mean(latencies_200)*1000:.1f} ms")
        print(f"  Median: {statistics.median(latencies_200)*1000:.1f} ms")
        print(f"  P95   : {sorted(latencies_200)[int(len(latencies_200)*0.95)]*1000:.1f} ms")
        print(f"  Max   : {max(latencies_200)*1000:.1f} ms")

    # Exit non-zero if more than 1% requests failed
    if failure / total > 0.01:
        print(f"\nFAIL: {failure}/{total} requests failed (>{1}% threshold)")
        sys.exit(1)
    else:
        print("\nPASS: error rate within acceptable limits.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tele-Rehab API load test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--requests", type=int, default=200, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--api-key", default=None, help="X-API-Key header value")
    args = parser.parse_args()

    run(args.url, args.requests, args.concurrency, args.api_key)


if __name__ == "__main__":
    main()
