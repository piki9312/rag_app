"""RAG API ベンチマーク — レイテンシ計測スクリプト。

Usage:
    python scripts/bench_api.py [--url URL] [--runs N] [--warmup N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000/ask"

PAYLOAD = {
    "question": "コアタイム何時から何時までですか？",
    "retrieval_k": 60,
    "context_k": 3,
    "use_multi": False,
    "debug": False,
    "max_new_tokens": 256,
}


def post_json(url: str, data: dict, timeout: int = 120) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG API latency benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help="ask endpoint URL")
    parser.add_argument("--runs", type=int, default=10, help="number of runs")
    parser.add_argument("--warmup", type=int, default=2, help="warmup iterations")
    args = parser.parse_args()

    # warmup
    for _ in range(args.warmup):
        post_json(args.url, PAYLOAD)

    times: list[float] = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        res = post_json(args.url, PAYLOAD)
        dt = time.perf_counter() - t0
        times.append(dt)

        answer = res.get("answer") or ""
        print(f"[{i+1}/{args.runs}] {dt:.3f}s | answer_len={len(answer)}")

    avg = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    print("\n--- Summary ---")
    print(f"runs={args.runs}, warmup={args.warmup}")
    print(f"avg={avg:.3f}s, p50={p50:.3f}s, p95≈{p95:.3f}s")
    print("raw:", [round(t, 3) for t in times])


if __name__ == "__main__":
    main()
