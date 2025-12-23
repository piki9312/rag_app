import time
import json
import statistics
import urllib.request

API_URL = "http://127.0.0.1:8000/ask"  
QUERY = "このシステムは何をしますか？"
RUNS = 5
WARMUP = 2

PAYLOAD = {
    "question": "このシステムは何をしますか？",
    "retrieval_k": 60,
    "context_k": 3,
    "use_multi": True,  # Multi-Query Retrievalを使うか
    "debug": False,   # まずはFalse推奨（レスポンスが軽くなる）
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

def main():
    # warmup
    for _ in range(WARMUP):
        _ = post_json(API_URL, PAYLOAD)

    times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        res = post_json(API_URL, PAYLOAD)
        t1 = time.perf_counter()

        dt = t1 - t0
        times.append(dt)

        # 返却に answer/text があるはずなので軽く確認
        answer = res.get("answer") or res.get("text") or ""
        print(f"[{i+1}/{RUNS}] {dt:.3f}s | answer_len={len(str(answer))}")

    avg = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    print("\n--- Summary ---")
    print(f"runs={RUNS}, warmup={WARMUP}")
    print(f"avg={avg:.3f}s, p50={p50:.3f}s, p95≈{p95:.3f}s")
    print("raw:", [round(t, 3) for t in times])

if __name__ == "__main__":
    main()
