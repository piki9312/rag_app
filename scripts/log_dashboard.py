#!/usr/bin/env python
"""ログ分析ダッシュボード — events.jsonl を集計して表示する。

使い方:
    python scripts/log_dashboard.py                  # デフォルト logs/events.jsonl
    python scripts/log_dashboard.py --log path/to.jsonl
    python scripts/log_dashboard.py --days 7         # 直近7日間のみ
    python scripts/log_dashboard.py --json            # JSON形式で出力
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median


def load_events(path: str, since: datetime | None = None) -> list[dict]:
    """events.jsonl を読み込む。"""
    events: list[dict] = []
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] ログファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] line {line_no} parse error, skipping", file=sys.stderr)
                continue

            if since and "ts" in ev:
                ts = datetime.fromisoformat(ev["ts"])
                if ts < since:
                    continue
            events.append(ev)
    return events


def analyze(events: list[dict]) -> dict:
    """イベントリストを集計し、サマリ辞書を返す。"""
    total = len(events)
    type_counts = Counter(ev.get("type", "unknown") for ev in events)

    # Ask系イベントのみ抽出
    ask_events = [ev for ev in events if ev.get("type") in ("ask", "ask_structured")]
    ask_total = len(ask_events)

    # レイテンシ
    latencies = [ev["total_ms"] for ev in ask_events if "total_ms" in ev]
    retrieval_latencies = [ev["retrieval_ms"] for ev in ask_events if "retrieval_ms" in ev]
    llm_latencies = [ev["llm_ms"] for ev in ask_events if "llm_ms" in ev]

    # キャッシュヒット
    cache_hits = sum(1 for ev in ask_events if ev.get("cache_hit"))
    cache_rate = (cache_hits / ask_total * 100) if ask_total else 0.0

    # no_retrieved (検索結果ゼロ)
    no_retrieved = sum(1 for ev in ask_events if ev.get("note") == "no_retrieved")

    # トークン使用量
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    cost_available = False
    for ev in ask_events:
        usage = ev.get("usage")
        if not usage:
            continue
        total_input_tokens += usage.get("input_tokens", 0) or 0
        total_output_tokens += usage.get("output_tokens", 0) or 0
        cost = usage.get("estimated_cost_usd")
        if cost is not None:
            total_cost += cost
            cost_available = True

    # Confidence 分布 (ask_structured)
    structured_events = [ev for ev in events if ev.get("type") == "ask_structured"]
    confidence_dist = Counter(ev.get("confidence", "unknown") for ev in structured_events)

    # 日別クエリ数
    daily: dict[str, int] = defaultdict(int)
    for ev in ask_events:
        ts = ev.get("ts", "")
        if ts:
            day = ts[:10]
            daily[day] += 1

    # Ingest 統計
    ingest_events = [ev for ev in events if ev.get("type") in ("ingest", "ingest_files")]
    ingested_chunks_total = sum(ev.get("ingested_chunks", 0) for ev in ingest_events)

    return {
        "period": {
            "total_events": total,
            "first_ts": events[0].get("ts", "") if events else "",
            "last_ts": events[-1].get("ts", "") if events else "",
        },
        "event_types": dict(type_counts),
        "ask_summary": {
            "total_queries": ask_total,
            "no_retrieved": no_retrieved,
            "cache_hit_rate_pct": round(cache_rate, 1),
            "latency_ms": {
                "avg": round(mean(latencies), 1) if latencies else None,
                "median": round(median(latencies), 1) if latencies else None,
                "p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
                "max": max(latencies) if latencies else None,
            },
            "retrieval_ms_avg": round(mean(retrieval_latencies), 1) if retrieval_latencies else None,
            "llm_ms_avg": round(mean(llm_latencies), 1) if llm_latencies else None,
        },
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "estimated_cost_usd": round(total_cost, 6) if cost_available else None,
        },
        "confidence_distribution": dict(confidence_dist) if confidence_dist else None,
        "ingest_summary": {
            "total_ingest_events": len(ingest_events),
            "total_ingested_chunks": ingested_chunks_total,
        },
        "daily_queries": dict(sorted(daily.items())),
    }


def print_report(summary: dict) -> None:
    """人に読みやすい形でサマリを表示する。"""
    sep = "=" * 60

    print(sep)
    print("  RAG API — ログ分析ダッシュボード")
    print(sep)

    p = summary["period"]
    print(f"\n  期間: {p['first_ts'][:19]} 〜 {p['last_ts'][:19]}")
    print(f"  総イベント数: {p['total_events']}")

    print(f"\n--- イベント種別 ---")
    for t, c in sorted(summary["event_types"].items(), key=lambda x: -x[1]):
        print(f"  {t:20s} : {c}")

    ask = summary["ask_summary"]
    print(f"\n--- 質問 (ask / ask_structured) ---")
    print(f"  質問数           : {ask['total_queries']}")
    print(f"  検索結果ゼロ     : {ask['no_retrieved']}")
    print(f"  キャッシュHit率  : {ask['cache_hit_rate_pct']}%")

    lat = ask["latency_ms"]
    if lat["avg"] is not None:
        print(f"  レイテンシ (ms)")
        print(f"    avg={lat['avg']}, median={lat['median']}, p95={lat['p95']}, max={lat['max']}")
    if ask["retrieval_ms_avg"] is not None:
        print(f"    検索: avg={ask['retrieval_ms_avg']} ms")
    if ask["llm_ms_avg"] is not None:
        print(f"    LLM : avg={ask['llm_ms_avg']} ms")

    tok = summary["token_usage"]
    print(f"\n--- トークン使用量 ---")
    print(f"  input_tokens   : {tok['total_input_tokens']:,}")
    print(f"  output_tokens  : {tok['total_output_tokens']:,}")
    print(f"  total_tokens   : {tok['total_tokens']:,}")
    if tok["estimated_cost_usd"] is not None:
        print(f"  推定コスト     : ${tok['estimated_cost_usd']:.6f}")

    if summary.get("confidence_distribution"):
        print(f"\n--- Confidence 分布 (structured) ---")
        for c, n in sorted(summary["confidence_distribution"].items()):
            print(f"  {c:10s} : {n}")

    ing = summary["ingest_summary"]
    print(f"\n--- Ingest ---")
    print(f"  ingest回数      : {ing['total_ingest_events']}")
    print(f"  投入チャンク合計: {ing['total_ingested_chunks']}")

    if summary["daily_queries"]:
        print(f"\n--- 日別クエリ数 ---")
        for day, cnt in summary["daily_queries"].items():
            bar = "█" * min(cnt, 50)
            print(f"  {day} : {cnt:4d} {bar}")

    print(f"\n{sep}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG API ログ分析ダッシュボード")
    parser.add_argument("--log", default="logs/events.jsonl", help="ログファイルパス")
    parser.add_argument("--days", type=int, default=0, help="直近 N 日間のみ (0=全期間)")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    args = parser.parse_args()

    since = None
    if args.days > 0:
        since = datetime.now(timezone.utc) - timedelta(days=args.days)

    events = load_events(args.log, since=since)
    if not events:
        print("イベントが見つかりません。", file=sys.stderr)
        sys.exit(0)

    summary = analyze(events)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_report(summary)


if __name__ == "__main__":
    main()
