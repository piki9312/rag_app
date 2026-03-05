"""Tests for scripts/log_dashboard.py — ログ分析ダッシュボード。"""

import json
import os
import sys

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _write_events(path: str, events: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


@pytest.fixture()
def sample_log(tmp_path):
    """サンプルログファイルを生成。"""
    events = [
        {
            "type": "ingest",
            "ts": "2026-02-18T10:00:00+00:00",
            "source": "doc1",
            "ingested_chunks": 5,
            "total_chunks": 5,
        },
        {
            "type": "ask",
            "ts": "2026-02-18T10:05:00+00:00",
            "trace_id": "aaa111",
            "question": "テスト質問1",
            "total_ms": 1200,
            "retrieval_ms": 50,
            "llm_ms": 1100,
            "cache_hit": False,
            "retrieved_count": 3,
            "usage": {
                "input_tokens": 200,
                "output_tokens": 50,
                "total_tokens": 250,
                "estimated_cost_usd": 0.00006,
            },
        },
        {
            "type": "ask",
            "ts": "2026-02-18T10:06:00+00:00",
            "trace_id": "bbb222",
            "question": "テスト質問2",
            "total_ms": 800,
            "retrieval_ms": 30,
            "llm_ms": 700,
            "cache_hit": True,
            "retrieved_count": 3,
            "usage": {
                "input_tokens": 180,
                "output_tokens": 40,
                "total_tokens": 220,
                "estimated_cost_usd": 0.00005,
            },
        },
        {
            "type": "ask_structured",
            "ts": "2026-02-18T10:07:00+00:00",
            "trace_id": "ccc333",
            "question": "構造化テスト",
            "total_ms": 1500,
            "confidence": "high",
            "usage": {
                "input_tokens": 300,
                "output_tokens": 80,
                "total_tokens": 380,
                "estimated_cost_usd": 0.0001,
            },
        },
        {
            "type": "ask",
            "ts": "2026-02-18T10:08:00+00:00",
            "trace_id": "ddd444",
            "question": "空結果テスト",
            "total_ms": 100,
            "note": "no_retrieved",
            "retrieved_count": 0,
            "cache_hit": False,
        },
        {
            "type": "reset",
            "ts": "2026-02-18T11:00:00+00:00",
            "delete_files": True,
        },
    ]
    path = str(tmp_path / "events.jsonl")
    _write_events(path, events)
    return path


class TestLogDashboard:
    def test_load_events(self, sample_log):
        from log_dashboard import load_events

        events = load_events(sample_log)
        assert len(events) == 6

    def test_load_events_with_since(self, sample_log):
        from datetime import datetime, timezone

        from log_dashboard import load_events

        since = datetime(2026, 2, 18, 10, 7, 0, tzinfo=timezone.utc)
        events = load_events(sample_log, since=since)
        # 10:07 以降: ask_structured, ask (no_retrieved), reset = 3件
        assert len(events) == 3

    def test_analyze_summary(self, sample_log):
        from log_dashboard import analyze, load_events

        events = load_events(sample_log)
        summary = analyze(events)

        assert summary["period"]["total_events"] == 6
        assert summary["event_types"]["ask"] == 3
        assert summary["event_types"]["ask_structured"] == 1

        ask = summary["ask_summary"]
        assert ask["total_queries"] == 4  # 3 ask + 1 ask_structured
        assert ask["no_retrieved"] == 1
        assert ask["cache_hit_rate_pct"] == 25.0  # 1/4

        latencies = ask["latency_ms"]
        assert latencies["avg"] is not None
        assert latencies["avg"] > 0

        tok = summary["token_usage"]
        assert tok["total_input_tokens"] == 680  # 200+180+300
        assert tok["total_output_tokens"] == 170  # 50+40+80
        assert tok["estimated_cost_usd"] is not None
        assert tok["estimated_cost_usd"] > 0

        assert summary["confidence_distribution"]["high"] == 1

        ing = summary["ingest_summary"]
        assert ing["total_ingest_events"] == 1
        assert ing["total_ingested_chunks"] == 5

    def test_analyze_empty(self):
        from log_dashboard import analyze

        summary = analyze([])
        assert summary["period"]["total_events"] == 0
        assert summary["ask_summary"]["total_queries"] == 0

    def test_print_report_no_error(self, sample_log, capsys):
        """print_report が例外なく実行されること。"""
        from log_dashboard import analyze, load_events, print_report

        events = load_events(sample_log)
        summary = analyze(events)
        print_report(summary)

        captured = capsys.readouterr()
        assert "ログ分析ダッシュボード" in captured.out
        assert "質問数" in captured.out
