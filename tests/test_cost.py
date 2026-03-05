"""Tests for cost estimation and token usage logging."""

import pytest

from llm_client import estimate_cost


class TestEstimateCost:
    def test_gpt4o_mini(self):
        """gpt-4o-mini の単価で計算されること。"""
        cost = estimate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert cost is not None
        # input: 1000/1000 * 0.00015 = 0.00015
        # output: 500/1000 * 0.0006 = 0.0003
        expected = 0.00015 + 0.0003
        assert abs(cost - expected) < 1e-8

    def test_gpt4o(self):
        cost = estimate_cost("gpt-4o", input_tokens=2000, output_tokens=1000)
        assert cost is not None
        expected = 2000 / 1000 * 0.0025 + 1000 / 1000 * 0.01
        assert abs(cost - expected) < 1e-8

    def test_unknown_model_returns_none(self):
        cost = estimate_cost("unknown-model", input_tokens=100, output_tokens=50)
        assert cost is None

    def test_zero_tokens(self):
        cost = estimate_cost("gpt-4o-mini", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_cost_table_has_common_models(self):
        """主要モデルが単価表に存在すること。"""
        from llm_client import _COST_PER_1K

        for model in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
            assert model in _COST_PER_1K, f"{model} not in cost table"


class TestTokenUsageInLog:
    """ask エンドポイントのログにトークン使用量が記録されること。"""

    def test_ask_log_contains_usage(self, api_client, tmp_path):
        """mocked LLM で /ask 呼び出し後にログに usage が含まれる。"""
        import json
        from unittest.mock import patch

        # Ingest sample data
        api_client.post(
            "/ingest", json={"source": "test", "text": "テスト文書。RAGのテストです。"}
        )

        usage = {
            "input_tokens": 150,
            "output_tokens": 30,
            "total_tokens": 180,
            "estimated_cost_usd": 0.0001,
        }

        with patch("api.generate_answer") as mock_gen:
            mock_gen.return_value = ("回答テスト", {"usage": usage})
            api_client.post("/ask", json={"question": "テスト質問"})

        # Read the log file
        import api as api_mod

        log_path = api_mod.LOG_PATH
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        ask_events = [ev for ev in lines if ev.get("type") == "ask"]
        assert len(ask_events) >= 1

        ev = ask_events[-1]
        assert "usage" in ev
        assert ev["usage"]["input_tokens"] == 150
        assert ev["usage"]["output_tokens"] == 30
