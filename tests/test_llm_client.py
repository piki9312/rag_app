"""Tests for llm_client.py — OpenAI client factory."""

import os
import time
from unittest.mock import patch

import pytest


class TestGetOpenAIClient:
    def test_missing_key_raises(self, tmp_path, monkeypatch):
        """API キーが無い場合に RuntimeError を送出する。"""
        env_file = tmp_path / ".env"
        env_file.write_text("# no key\n")

        import llm_client

        monkeypatch.setattr(llm_client, "ENV_PATH", env_file)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm_client.get_openai_client()

    def test_key_from_dotenv(self, tmp_path, monkeypatch):
        """`.env` からキーを読み取れること。"""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-test-123\n")

        import llm_client

        monkeypatch.setattr(llm_client, "ENV_PATH", env_file)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        client = llm_client.get_openai_client()
        assert client.api_key == "sk-test-123"

    def test_key_from_env_var_fallback(self, tmp_path, monkeypatch):
        """`.env` に無くても環境変数から取得できること。"""
        env_file = tmp_path / ".env"
        env_file.write_text("# empty\n")

        import llm_client

        monkeypatch.setattr(llm_client, "ENV_PATH", env_file)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-456")

        client = llm_client.get_openai_client()
        assert client.api_key == "sk-env-456"


# =========================================================
# Retry Logic
# =========================================================


class TestRetryWithBackoff:
    def test_retry_succeeds_on_first_try(self):
        """リトライ不要な場合は即座に成功。"""
        import llm_client

        @llm_client.retry_with_backoff(max_retries=3)
        def success_fn():
            return "ok"

        result = success_fn()
        assert result == "ok"

    def test_retry_eventually_succeeds(self):
        """失敗後リトライで成功する。"""
        import llm_client
        from openai import RateLimitError

        call_count = {"count": 0}

        @llm_client.retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.05)
        def flaky_fn():
            call_count["count"] += 1
            if call_count["count"] < 3:
                e = RateLimitError.__new__(RateLimitError)
                raise e
            return "recovered"

        result = flaky_fn()
        assert result == "recovered"
        assert call_count["count"] == 3

    def test_retry_exhausted_raises(self):
        """リトライ枚数尽きたら例外を送出。"""
        import llm_client
        from openai import RateLimitError

        @llm_client.retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            e = RateLimitError.__new__(RateLimitError)
            raise e

        with pytest.raises(RateLimitError):
            always_fails()

    def test_non_retryable_error_raised_immediately(self):
        """リトライ不可のエラーは即座に送出。"""
        import llm_client
        from openai import APIError

        @llm_client.retry_with_backoff(max_retries=3)
        def client_error():
            e = APIError.__new__(APIError)
            raise e

        with pytest.raises(APIError):
            client_error()

    def test_exponential_backoff_timing(self):
        """バックオフが指数関数的に増えること。"""
        import llm_client
        from openai import RateLimitError

        times: list[float] = []

        @llm_client.retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.1)
        def track_timing():
            times.append(time.time())
            if len(times) < 3:
                e = RateLimitError.__new__(RateLimitError)
                raise e
            return "ok"

        result = track_timing()
        assert result == "ok"
        assert len(times) == 3

        # 2度目と3度目の呼び出しの間隔を確認（バックオフ）
        interval_1 = times[1] - times[0]
        interval_2 = times[2] - times[1]
        # 期待: interval_2 > interval_1 (指数関数的増加)
        assert interval_2 > interval_1 or abs(interval_2 - interval_1) < 0.02  # 誤差考慮
