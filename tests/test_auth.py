"""Tests for API key authentication."""

from unittest.mock import patch

import pytest


class TestAuthDisabled:
    """RAG_API_KEY 未設定時は認証スキップ。"""

    def test_health_no_auth(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["auth_enabled"] is False

    def test_ingest_no_auth(self, api_client):
        resp = api_client.post("/ingest", json={"source": "s", "text": "data"})
        assert resp.status_code == 200

    def test_reset_no_auth(self, api_client):
        resp = api_client.post("/reset?delete_files=false")
        assert resp.status_code == 200

    @patch("api.generate_answer")
    def test_ask_no_auth(self, mock_gen, api_client):
        api_client.post("/ingest", json={"source": "s", "text": "テスト文書。"})
        mock_gen.return_value = ("ans", {})
        resp = api_client.post("/ask", json={"question": "q"})
        assert resp.status_code == 200


class TestAuthEnabled:
    """RAG_API_KEY 設定時、未認証リクエストを拒否。"""

    AUTH_HEADERS = {"X-API-Key": "test-secret-key"}

    def test_health_remains_public(self, api_client_auth):
        """health はキー不要。"""
        resp = api_client_auth.get("/health")
        assert resp.status_code == 200
        assert resp.json()["auth_enabled"] is True

    def test_stats_remains_public(self, api_client_auth):
        resp = api_client_auth.get("/stats")
        assert resp.status_code == 200

    # --- ingest ---
    def test_ingest_rejected_without_key(self, api_client_auth):
        resp = api_client_auth.post("/ingest", json={"source": "s", "text": "d"})
        assert resp.status_code == 401
        assert "error" in resp.json()

    def test_ingest_rejected_with_wrong_key(self, api_client_auth):
        resp = api_client_auth.post(
            "/ingest",
            json={"source": "s", "text": "d"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_ingest_accepted_with_correct_key(self, api_client_auth):
        resp = api_client_auth.post(
            "/ingest",
            json={"source": "s", "text": "テスト文書。"},
            headers=self.AUTH_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["ingested_chunks"] >= 1

    # --- reset ---
    def test_reset_rejected_without_key(self, api_client_auth):
        resp = api_client_auth.post("/reset?delete_files=false")
        assert resp.status_code == 401

    def test_reset_accepted_with_key(self, api_client_auth):
        resp = api_client_auth.post(
            "/reset?delete_files=false", headers=self.AUTH_HEADERS
        )
        assert resp.status_code == 200

    # --- ask ---
    def test_ask_rejected_without_key(self, api_client_auth):
        resp = api_client_auth.post("/ask", json={"question": "q"})
        assert resp.status_code == 401

    @patch("api.generate_answer")
    def test_ask_accepted_with_key(self, mock_gen, api_client_auth):
        api_client_auth.post(
            "/ingest",
            json={"source": "s", "text": "テスト文書。"},
            headers=self.AUTH_HEADERS,
        )
        mock_gen.return_value = ("ans", {})
        resp = api_client_auth.post(
            "/ask", json={"question": "q"}, headers=self.AUTH_HEADERS
        )
        assert resp.status_code == 200

    # --- ask_structured ---
    def test_ask_structured_rejected_without_key(self, api_client_auth):
        resp = api_client_auth.post("/ask_structured", json={"question": "q"})
        assert resp.status_code == 401


class TestErrorResponseFormat:
    """エラーレスポンスが統一 JSON 形式であること。"""

    def test_401_format(self, api_client_auth):
        resp = api_client_auth.post("/ingest", json={"source": "s", "text": "d"})
        data = resp.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == "HTTP_401"

    def test_422_format(self, api_client):
        """Pydantic バリデーションエラーは FastAPI 標準形式。"""
        resp = api_client.post("/ask", json={})
        # 422 is handled by FastAPI's own handler; just confirm it's 422
        assert resp.status_code == 422
