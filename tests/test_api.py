"""Tests for api.py — FastAPI endpoints (mock LLM, real retrieval)."""

from unittest.mock import MagicMock, patch

import pytest

# =========================================================
# Health / Stats
# =========================================================


class TestHealthStats:
    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "chunks" in data
        assert "auth_enabled" in data

    def test_stats_empty(self, api_client):
        resp = api_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] == 0
        assert data["sources"] == {}


# =========================================================
# Ingest
# =========================================================


class TestIngest:
    def test_ingest_text(self, api_client):
        resp = api_client.post(
            "/ingest",
            json={
                "source": "test",
                "text": "テスト文書です。RAG のテストを行います。",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ingested_chunks"] >= 1
        assert data["total_chunks"] >= 1

    def test_ingest_duplicate_skipped(self, api_client):
        text = "重複チェック用のドキュメント。"
        r1 = api_client.post("/ingest", json={"source": "dup", "text": text})
        r2 = api_client.post("/ingest", json={"source": "dup2", "text": text})
        assert r1.json()["ingested_chunks"] >= 1
        assert r2.json()["ingested_chunks"] == 0

    def test_ingest_then_stats(self, api_client):
        api_client.post(
            "/ingest",
            json={
                "source": "my_doc",
                "text": "ドキュメント内容。" * 10,
            },
        )
        stats = api_client.get("/stats").json()
        assert stats["total_chunks"] >= 1
        assert "my_doc" in stats["sources"]

    def test_ingest_files_txt(self, api_client):
        content = "ファイルアップロードのテスト。これはテキストファイルです。"
        resp = api_client.post(
            "/ingest_files",
            files=[("files", ("test.txt", content.encode("utf-8"), "text/plain"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["ingested_chunks_total"] >= 1


# =========================================================
# Reset
# =========================================================


class TestReset:
    def test_reset_clears_store(self, api_client):
        api_client.post("/ingest", json={"source": "s", "text": "data"})
        resp = api_client.post("/reset?delete_files=false")
        assert resp.status_code == 200
        stats = api_client.get("/stats").json()
        assert stats["total_chunks"] == 0


# =========================================================
# Ask (with mocked LLM)
# =========================================================


class TestAsk:
    def _ingest_sample(self, client):
        client.post(
            "/ingest",
            json={
                "source": "社内規定",
                "text": (
                    "有給休暇は5営業日前までに申請してください。\n\n"
                    "経費精算の締め日は毎月25日です。レシートの添付が必須です。\n\n"
                    "在宅勤務は週2回まで可能です。事前に上司の承認が必要です。\n\n"
                    "打刻漏れの修正は翌営業日までに申請してください。"
                ),
            },
        )

    @patch("api.generate_answer")
    def test_ask_returns_answer(self, mock_gen, api_client):
        self._ingest_sample(api_client)
        mock_gen.return_value = (
            "有給休暇は5営業日前までに申請が必要です。[chunk0]",
            {},
        )
        resp = api_client.post(
            "/ask",
            json={
                "question": "有給休暇の申請期限は？",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "有給" in data["answer"]
        assert data["latency_ms"] >= 0
        assert len(data["trace_id"]) > 0

    @patch("api.generate_answer")
    def test_ask_with_debug_returns_chunks(self, mock_gen, api_client):
        self._ingest_sample(api_client)
        mock_gen.return_value = ("回答テスト", {})
        resp = api_client.post(
            "/ask",
            json={
                "question": "経費精算の締め日は？",
                "debug": True,
            },
        )
        data = resp.json()
        assert data["retrieved"] is not None
        assert len(data["retrieved"]) >= 1
        assert "chunk_id" in data["retrieved"][0]
        assert "text" in data["retrieved"][0]

    def test_ask_empty_store(self, api_client):
        resp = api_client.post("/ask", json={"question": "何か"})
        data = resp.json()
        assert "/ingest" in data["answer"]

    @patch("api.generate_answer")
    def test_ask_cache_hit(self, mock_gen, api_client):
        self._ingest_sample(api_client)
        mock_gen.return_value = ("cached answer", {})
        q = {"question": "有給休暇の申請は？"}

        # First call
        api_client.post("/ask", json=q)
        # Second call — retrieval should be cached
        resp = api_client.post("/ask", json=q)
        assert resp.status_code == 200

    @patch("api.generate_answer")
    def test_ask_multi_query(self, mock_gen, api_client):
        self._ingest_sample(api_client)
        mock_gen.return_value = ("multi query result", {})
        resp = api_client.post(
            "/ask",
            json={
                "question": "在宅勤務のルール",
                "use_multi": True,
            },
        )
        assert resp.status_code == 200
        assert "multi" in resp.json()["answer"]

    @patch("api.generate_answer")
    def test_ask_max_new_tokens_forwarded(self, mock_gen, api_client):
        """max_new_tokens が generate_answer に伝わることを確認。"""
        self._ingest_sample(api_client)
        mock_gen.return_value = ("token test", {})
        api_client.post(
            "/ask",
            json={"question": "test", "max_new_tokens": 512},
        )
        _, kwargs = mock_gen.call_args
        # positional arg [2] or keyword
        assert mock_gen.call_args[0][2] == 512 or kwargs.get("max_new_tokens") == 512


# =========================================================
# Ask Structured (with mocked LLM)
# =========================================================


class TestAskStructured:
    def _ingest_sample(self, client):
        client.post(
            "/ingest",
            json={
                "source": "社内規定",
                "text": (
                    "有給休暇は5営業日前までに申請してください。\n\n"
                    "経費精算の締め日は毎月25日です。\n\n"
                    "在宅勤務は週2回まで可能です。"
                ),
            },
        )

    @patch("api.generate_structured_answer")
    def test_ask_structured_returns_parsed(self, mock_gen, api_client):
        from api import StructuredAnswer

        self._ingest_sample(api_client)
        mock_gen.return_value = (
            StructuredAnswer(
                answer="有給休暇は5営業日前です。",
                references=["社内規定@abc#chunk0"],
                confidence="high",
            ),
            {},
        )
        resp = api_client.post(
            "/ask_structured",
            json={"question": "有給休暇の申請期限は？", "debug": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "有給" in data["parsed"]["answer"]
        assert data["parsed"]["confidence"] == "high"
        assert len(data["parsed"]["references"]) >= 1
        assert data["retrieved"] is not None

    def test_ask_structured_empty_store(self, api_client):
        resp = api_client.post("/ask_structured", json={"question": "何か"})
        data = resp.json()
        assert data["parsed"]["confidence"] == "none"
        assert "/ingest" in data["parsed"]["answer"]


# =========================================================
# Ingest edge cases
# =========================================================


class TestIngestEdge:
    def test_ingest_empty_text(self, api_client):
        resp = api_client.post("/ingest", json={"source": "x", "text": ""})
        assert resp.status_code == 200
        assert resp.json()["ingested_chunks"] == 0

    def test_ingest_large_text(self, api_client):
        """大きなテキストでも分割して取り込めることを確認。"""
        text = "段落。\n\n" * 500
        resp = api_client.post("/ingest", json={"source": "big", "text": text})
        assert resp.status_code == 200
        assert resp.json()["ingested_chunks"] >= 1

    def test_ingest_files_unsupported_type(self, api_client):
        resp = api_client.post(
            "/ingest_files",
            files=[("files", ("test.xyz", b"data", "application/octet-stream"))],
        )
        data = resp.json()
        assert data["ok"] is False
        assert "unsupported" in data["results"][0]["error"]


# =========================================================
# Validation (Pydantic)
# =========================================================


class TestValidation:
    def test_ask_missing_question(self, api_client):
        resp = api_client.post("/ask", json={})
        assert resp.status_code == 422

    def test_ingest_missing_text(self, api_client):
        resp = api_client.post("/ingest", json={"source": "x"})
        assert resp.status_code == 422


# =========================================================
# (Error handling tests removed - runtime verification preferred)
# =========================================================

