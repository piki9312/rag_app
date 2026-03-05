"""Tests for api.py internal functions — coverage boost for CI."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

# =========================================================
# reset with delete_files=True
# =========================================================


class TestResetDeleteFiles:
    def test_reset_delete_files_removes_index(self, api_client):
        """delete_files=True で index ファイルが削除されることを確認。"""
        # ingest でデータを入れる → store.save() が呼ばれ index ファイルが作成される
        api_client.post(
            "/ingest",
            json={"source": "doc", "text": "テスト文書。" * 30},
        )
        resp = api_client.post("/reset?delete_files=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        # ファイルが削除された (removed リストに faiss.index が含まれる)
        assert any("faiss.index" in r for r in data["removed"])

        # store が空になっていることを確認
        stats = api_client.get("/stats").json()
        assert stats["total_chunks"] == 0

    def test_reset_delete_files_no_index(self, api_client):
        """index ファイルが存在しない状態で delete_files=True でもエラーにならない。"""
        resp = api_client.post("/reset?delete_files=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["removed"] == []


# =========================================================
# extract_text_from_upload
# =========================================================


class TestExtractText:
    def test_cp932_encoding_fallback(self, api_client):
        """CP932 エンコードのテキストファイルが正しく読み取れることを確認。"""
        text = "経費精算の締め日は毎月25日です。"
        data = text.encode("cp932")
        resp = api_client.post(
            "/ingest_files",
            files=[("files", ("report.txt", data, "text/plain"))],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["ingested_chunks_total"] >= 1

    def test_pdf_extraction(self, api_client):
        """PDF ファイルからテキスト抽出できることを確認（pypdf mock）。"""
        import api as api_mod

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF内のテキストデータ。"
        mock_reader.pages = [mock_page]

        with patch.dict("sys.modules", {"pypdf": MagicMock()}):
            with patch("api.extract_text_from_upload") as mock_extract:
                mock_extract.return_value = "PDF内のテキストデータ。"
                resp = api_client.post(
                    "/ingest_files",
                    files=[("files", ("doc.pdf", b"%PDF-1.4 fake", "application/pdf"))],
                )
                assert resp.status_code == 200

    def test_extract_text_direct_txt(self):
        """extract_text_from_upload を直接呼び出し (txt)。"""
        from api import extract_text_from_upload

        result = extract_text_from_upload("test.txt", "Hello World".encode("utf-8"))
        assert result == "Hello World"

    def test_extract_text_direct_unsupported(self):
        """サポート外の拡張子で ValueError が発生する。"""
        from api import extract_text_from_upload

        with pytest.raises(ValueError, match="unsupported"):
            extract_text_from_upload("data.xyz", b"binary data")

    def test_extract_text_direct_cp932(self):
        """CP932 エンコードの直接テスト。"""
        from api import extract_text_from_upload

        text = "日本語テスト文書"
        result = extract_text_from_upload("file.txt", text.encode("cp932"))
        assert "日本語テスト文書" in result


# =========================================================
# _call_openai_chat (mocked OpenAI client)
# =========================================================


class TestCallOpenAIChat:
    def test_call_openai_chat_success(self):
        """_call_openai_chat が正常に動作しメタデータを返す。"""
        import api as api_mod

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_message = MagicMock()
        mock_message.content = "テスト回答"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("api.get_openai_client", return_value=mock_client):
            result = api_mod._call_openai_chat("gpt-4o-mini", "system", "user msg", 1024)

        assert result["text"] == "テスト回答"
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert result["usage"]["total_tokens"] == 150


# =========================================================
# _call_openai_structured (mocked OpenAI client)
# =========================================================


class TestCallOpenAIStructured:
    def test_call_openai_structured_success(self):
        """_call_openai_structured が正常に JSON を解析して返す。"""
        import api as api_mod

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 80
        mock_usage.completion_tokens = 40

        mock_message = MagicMock()
        mock_message.content = '{"answer": "回答", "references": ["chunk0"], "confidence": "high"}'

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("api.get_openai_client", return_value=mock_client):
            result = api_mod._call_openai_structured("gpt-4o-mini", "system", "user msg", 1024)

        assert result["parsed"]["answer"] == "回答"
        assert result["parsed"]["confidence"] == "high"
        assert result["usage"]["input_tokens"] == 80

    def test_call_openai_structured_invalid_json(self):
        """JSON パース失敗時のフォールバック。"""
        import api as api_mod

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_message = MagicMock()
        mock_message.content = "これはJSONではない plain text"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("api.get_openai_client", return_value=mock_client):
            result = api_mod._call_openai_structured("gpt-4o-mini", "system", "user msg", 1024)

        # JSON parse 失敗 → フォールバック
        assert result["parsed"]["confidence"] == "none"
        assert result["parsed"]["references"] == []


# =========================================================
# generate_answer (mocked _call_openai_chat)
# =========================================================


class TestGenerateAnswer:
    def test_generate_answer_success(self):
        """generate_answer が正常に回答を返す。"""
        from api import generate_answer

        mock_result = {
            "text": "有給休暇は5営業日前です。[chunk0]",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "estimated_cost_usd": 0.001,
            },
        }
        chunks = [{"chunk_id": "doc@abc#chunk0", "source": "社内規定", "text": "有給休暇は..."}]

        with patch("api._call_openai_chat", return_value=mock_result):
            answer, meta = generate_answer("有給休暇の申請は？", chunks, 1024)

        assert "有給" in answer
        assert meta["usage"]["input_tokens"] == 100
        assert meta["context_chars"] > 0

    def test_generate_answer_timeout(self):
        """OpenAI タイムアウト時に 504 を返す。"""
        from openai import APITimeoutError

        from api import generate_answer

        chunks = [{"chunk_id": "c0", "source": "s", "text": "text"}]

        with patch("api._call_openai_chat", side_effect=APITimeoutError(request=MagicMock())):
            with pytest.raises(HTTPException) as exc_info:
                generate_answer("q", chunks, 1024)
            assert exc_info.value.status_code == 504

    def test_generate_answer_api_error(self):
        """OpenAI API エラー時に 502 を返す。"""
        from openai import APIError

        from api import generate_answer

        chunks = [{"chunk_id": "c0", "source": "s", "text": "text"}]

        with patch(
            "api._call_openai_chat",
            side_effect=APIError(message="server error", request=MagicMock(), body=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                generate_answer("q", chunks, 1024)
            assert exc_info.value.status_code == 502


# =========================================================
# generate_structured_answer (mocked _call_openai_structured)
# =========================================================


class TestGenerateStructuredAnswer:
    def test_generate_structured_answer_success(self):
        """generate_structured_answer が StructuredAnswer を返す。"""
        from api import StructuredAnswer, generate_structured_answer

        mock_result = {
            "parsed": {
                "answer": "回答テキスト",
                "references": ["chunk0"],
                "confidence": "high",
            },
            "usage": {
                "input_tokens": 80,
                "output_tokens": 30,
                "total_tokens": 110,
                "estimated_cost_usd": 0.0005,
            },
        }
        chunks = [{"chunk_id": "chunk0", "source": "doc", "text": "内容"}]

        with patch("api._call_openai_structured", return_value=mock_result):
            parsed, meta = generate_structured_answer("質問？", chunks, 1024)

        assert isinstance(parsed, StructuredAnswer)
        assert parsed.confidence == "high"
        assert meta["usage"]["input_tokens"] == 80

    def test_generate_structured_answer_timeout(self):
        """タイムアウト時に 504 を返す。"""
        from openai import APITimeoutError

        from api import generate_structured_answer

        chunks = [{"chunk_id": "c0", "source": "s", "text": "text"}]

        with patch(
            "api._call_openai_structured",
            side_effect=APITimeoutError(request=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                generate_structured_answer("q", chunks, 1024)
            assert exc_info.value.status_code == 504

    def test_generate_structured_answer_api_error(self):
        """API エラー時に 502 を返す。"""
        from openai import APIError

        from api import generate_structured_answer

        chunks = [{"chunk_id": "c0", "source": "s", "text": "text"}]

        with patch(
            "api._call_openai_structured",
            side_effect=APIError(message="server error", request=MagicMock(), body=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                generate_structured_answer("q", chunks, 1024)
            assert exc_info.value.status_code == 502
