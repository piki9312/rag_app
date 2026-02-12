"""E2E test — ingest → ask with real OpenAI API.

Skipped when OPENAI_API_KEY is not set.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture()
def live_client(tmp_path, monkeypatch):
    """TestClient with real OpenAI but isolated storage."""
    import api as api_mod
    import rag as rag_mod

    monkeypatch.setattr(rag_mod, "INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setattr(rag_mod, "INDEX_PATH", str(tmp_path / "index" / "faiss.index"))
    monkeypatch.setattr(rag_mod, "META_PATH", str(tmp_path / "index" / "meta.json"))
    monkeypatch.setattr(api_mod, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(api_mod, "LOG_PATH", str(tmp_path / "logs" / "events.jsonl"))

    api_mod.store.index = None
    api_mod.store.metas = []
    api_mod.store.doc_ids = set()
    api_mod.store.next_id = 0
    api_mod._retrieval_cache.clear()

    from fastapi.testclient import TestClient

    return TestClient(api_mod.app)


SAMPLE_DOC = (
    "有給休暇は5営業日前までに申請してください。承認者は直属の上司です。\n\n"
    "経費精算の締め日は毎月25日です。レシートの添付が必須です。\n\n"
    "在宅勤務は週2回まで可能です。事前に上司の承認が必要です。\n\n"
    "打刻漏れの修正は翌営業日までに申請してください。\n\n"
    "遅刻した場合は、始業前にSlackで上司に連絡してください。"
)


class TestE2EOpenAI:
    def test_ingest_and_ask(self, live_client):
        # Ingest
        r = live_client.post("/ingest", json={"source": "社内規定", "text": SAMPLE_DOC})
        assert r.status_code == 200
        assert r.json()["ingested_chunks"] >= 1

        # Ask
        r = live_client.post(
            "/ask",
            json={
                "question": "有給休暇はいつまでに申請する必要がありますか？",
                "debug": True,
                "max_new_tokens": 200,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "5" in data["answer"] or "営業日" in data["answer"]
        assert data["latency_ms"] > 0
        assert data["retrieved"] is not None
        assert len(data["retrieved"]) >= 1

    def test_no_hallucination(self, live_client):
        """Answer should refuse when context has no relevant info."""
        live_client.post(
            "/ingest",
            json={
                "source": "rules",
                "text": "社内カフェテリアの営業時間は11:30〜13:30です。",
            },
        )
        r = live_client.post(
            "/ask",
            json={
                "question": "退職金の計算方法を教えてください。",
                "max_new_tokens": 200,
            },
        )
        data = r.json()
        # Should not fabricate an answer about 退職金
        answer_lower = data["answer"]
        assert "根拠が見つかりません" in answer_lower or "文書内" in answer_lower
