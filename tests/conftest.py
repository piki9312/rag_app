"""Shared fixtures for RAG App tests."""

import os
import sys

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip manual test scripts during collection (they execute at import time)
collect_ignore = ["test_openai_manual.py", "test_questions_manual.py"]


@pytest.fixture()
def rag_store():
    """Create a fresh in-memory RAGStore (no disk persistence)."""
    from rag import RAGStore

    store = RAGStore.__new__(RAGStore)
    store.emb_model_name = os.getenv(
        "EMB_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    # Load model but skip loading index from disk
    from sentence_transformers import SentenceTransformer

    store.model = SentenceTransformer(store.emb_model_name)
    store.index = None
    store.metas = []
    store.doc_ids = set()
    store.next_id = 0
    return store


def _make_api_client(monkeypatch, api_key_env: str = ""):
    """TestClient を生成する共通ヘルパー。

    api_key_env が空文字なら認証無効 (PoC モード)、
    値を渡せばその API キーで認証有効になる。
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    # 認証キーを設定 (空文字 = 認証無効)
    import api as api_mod

    monkeypatch.setattr(api_mod, "RAG_API_KEY", api_key_env)

    # FAISS C++ cannot handle non-ASCII paths (e.g. Japanese usernames in %TEMP%)
    ascii_tmp = os.path.join("C:\\tmp", "rag_test_" + os.urandom(4).hex())
    os.makedirs(ascii_tmp, exist_ok=True)

    idx_dir = os.path.join(ascii_tmp, "index")
    log_dir = os.path.join(ascii_tmp, "logs")
    os.makedirs(idx_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Isolate file I/O
    import rag as rag_mod

    monkeypatch.setattr(rag_mod, "INDEX_DIR", idx_dir)
    monkeypatch.setattr(rag_mod, "INDEX_PATH", os.path.join(idx_dir, "faiss.index"))
    monkeypatch.setattr(rag_mod, "META_PATH", os.path.join(idx_dir, "meta.json"))
    monkeypatch.setattr(api_mod, "LOG_DIR", log_dir)
    monkeypatch.setattr(api_mod, "LOG_PATH", os.path.join(log_dir, "events.jsonl"))

    # Reset store state
    api_mod.store.index = None
    api_mod.store.metas = []
    api_mod.store.doc_ids = set()
    api_mod.store.next_id = 0
    api_mod._retrieval_cache.clear()

    from fastapi.testclient import TestClient

    client = TestClient(api_mod.app)
    return client, ascii_tmp


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    """FastAPI TestClient with isolated index/log dirs (auth disabled)."""
    client, ascii_tmp = _make_api_client(monkeypatch, api_key_env="")
    yield client

    import shutil

    shutil.rmtree(ascii_tmp, ignore_errors=True)


@pytest.fixture()
def api_client_auth(monkeypatch, tmp_path):
    """FastAPI TestClient with API key auth enabled.

    テスト内では ``headers={"X-API-Key": "test-secret-key"}`` を付与する。
    """
    client, ascii_tmp = _make_api_client(monkeypatch, api_key_env="test-secret-key")
    yield client

    import shutil

    shutil.rmtree(ascii_tmp, ignore_errors=True)
