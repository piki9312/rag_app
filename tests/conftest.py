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


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    """FastAPI TestClient with isolated index/log dirs."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    # FAISS C++ cannot handle non-ASCII paths (e.g. Japanese usernames in %TEMP%)
    import tempfile

    ascii_tmp = os.path.join("C:\\tmp", "rag_test_" + os.urandom(4).hex())
    os.makedirs(ascii_tmp, exist_ok=True)

    idx_dir = os.path.join(ascii_tmp, "index")
    log_dir = os.path.join(ascii_tmp, "logs")
    os.makedirs(idx_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Isolate file I/O
    import api as api_mod
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
    yield client

    # Cleanup
    import shutil

    shutil.rmtree(ascii_tmp, ignore_errors=True)
