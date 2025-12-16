# rag.py
import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

# 日本語も混ざるなら多言語寄りが無難（必要なら後で変更）
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def ensure_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j].strip())
        if j == n:
            break
        i = max(0, j - overlap)
    return [c for c in chunks if c]


class RAGStore:
    def __init__(self):
        ensure_dir()
        self.model = SentenceTransformer(EMB_MODEL)
        self.index: Optional[faiss.Index] = None
        self.metas: List[Dict[str, Any]] = []
        self.next_id = 0
        self._load()

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine相当で扱いやすい
        ).astype(np.float32)
        return vecs

    def _init_index(self, dim: int):
        # normalize_embeddings=True + IndexFlatIP => cosine類似度の近傍検索っぽくなる
        self.index = faiss.IndexFlatIP(dim)

    def add_text(self, source: str, text: str, chunk_size: int = 900, overlap: int = 150) -> int:
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            return 0

        vecs = self._embed(chunks)
        if self.index is None:
            self._init_index(vecs.shape[1])

        # IndexFlatIPはIDを保持しないので、メタ配列の順番＝検索で返るposとして扱う
        self.index.add(vecs)

        for ch in chunks:
            self.metas.append({"chunk_id": self.next_id, "source": source, "text": ch})
            self.next_id += 1

        self.save()
        return len(chunks)

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        if self.index is None or not self.metas:
            return []
        qvec = self._embed([query])
        k = min(top_k, len(self.metas))
        scores, idxs = self.index.search(qvec, k)
        idxs = idxs[0].tolist()

        results = []
        for pos in idxs:
            results.append(self.metas[pos])
        return results

    def save(self):
        ensure_dir()
        if self.index is not None:
            faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"metas": self.metas, "next_id": self.next_id, "emb_model": EMB_MODEL}, f, ensure_ascii=False)

    def _load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.metas = d.get("metas", [])
                self.next_id = d.get("next_id", len(self.metas))
            except Exception:
                self.index = None
                self.metas = []
                self.next_id = 0
