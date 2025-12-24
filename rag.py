# rag.py
import os
import json
import re
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from pathlib import Path
import hashlib

INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def ensure_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    改善ポイント：意味のまとまり（段落）を壊しにくい分割にする
    → retrievalのRecallが上がりやすい
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    # 段落単位（空行）で分割してから積む
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""

    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= chunk_size:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    # overlap（簡易）：前チャンク末尾を次チャンクの先頭に付与
    if overlap > 0 and len(chunks) >= 2:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1]
            prefix = prev[-overlap:]
            out.append((prefix + "\n" + chunks[i]).strip())
        chunks = out

    # 末尾が短すぎる場合は前に吸収
    if len(chunks) >= 2 and len(chunks[-1]) < chunk_size * 0.25:
        chunks[-2] = (chunks[-2] + "\n" + chunks[-1]).strip()
        chunks.pop()

    return [c for c in chunks if c]

def _file_fp(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    st = p.stat()
    return f"{int(st.st_mtime)}:{st.st_size}"

def compute_index_version() -> str:
    # まずは mtime+size で十分（必要になったら内容hashへ拡張）
    return f"faiss={_file_fp(INDEX_PATH)}|meta={_file_fp(META_PATH)}"

class RAGStore:
    def __init__(self):
        ensure_dir()
        self.emb_model_name = EMB_MODEL
        self.model = SentenceTransformer(self.emb_model_name)
        self.index: Optional[faiss.Index] = None
        self.metas: List[Dict[str, Any]] = []
        self.doc_ids: set[str] = set()
        self.next_id = 0
        self._load()

    def index_version(self) -> str:
        return compute_index_version()
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        return vecs

    def _init_index(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add_text(self, source: str, text: str, chunk_size: int = 900, overlap: int = 150) -> int:
        # 1) doc単位の重複チェック（最初にやる）
        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if hasattr(self, "doc_ids") and doc_hash in self.doc_ids:
            return 0  # 既に同じ文書が入ってるのでスキップ

        # 2) chunk化
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            return 0

        # 3) embed → indexに追加
        vecs = self._embed(chunks)
        if self.index is None:
            self._init_index(vecs.shape[1])

        self.index.add(vecs)

        # 4) doc_idsに登録（index追加が成功した後）
        if not hasattr(self, "doc_ids"):
            self.doc_ids = set()
        self.doc_ids.add(doc_hash)

        # 5) metas登録
        for i, ch in enumerate(chunks):
            chunk_id = f"{source}@{doc_hash}#chunk{i}"
            self.metas.append({
                "doc_id": doc_hash,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "source": source,
                "text": ch
            })
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
        scs = scores[0].tolist()

        # 改善ポイント：scoreも返す（Multi-Queryで統合に必要）
        results = []
        for pos, sc in zip(idxs, scs):
            if pos < 0:
                continue
            meta = dict(self.metas[pos])
            meta["score"] = float(sc)
            results.append(meta)
        return results

    def _keywordize(self, q: str) -> str:
        q2 = re.sub(r"[^\wぁ-んァ-ヶ一-龥]+", " ", q)
        toks = [t for t in q2.split() if len(t) >= 2]
        return " ".join(toks[:12])

    def search_multi(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        改善ポイント④：Multi-Query Retrieval（Recall改善）
        - 元質問 + キーワード抽出版 で検索して結果をマージ
        """
        queries = [query, self._keywordize(query)]
        pool: List[Dict[str, Any]] = []
        for q in queries:
            pool.extend(self.search(q, top_k=min(10, len(self.metas) or 10)))

        best: Dict[str, Dict[str, Any]] = {}
        for r in pool:
            cid = r["chunk_id"]
            if cid not in best or r.get("score", -1e9) > best[cid].get("score", -1e9):
                best[cid] = r

        merged = sorted(best.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:top_k]

    def save(self):
        ensure_dir()
        if self.index is not None:
            faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"metas": self.metas, "next_id": self.next_id, "emb_model": self.emb_model_name, "doc_ids": sorted(list(self.doc_ids))},
                f,
                ensure_ascii=False,
            )

    def _load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.metas = d.get("metas", [])
                self.next_id = d.get("next_id", len(self.metas))
                self.doc_ids = set(d.get("doc_ids", []))

                # 改善ポイント：indexとmetasの整合性チェック（ズレると致命傷）
                if self.index is not None and self.index.ntotal != len(self.metas):
                    self.index = None
                    self.metas = []
                    self.next_id = 0

            except Exception:
                self.index = None
                self.metas = []
                self.next_id = 0
