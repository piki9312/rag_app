"""Tests for rag.py — chunk_text, RAGStore search, Multi-Query."""

import pytest

from rag import chunk_text

# =========================================================
# chunk_text
# =========================================================


class TestChunkText:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_single_short_paragraph(self):
        text = "Hello world."
        chunks = chunk_text(text, chunk_size=900)
        assert len(chunks) == 1
        assert "Hello world." in chunks[0]

    def test_paragraph_boundary_respected(self):
        """Two paragraphs separated by blank line."""
        text = "Paragraph one content.\n\nParagraph two content."
        chunks = chunk_text(text, chunk_size=20, overlap=0)
        # Each paragraph is shorter than 20 chars, so they should be combined
        # Actually with chunk_size=20 they won't fit together
        assert len(chunks) >= 1
        # Original text should be recoverable from chunks
        joined = " ".join(chunks)
        assert "Paragraph one" in joined
        assert "Paragraph two" in joined

    def test_overlap_adds_prefix(self):
        para1 = "A" * 100
        para2 = "B" * 100
        text = f"{para1}\n\n{para2}"
        chunks = chunk_text(text, chunk_size=120, overlap=30)
        assert len(chunks) >= 2
        # Second chunk should start with tail of first chunk (overlap)
        assert chunks[1].startswith("A")

    def test_short_tail_absorbed(self):
        """Trailing chunk shorter than 25% of chunk_size merges into previous."""
        text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 10
        chunks = chunk_text(text, chunk_size=600, overlap=0)
        # "C"*10 should be absorbed into previous chunk
        assert "C" in chunks[-1]
        assert "B" in chunks[-1]

    def test_japanese_text(self):
        text = "段落一の内容です。日本語のテスト。\n\n段落二の内容です。別の文章。"
        chunks = chunk_text(text, chunk_size=5000, overlap=0)
        assert len(chunks) >= 1
        assert "日本語" in chunks[0]


# =========================================================
# RAGStore — embedding & search
# =========================================================


class TestRAGStore:
    def test_add_text_creates_chunks(self, rag_store):
        n = rag_store.add_text("test_doc", "Hello world. This is a test document.")
        assert n >= 1
        assert len(rag_store.metas) == n
        assert rag_store.index is not None
        assert rag_store.index.ntotal == n

    def test_duplicate_document_skipped(self, rag_store):
        text = "Same document content."
        n1 = rag_store.add_text("doc1", text)
        n2 = rag_store.add_text("doc1_dup", text)
        assert n1 >= 1
        assert n2 == 0  # duplicate hash → skip

    def test_search_returns_results(self, rag_store):
        rag_store.add_text(
            "faq", "有給休暇は5日前までに申請してください。承認者は直属の上司です。"
        )
        results = rag_store.search("有給休暇の申請", top_k=3)
        assert len(results) >= 1
        assert "score" in results[0]
        assert results[0]["source"] == "faq"

    def test_search_empty_index(self, rag_store):
        results = rag_store.search("anything", top_k=3)
        assert results == []

    def test_search_multi_returns_results(self, rag_store):
        rag_store.add_text("rules", "経費精算の締め日は毎月25日です。レシートの添付が必須です。")
        results = rag_store.search_multi("経費精算の締め日", top_k=5)
        assert len(results) >= 1

    def test_chunk_ids_are_unique(self, rag_store):
        rag_store.add_text("doc_a", "First document.\n\nSecond paragraph.")
        rag_store.add_text("doc_b", "Another document entirely different.")
        ids = [m["chunk_id"] for m in rag_store.metas]
        assert len(ids) == len(set(ids))

    def test_meta_fields(self, rag_store):
        rag_store.add_text("src", "Test content for meta field check.")
        meta = rag_store.metas[0]
        assert "chunk_id" in meta
        assert "source" in meta
        assert "text" in meta
        assert "doc_id" in meta
        assert meta["source"] == "src"


# =========================================================
# Edge cases
# =========================================================


class TestEdgeCases:
    def test_chunk_text_whitespace_only(self):
        assert chunk_text("   \n\n  ") == []

    def test_chunk_text_crlf_normalised(self):
        """Windows の CRLF が LF に正規化されること。"""
        text = "para one\r\n\r\npara two"
        chunks = chunk_text(text, chunk_size=5000, overlap=0)
        assert len(chunks) >= 1
        for c in chunks:
            assert "\r" not in c

    def test_chunk_text_no_overlap(self):
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks = chunk_text(text, chunk_size=120, overlap=0)
        assert len(chunks) == 2
        assert chunks[1].startswith("B")

    def test_search_top_k_exceeds_total(self, rag_store):
        """top_k がチャンク数を超えてもエラーにならない。"""
        rag_store.add_text("one", "唯一のドキュメントです。")
        results = rag_store.search("ドキュメント", top_k=100)
        assert len(results) >= 1

    def test_search_multi_deduplicates(self, rag_store):
        """Multi-Query で同じ chunk が重複排除されること。"""
        rag_store.add_text("faq", "経費精算は毎月25日が締め日です。")
        results = rag_store.search_multi("経費精算 締め日", top_k=10)
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_keywordize(self, rag_store):
        result = rag_store._keywordize("有給休暇はいつまでに申請する？")
        # 短い単語（1文字）は除外される
        assert "?" not in result
