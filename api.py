# api.py
import os
import json
import datetime
from collections import Counter
import time
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import io

from llm_client import get_openai_client
from rag import RAGStore

app = FastAPI(title="RAG API (FAISS + sentence-transformers + OpenAI)")

store = RAGStore()

ENV_PATH = Path(__file__).with_name(".env")
# OpenAI clientは「リクエスト時に作る」方が環境変数トラブルが少ない

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "events.jsonl")


def log_event(event: dict) -> None:
    """
    ★ 運用ログを 1行JSON(jsonl) で追記する
    - 後から grep / Python で分析しやすい
    - ローカル運用ならこのシンプル版で十分
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    event["ts"] = datetime.datetime.utcnow().isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


class IngestRequest(BaseModel):
    source: str = "manual_txt"
    text: str
    chunk_size: int = 900
    overlap: int = 150

class IngestResponse(BaseModel):
    ingested_chunks: int
    total_chunks: int

class AskRequest(BaseModel):
    question: str
    retrieval_k: int = 30
    context_k: int = 6
    use_multi: bool = True
    debug: bool = True

class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    retrieved: Optional[List[RetrievedChunk]] = None
    latency_ms: int


@app.get("/health")
def health():
    return {"ok": True, "chunks": len(store.metas), "model": OPENAI_MODEL}

@app.get("/stats")
def stats():
    """
    ★ 現在の投入状況を返す（運用でめちゃ便利）
    - 何チャンクある？
    - source別にどれだけ入ってる？
    """
    by_source = Counter(m["source"] for m in store.metas)  # metasの中身を集計
    return {
        "ok": True,
        "total_chunks": len(store.metas),
        "sources": dict(by_source),
        "emb_model": os.getenv("EMB_MODEL", "unknown"),
        "openai_model": OPENAI_MODEL,
    }

@app.post("/reset")
def reset(delete_files: bool = True):
    """
    ★ 全消しして作り直す（開発/運用で必須）
    delete_files=True:
      - index/faiss.index と index/meta.json を消す
      - メモリ上のstoreも初期化
    """
    removed = []
    errors = []

    # 1) メモリ上の状態を初期化
    store.index = None
    store.metas = []
    store.next_id = 0

    # 2) 永続化ファイルを削除（任意）
    if delete_files:
        # rag.py と同じ場所を削除する（indexディレクトリ）
        paths = [
            os.path.join("index", "faiss.index"),
            os.path.join("index", "meta.json"),
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    removed.append(p)
                except Exception as e:
                    errors.append(f"{p}: {e}")

    log_event({"type": "reset", "delete_files": delete_files, "removed": removed, "errors": errors})
    return {"ok": len(errors) == 0, "removed": removed, "errors": errors}

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    n = store.add_text(req.source, req.text, chunk_size=req.chunk_size, overlap=req.overlap)
    log_event({
        "type": "ingest",
        "source": req.source,
        "ingested_chunks": n,
        "total_chunks": len(store.metas),
        "chunk_size": req.chunk_size,
        "overlap": req.overlap,
    })
    return IngestResponse(ingested_chunks=n, total_chunks=len(store.metas))

def extract_text_from_upload(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    # --- txt / md ---
    if suffix in [".txt", ".md"]:
        # ★ encodingが怪しいファイルもあるので errors="ignore" で落ちないようにする
        for enc in ("utf-8", "utf-8-sig", "cp932"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                pass
        return data.decode("utf-8", errors="ignore")

    # --- pdf ---
    if suffix == ".pdf":
        # pip install pypdf
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        # ★ PDFはページごとに抽出。extract_text()がNoneのこともあるので or "" を入れる
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)

    # --- docx ---
    if suffix == ".docx":
        # pip install python-docx
        from docx import Document
        doc = Document(io.BytesIO(data))
        # ★ 段落テキストを結合（表が多い場合は追加処理が必要になることもある）
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"unsupported file type: {suffix}")


@app.post("/ingest_file")
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = 900,
    overlap: int = 150,
):
    """
    ★ 実務向けの投入口：ファイルをアップロードして、その中身をテキスト化→RAGに登録
    - source にはファイル名を入れる（後でどの文書由来か追える）
    - chunk_size/overlap をクエリで調整可能にしておくとチューニングが楽
    """
    filename = file.filename or "uploaded"
    data = await file.read()

    try:
        text = extract_text_from_upload(filename, data)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    # ★ 空っぽ抽出のPDF等もあるので軽くチェック
    if not text.strip():
        return {"ok": False, "error": "text extraction produced empty text (scanned PDF etc.)"}

    n = store.add_text(source=filename, text=text, chunk_size=chunk_size, overlap=overlap)
    log_event({
    "type": "ingest_file",
    "source": filename,
    "ingested_chunks": n,
    "total_chunks": len(store.metas),
    "chunk_size": chunk_size,
    "overlap": overlap,
    })
    return {"ok": True, "source": filename, "ingested_chunks": n, "total_chunks": len(store.metas)}

def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for ch in retrieved_chunks:
        context_parts.append(f"[{ch['chunk_id']}] ({ch['source']})\n{ch['text']}")

    context_text = "\n\n".join(context_parts)

    instructions = (
        "あなたは社内文書QAです。必ずCONTEXTの内容だけで回答してください。"
        "CONTEXTに根拠がない場合は『文書内に根拠が見つかりません』とだけ答えてください。"
        "根拠に使った箇所は [chunk_id] を文中に引用してください。"
        "回答の末尾に必ず「根拠: [id], [id]」の形式で参照chunk_idを列挙してください。"
    )

    client = get_openai_client()
    resp = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=f"CONTEXT:\n\n{context_text}\n\nQUESTION:\n{question}\n",
    )
    return resp.output_text

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()

    # まず広く取る（Recall）
    if req.use_multi:
        retrieved_all = store.search_multi(req.question, top_k=req.retrieval_k)
    else:
        retrieved_all = store.search(req.question, top_k=req.retrieval_k)

    # LLMに渡す分だけ絞る（ノイズ減）
    retrieved = retrieved_all[: req.context_k]

    if not retrieved:
        latency = int((time.time() - t0) * 1000)
        log_event({
            "type": "ask",
            "question": req.question,
            "retrieval_k": req.retrieval_k,
            "context_k": req.context_k,
            "use_multi": req.use_multi,
            "retrieved_count": 0,
            "retrieved_ids": [],
            "latency_ms": latency,
            "model": OPENAI_MODEL,
            "note": "no_retrieved",
        })
        return AskResponse(
            answer="まず /ingest で文書を投入してください。",
            retrieved=[] if req.debug else None,
            latency_ms=latency,
        )

    answer = generate_answer(req.question, retrieved)
    latency = int((time.time() - t0) * 1000)

    log_event({
        "type": "ask",
        "question": req.question,
        "retrieval_k": req.retrieval_k,
        "context_k": req.context_k,
        "use_multi": req.use_multi,
        "retrieved_count": len(retrieved),
        "retrieved_ids": [r["chunk_id"] for r in retrieved],
        "retrieved_sources": list({r["source"] for r in retrieved}),
        "latency_ms": latency,
        "model": OPENAI_MODEL,
        "answer_len": len(answer),
    })

    return AskResponse(
        answer=answer,
        retrieved=[RetrievedChunk(**r) for r in retrieved] if req.debug else None,
        latency_ms=latency,
    )


