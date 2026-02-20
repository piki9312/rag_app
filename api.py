# api.py
"""FastAPI エントリポイント — RAG 検索・回答生成 API。"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import logging
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from openai import APIError, APITimeoutError
from pydantic import BaseModel, Field
from starlette.requests import Request

from llm_client import estimate_cost, get_openai_client, retry_with_backoff
from rag import RAGStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Key 認証
# ---------------------------------------------------------------------------
RAG_API_KEY = os.getenv("RAG_API_KEY", "").strip()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(_api_key_header)) -> str | None:
    """API キー認証。``RAG_API_KEY`` 未設定時は認証スキップ (開発用)。"""
    if not RAG_API_KEY:
        return None  # 未設定なら認証なし（PoC モード）
    if api_key != RAG_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# ---------------------------------------------------------------------------
# 標準エラーレスポンス
# ---------------------------------------------------------------------------
class ErrorDetail(BaseModel):
    code: str
    message: str
    trace_id: str = ""


class ErrorResponse(BaseModel):
    """すべてのエラーで統一されたレスポンス形式。"""
    error: ErrorDetail


app = FastAPI(
    title="RAG API (FAISS + sentence-transformers + OpenAI)",
    responses={
        401: {"model": ErrorResponse, "description": "認証エラー"},
        422: {"description": "バリデーションエラー"},
        502: {"model": ErrorResponse, "description": "OpenAI API エラー"},
        504: {"model": ErrorResponse, "description": "OpenAI タイムアウト"},
    },
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTPException を統一 JSON 形式で返す。"""
    trace_id = getattr(request.state, "trace_id", uuid.uuid4().hex[:12])
    body = ErrorResponse(
        error=ErrorDetail(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            trace_id=trace_id,
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(),
        headers=getattr(exc, "headers", None),
    )

# CORS middleware を追加（本番では origin を制限）
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = RAGStore()

ENV_PATH = Path(__file__).with_name(".env")
# OpenAI clientは「リクエスト時に作る」方が環境変数トラブルが少ない

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "events.jsonl")

RETRIEVAL_CACHE_TTL_SEC = int(os.getenv("RETRIEVAL_CACHE_TTL_SEC", "600"))
_retrieval_cache: dict[str, tuple[float, list[dict]]] = {}

# リトライポリシー（環境変数で外部化）
RETRY_MAX = int(os.getenv("RETRY_MAX_RETRIES", "3"))
RETRY_BASE = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
RETRY_MAX_DELAY = float(os.getenv("RETRY_MAX_DELAY", "10.0"))


def _norm_q(q: str) -> str:
    return " ".join(q.strip().split()).lower()


def _cache_key_retrieval(
    question: str, retrieval_k: int, use_multi: bool, index_version: str, emb_model: str
) -> str:
    s = f"{emb_model}|{index_version}|{retrieval_k}|{int(use_multi)}|{_norm_q(question)}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _get_retrieval_cache(key: str) -> tuple[bool, list[dict]]:
    hit = False
    now = time.time()
    item = _retrieval_cache.get(key)
    if item:
        ts, val = item
        if now - ts <= RETRIEVAL_CACHE_TTL_SEC:
            hit = True
            return hit, val
        else:
            _retrieval_cache.pop(key, None)
    return hit, []


def _set_retrieval_cache(key: str, val: list[dict]) -> None:
    _retrieval_cache[key] = (time.time(), val)


def log_event(event: dict) -> None:
    """
    ★ 運用ログを 1行JSON(jsonl) で追記する
    - 後から grep / Python で分析しやすい
    - ローカル運用ならこのシンプル版で十分
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    event["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


class IngestRequest(BaseModel):
    """POST /ingest のリクエストボディ。"""

    source: str = "manual_txt"
    text: str
    chunk_size: int = Field(default=900, ge=100, le=10000)
    overlap: int = Field(default=150, ge=0, le=500)


class IngestResponse(BaseModel):
    ingested_chunks: int
    total_chunks: int


class AskRequest(BaseModel):
    """POST /ask のリクエストボディ。"""

    question: str
    retrieval_k: int = Field(default=60, ge=1, le=200)
    context_k: int = Field(default=3, ge=1, le=20)
    use_multi: bool = False
    debug: bool = False
    max_new_tokens: int = Field(default=128, ge=16, le=4096)


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: Optional[float] = None


class AskResponse(BaseModel):
    """POST /ask のレスポンスボディ。"""

    answer: str
    retrieved: Optional[list[RetrievedChunk]] = None
    latency_ms: int
    trace_id: str


@app.get("/health")
def health():
    return {
        "ok": True,
        "chunks": len(store.metas),
        "model": OPENAI_MODEL,
        "auth_enabled": bool(RAG_API_KEY),
    }


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
def reset(delete_files: bool = True, _key: str | None = Depends(verify_api_key)):
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

    if hasattr(store, "doc_ids"):
        store.doc_ids = set()

    # ★ ついでに retrieval cache もクリア（安全）
    _retrieval_cache.clear()

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

    log_event(
        {
            "type": "reset",
            "delete_files": delete_files,
            "removed": removed,
            "errors": errors,
        }
    )
    return {"ok": len(errors) == 0, "removed": removed, "errors": errors}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, _key: str | None = Depends(verify_api_key)):
    n = store.add_text(req.source, req.text, chunk_size=req.chunk_size, overlap=req.overlap)
    log_event(
        {
            "type": "ingest",
            "source": req.source,
            "ingested_chunks": n,
            "total_chunks": len(store.metas),
            "chunk_size": req.chunk_size,
            "overlap": req.overlap,
        }
    )
    return IngestResponse(ingested_chunks=n, total_chunks=len(store.metas))


def extract_text_from_upload(filename: str, data: bytes) -> str:
    """Upload ファイルからテキストを抽出する。txt/md/pdf/docx 対応。"""
    suffix = Path(filename).suffix.lower()

    # --- txt / md ---
    if suffix in {".txt", ".md"}:
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


@app.post("/ingest_files")
async def ingest_files(
    files: list[UploadFile] = File(...),
    chunk_size: int = 900,
    overlap: int = 150,
    _key: str | None = Depends(verify_api_key),
) -> dict:
    """
    ★ 複数ファイルをまとめて投入する
    - files は multipart/form-data で複数指定
    - 各ファイルごとに成功/失敗を返す（途中で落とさない）
    """
    results = []
    ingested_total = 0

    for file in files:
        filename = file.filename or "uploaded"
        try:
            data = await file.read()
            text = extract_text_from_upload(filename, data)

            if not text.strip():
                results.append(
                    {
                        "ok": False,
                        "source": filename,
                        "error": "text extraction produced empty text",
                    }
                )
                continue

            n = store.add_text(source=filename, text=text, chunk_size=chunk_size, overlap=overlap)
            ingested_total += n

            results.append({"ok": True, "source": filename, "ingested_chunks": n})
        except Exception as e:
            results.append({"ok": False, "source": filename, "error": str(e)})

    log_event(
        {
            "type": "ingest_files",
            "files": [r.get("source") for r in results],
            "results": results,
            "ingested_chunks_total": ingested_total,
            "total_chunks": len(store.metas),
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
    )

    return {
        "ok": all(r["ok"] for r in results),
        "results": results,
        "ingested_chunks_total": ingested_total,
        "total_chunks": len(store.metas),
    }


@retry_with_backoff(max_retries=RETRY_MAX, base_delay=RETRY_BASE, max_delay=RETRY_MAX_DELAY)
def _call_openai_chat(
    model: str, system: str, user_msg: str, max_tokens: int
) -> dict:
    """OpenAI chat.completions を呼び出す（retry 付き）。"""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0
    return {
        "text": resp.choices[0].message.content or "",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": estimate_cost(model, input_tokens, output_tokens),
        },
    }


def generate_answer(
    question: str, retrieved_chunks: list[dict], max_new_tokens: int
) -> tuple[str, dict]:
    """検索済み chunk をコンテキストに OpenAI で回答を生成する（リトライ付き）。

    Returns:
        ``(answer_text, metadata_dict)`` のタプル。

    Raises:
        APIError: 最終的に OpenAI API 呼び出しに失敗した場合。
    """
    context_parts = []
    for ch in retrieved_chunks:
        context_parts.append(f"[{ch['chunk_id']}] ({ch['source']})\n{ch['text']}")

    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "あなたは社内文書QAです。必ずCONTEXTの内容だけで回答してください。"
        "CONTEXTに根拠がない場合は『文書内に根拠が見つかりません』とだけ答えてください。"
        "根拠に使った箇所は [chunk_id] を文中に引用してください。"
        "回答の末尾に必ず「根拠: [id], [id]」の形式で参照chunk_idを列挙してください。"
    )

    user_msg = f"CONTEXT:\n\n{context_text}\n\nQUESTION:\n{question}\n"

    try:
        result = _call_openai_chat(OPENAI_MODEL, system_prompt, user_msg, max_new_tokens)
        usage = result["usage"]
        meta: dict = {"context_chars": len(context_text), "usage": usage}
        return result["text"], meta
    except APITimeoutError as e:
        logger.error(f"OpenAI timeout: {e}")
        raise HTTPException(status_code=504, detail=f"OpenAI API timeout: {str(e)[:100]}")
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)[:100]}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _key: str | None = Depends(verify_api_key)):
    t0 = time.time()
    trace_id = uuid.uuid4().hex[:12]  # ログの相関用に短いIDを作る

    index_v = store.index_version()
    emb_model = os.getenv("EMB_MODEL", "unknown")
    cache_key = _cache_key_retrieval(
        req.question, req.retrieval_k, req.use_multi, index_v, emb_model
    )

    t_retrieval0 = time.time()
    cache_hit, retrieved_all = _get_retrieval_cache(cache_key)

    # まず広く取る（Recall）
    if not cache_hit:
        if req.use_multi:
            retrieved_all = store.search_multi(req.question, top_k=req.retrieval_k)
        else:
            retrieved_all = store.search(req.question, top_k=req.retrieval_k)
        _set_retrieval_cache(cache_key, retrieved_all)

    t_retrieval1 = time.time()
    retrieval_ms = int((t_retrieval1 - t_retrieval0) * 1000)

    retrieved_all_raw_n = len(retrieved_all)

    seen = set()
    deduped = []
    for r in retrieved_all:
        cid = r.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        deduped.append(r)

    retrieved_all = deduped
    retrieved_all_dedup_n = len(retrieved_all)

    retrieved = retrieved_all[: req.context_k]

    common = {
        "type": "ask",
        "trace_id": trace_id,
        "question": req.question,
        "context_k": req.context_k,
        "use_multi": req.use_multi,
        "retrieval_k": req.retrieval_k,
        "max_new_tokens": req.max_new_tokens,
        "retrieval_mode": "multi" if req.use_multi else "single",
        "index_version": index_v,
        "emb_model": store.emb_model_name,
        "model": OPENAI_MODEL,
        "cache_key_prefix": cache_key[:8],
        "cache_hit": cache_hit,
        "cache_ttl_sec": RETRIEVAL_CACHE_TTL_SEC,
        "retrieval_ms": retrieval_ms,
        "retrieved_count_all_raw": retrieved_all_raw_n,
        "retrieved_count_all_dedup": retrieved_all_dedup_n,
    }

    if not retrieved:
        total_ms = int((time.time() - t0) * 1000)
        log_event(
            {
                **common,
                "retrieved_count": 0,
                "retrieved_ids": [],
                "total_ms": total_ms,
                "note": "no_retrieved",
            }
        )
        return AskResponse(
            answer="まず /ingest で文書を投入してください。",
            retrieved=[] if req.debug else None,
            latency_ms=total_ms,
            trace_id=trace_id,
        )

    t_llm0 = time.time()
    try:
        answer, llm_meta = generate_answer(req.question, retrieved, req.max_new_tokens)
    except HTTPException:
        raise
    t_llm1 = time.time()
    total_ms = int((time.time() - t0) * 1000)

    log_event(
        {
            **common,
            "retrieved_count": len(retrieved),
            "retrieved_ids": [r["chunk_id"] for r in retrieved],
            "llm_ms": int((t_llm1 - t_llm0) * 1000),
            "total_ms": total_ms,
            "answer_len": len(answer),
            "usage": llm_meta.get("usage"),
        }
    )

    return AskResponse(
        answer=answer,
        retrieved=[RetrievedChunk(**r) for r in retrieved] if req.debug else None,
        latency_ms=total_ms,
        trace_id=trace_id,
    )


# =========================================================
# Structured Outputs (/ask_structured)
# =========================================================


class StructuredAnswer(BaseModel):
    """Structured Outputs で返す回答スキーマ。"""

    answer: str = Field(description="回答テキスト")
    references: list[str] = Field(description="参照した chunk_id のリスト")
    confidence: str = Field(description="high / medium / low / none")


class AskStructuredResponse(BaseModel):
    """POST /ask_structured のレスポンス。"""

    parsed: StructuredAnswer
    retrieved: Optional[list[RetrievedChunk]] = None
    latency_ms: int
    trace_id: str


@retry_with_backoff(max_retries=RETRY_MAX, base_delay=RETRY_BASE, max_delay=RETRY_MAX_DELAY)
def _call_openai_structured(
    model: str, system: str, user_msg: str, max_tokens: int
) -> dict:
    """OpenAI chat.completions を Structured Outputs で呼び出す（retry 付き）。"""
    client = get_openai_client()
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "回答テキスト"},
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "description": "参照した chunk_id のリスト",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "none"],
                "description": "信頼度",
            },
        },
        "required": ["answer", "references", "confidence"],
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user_msg
                + "\n\n要求: 以下の JSON スキーマに従って応答してください:\n"
                + json.dumps(schema, ensure_ascii=False),
            },
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    text = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # JSON parse 失敗時は手動で抽出
        parsed = {
            "answer": text,
            "references": [],
            "confidence": "none",
        }

    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0
    return {
        "parsed": parsed,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": estimate_cost(model, input_tokens, output_tokens),
        },
    }


def generate_structured_answer(
    question: str, retrieved_chunks: list[dict], max_new_tokens: int
) -> tuple[StructuredAnswer, dict]:
    """Structured Outputs で JSON 回答を生成する（リトライ付き）。

    Returns:
        ``(StructuredAnswer, metadata_dict)`` のタプル。

    Raises:
        APIError: 最終的に OpenAI API 呼び出しに失敗した場合。
    """
    context_parts = []
    for ch in retrieved_chunks:
        context_parts.append(f"[{ch['chunk_id']}] ({ch['source']})\n{ch['text']}")

    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "あなたは社内文書QAです。必ずCONTEXTの内容だけで回答してください。"
        "CONTEXTに根拠がない場合は answer を『文書内に根拠が見つかりません』とし、"
        "confidence を 'none' にしてください。"
        "使用した chunk_id を references に列挙してください。"
    )

    user_msg = f"CONTEXT:\n\n{context_text}\n\nQUESTION:\n{question}\n"

    try:
        result = _call_openai_structured(
            OPENAI_MODEL, system_prompt, user_msg, max_new_tokens
        )
        parsed_dict = result["parsed"]
        structured = StructuredAnswer(
            answer=parsed_dict.get("answer", ""),
            references=parsed_dict.get("references", []),
            confidence=parsed_dict.get("confidence", "none"),
        )
        usage = result["usage"]
        meta: dict = {"context_chars": len(context_text), "usage": usage}
        return structured, meta
    except APITimeoutError as e:
        logger.error(f"OpenAI timeout: {e}")
        raise HTTPException(status_code=504, detail=f"OpenAI API timeout: {str(e)[:100]}")
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)[:100]}")


@app.post("/ask_structured", response_model=AskStructuredResponse)
def ask_structured(req: AskRequest, _key: str | None = Depends(verify_api_key)):
    """Structured Outputs でJSON形式の回答を返す。"""
    t0 = time.time()
    trace_id = uuid.uuid4().hex[:12]

    index_v = store.index_version()
    emb_model = os.getenv("EMB_MODEL", "unknown")
    cache_key = _cache_key_retrieval(
        req.question, req.retrieval_k, req.use_multi, index_v, emb_model
    )

    cache_hit, retrieved_all = _get_retrieval_cache(cache_key)
    if not cache_hit:
        if req.use_multi:
            retrieved_all = store.search_multi(req.question, top_k=req.retrieval_k)
        else:
            retrieved_all = store.search(req.question, top_k=req.retrieval_k)
        _set_retrieval_cache(cache_key, retrieved_all)

    # deduplicate
    seen: set[str] = set()
    deduped = []
    for r in retrieved_all:
        cid = r.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        deduped.append(r)

    retrieved = deduped[: req.context_k]

    if not retrieved:
        total_ms = int((time.time() - t0) * 1000)
        return AskStructuredResponse(
            parsed=StructuredAnswer(
                answer="まず /ingest で文書を投入してください。",
                references=[],
                confidence="none",
            ),
            retrieved=[] if req.debug else None,
            latency_ms=total_ms,
            trace_id=trace_id,
        )

    try:
        parsed, llm_meta = generate_structured_answer(
            req.question, retrieved, req.max_new_tokens
        )
    except HTTPException:
        raise
    total_ms = int((time.time() - t0) * 1000)

    log_event(
        {
            "type": "ask_structured",
            "trace_id": trace_id,
            "question": req.question,
            "retrieved_count": len(retrieved),
            "confidence": parsed.confidence,
            "total_ms": total_ms,
            "usage": llm_meta.get("usage"),
        }
    )

    return AskStructuredResponse(
        parsed=parsed,
        retrieved=[RetrievedChunk(**r) for r in retrieved] if req.debug else None,
        latency_ms=total_ms,
        trace_id=trace_id,
    )
