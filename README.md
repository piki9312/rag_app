# RAG API (PoC)

[![CI](https://github.com/piki9312/rag_app/actions/workflows/ci.yml/badge.svg)](https://github.com/piki9312/rag_app/actions/workflows/ci.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

## 概要
本プロジェクトは、社内文書やマニュアルを対象とした
**検索型QAシステムの技術検証**として実装したRAG APIです。

本番運用を目的とせず、
- RAG設計(chunk / retrieval / prompt)
- 回答の安全性(hallucination対策)
- 運用時の確認性(ログ・統計)

を短期間で検証できる構成を重視しています。

## 特徴
- FAISS + sentence-transformersによる文書検索
- OpenAI APIを用いた回答生成
- **Structured Outputs** による構造化JSON応答 (`/ask_structured`)
- **API キー認証**（`X-API-Key` ヘッダー / 環境変数で ON/OFF）
- **統一エラーレスポンス**（全エラーが `{"error": {...}}` 形式）
- **トークン使用量・コスト記録**（リクエスト毎の推定コストをログ出力）
- **ログ分析ダッシュボード**（`scripts/log_dashboard.py`）
- **リトライポリシーの外部化**（環境変数で回数・待機時間を制御）
- 参照した文書チャンク(chunk_id)を回答内に明示
- 根拠のない情報は回答しない安全設計
- FastAPIによるREST API提供
- 検索結果キャッシュ(TTL制御)
- JSONL形式の構造化ログ
- ローカル環境で実行可能(PoC向け)

## アーキテクチャ
```
Client
  ↓  (X-API-Key 認証)
FastAPI (/ask, /ask_structured, /ingest)
  ↓
RAG Pipeline
  ├─ 文書分割（段落単位 chunk + overlap）
  ├─ embedding（sentence-transformers）
  ├─ FAISS 検索（Multi-Query対応）
  ├─ 検索キャッシュ（TTL制御）
  ↓
OpenAI API  ←─ リトライ（指数バックオフ）+ コスト記録
  ↓
回答生成（根拠付き / Structured Outputs）
  ↓
JSONL ログ（トークン数・コスト・レイテンシ）
  ↓
ログ分析ダッシュボード (scripts/log_dashboard.py)
```

## RAG設計のポイント

### チャンク分割
- 文書を**段落単位**で分割し、意味のまとまりを保持
- 文脈切れを防ぐため overlap を付与
- 極端に短いチャンクは前チャンクに統合し、ノイズを抑制

### 検索戦略
- 通常のベクトル検索に加え、質問文をキーワード化した
**Multi-Query Retrieval** を実装
- Recall を優先して広めに検索し、LLM投入前に `context_k` 件へ絞り込み

### 回答制御（安全設計）
- CONTEXT に存在しない情報は回答しない
- 使用した文書チャンク（chunk_id）を回答文中に引用
- 根拠が見つからない場合は
**「文書内に根拠が見つかりません」** と返却

## ディレクトリ構成
```
rag_app/
├ api.py            # FastAPI（入出力・ルーティング・認証）
├ rag.py            # RAGロジック（検索・chunk・embedding）
├ llm_client.py     # OpenAI API 呼び出し（リトライ・コスト見積）
├ tests/            # pytest テストスイート（74件）
│  ├ test_api.py    # API エンドポイントテスト
│  ├ test_auth.py   # API キー認証テスト
│  ├ test_cost.py   # コスト見積もりテスト
│  ├ test_log_dashboard.py  # ログ分析テスト
│  ├ test_rag.py    # chunk / search テスト
│  └ test_llm_client.py  # クライアント生成テスト
├ scripts/          # 運用スクリプト
│  ├ bench_api.py   # ベンチマーク
│  └ log_dashboard.py  # ログ分析ダッシュボード
├ index/            # FAISS index / metadata（自動生成）
├ logs/             # JSONL形式のリクエストログ（自動生成）
├ .github/workflows/ci.yml  # CI（lint + test + coverage）
├ .env.example      # 環境変数テンプレート
├ requirements.txt
├ Dockerfile        # multi-stage build
└ README.md
```

## セットアップ
```bash
# 1. venv 作成 & 依存インストール
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt

# 2. 環境変数を設定
cp .env.example .env
# .env の OPENAI_API_KEY を設定

# 3. API サーバー起動
uvicorn api:app --reload

# 4. ブラウザで確認
# http://127.0.0.1:8000/docs
```
※ `.env` は GitHub に push しません。

## テスト
```bash
# モックテスト（OpenAI不要）— 74件
pytest tests/ -k "not e2e and not manual"

# カバレッジ付き
pytest tests/ -k "not e2e and not manual" --cov=. --cov-report=term-missing

# E2Eテスト（OPENAI_API_KEY が必要）
pytest tests/test_e2e_openai.py
```

## 主なAPIエンドポイント

| エンドポイント | メソッド | 認証 | 説明 |
|---|---|---|---|
| `/health` | GET | 不要 | ヘルスチェック |
| `/stats` | GET | 不要 | チャンク数・source別内訳・モデル情報 |
| `/ingest` | POST | 必要* | テキスト文書をRAGに登録 |
| `/ingest_files` | POST | 必要* | PDF/txt/md/docxファイルをアップロード登録 |
| `/ask` | POST | 必要* | 質問に対してRAGに基づく回答を返却 |
| `/ask_structured` | POST | 必要* | Structured Outputsで構造化JSON応答 |
| `/reset` | POST | 必要* | 全データ初期化 |

\* `RAG_API_KEY` 未設定時は認証スキップ（PoC モード）

### `/ask` パラメータ
| パラメータ | デフォルト | 説明 |
|---|---|---|
| `question` | (必須) | 質問文 |
| `retrieval_k` | 60 | 検索候補数 |
| `context_k` | 3 | LLMに渡すチャンク数 |
| `use_multi` | false | Multi-Query Retrieval |
| `debug` | false | 取得チャンク情報を返却 |
| `max_new_tokens` | 128 | 生成する最大トークン数 |

### `/ask_structured` レスポンス例
```json
{
  "parsed": {
    "answer": "有給休暇は5営業日前までに申請が必要です。",
    "references": ["社内規定@abc123#chunk0"],
    "confidence": "high"
  },
  "latency_ms": 1234,
  "trace_id": "a1b2c3d4e5f6"
}
```

## Docker
```bash
docker build -t rag-api .
docker run -p 8000:8000 --env-file .env rag-api
```

## API 認証
`RAG_API_KEY` 環境変数にキーを設定すると、
保護対象エンドポイント（`/ingest`, `/ask`, `/reset` 等）で `X-API-Key` ヘッダー認証が有効化されます。

```bash
# .env に追記
RAG_API_KEY=my-secret-key-123

# リクエスト例
curl -H "X-API-Key: my-secret-key-123" \
     -H "Content-Type: application/json" \
     -d '{"question":"有給休暇の申請期限は？"}' \
     http://localhost:8000/ask
```

未設定時は認証なし（PoC モード）で動作します。

## ログ分析ダッシュボード
```bash
# 全期間のログを分析
python scripts/log_dashboard.py

# 直近7日間のみ
python scripts/log_dashboard.py --days 7

# JSON形式で出力（他ツール連携用）
python scripts/log_dashboard.py --json
```

出力例:
```
============================================================
  RAG API — ログ分析ダッシュボード
============================================================
  質問数           : 42
  キャッシュHit率  : 28.6%
  レイテンシ (ms)
    avg=1200, median=1050, p95=2800, max=3500
  トークン使用量
    total_tokens   : 12,500
    推定コスト     : $0.002350
```

## 環境変数一覧

| 変数名 | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `OPENAI_API_KEY` | ○ | — | OpenAI API キー |
| `OPENAI_MODEL` | — | `gpt-4o-mini` | 使用するモデル |
| `RAG_API_KEY` | — | (空=認証無効) | API キー認証 |
| `CORS_ORIGINS` | — | `*` | 許可オリジン (カンマ区切り) |
| `EMB_MODEL` | — | `paraphrase-multilingual-MiniLM-L12-v2` | 埋め込みモデル |
| `RETRIEVAL_CACHE_TTL_SEC` | — | `600` | 検索キャッシュ TTL(秒) |
| `RETRY_MAX_RETRIES` | — | `3` | OpenAI リトライ回数 |
| `RETRY_BASE_DELAY` | — | `1.0` | リトライ初回待機(秒) |
| `RETRY_MAX_DELAY` | — | `10.0` | リトライ最大待機(秒) |

## 想定ユースケース
- 社内文書検索・FAQ の PoC
- RAG設計（chunk / retrieval）の検証
- LLM活用可否の技術調査
- 本番導入前の精度・挙動確認

## 本実装の位置づけ
- ⭕ 技術検証・PoC用途
- ⭕ RAG設計の評価・改善
- ⭕ 社内データ活用の可否確認
- ⭕ API キー認証による基本的なアクセス制御
- ⭕ トークン使用量・コストのモニタリング
- ❌ 高トラフィック本番運用
- ❌ OAuth / JWT を含む業務システム
- ❌ SLAを伴うサービス提供

## セキュリティ
- Python 3.12 以上が必須です
- APIキーは `.env` で管理し、コードには含めません
- `.env` は GitHub に push しません
- 本番運用では認証・ログ・アクセス制御を追加してください

## 今後の拡張案
- rerankingモデルの導入
- chunk戦略の高度化（セマンティック分割）
- 推論速度の最適化
- ローカルLLM対応

## 補足
本プロジェクトは**ポートフォリオおよび案件向けPoC実装**として整理しています。コードの再利用・拡張を前提とした構成です。