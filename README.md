# RAG API (PoC)


## 概要
本プロジェクトは、社内文書やマニュアルを対象とした
**検索型QAシステムの技術検証**として実装したRAG APIです。

本番運用を目的とせず、
- RAG設計(chunk / retrieval /prompt)
- 回答の安全性(hallucination対策)
- 運用時の確認性(ログ・統計)

を短期間で検証できる構成を重視しています。

## 特徴
- FAISS + sentence-transformersによる文書検索
- OpenAI APIを用いた回答生成
- 参照した文書チャンク(chunk_id)を回答内に明示
- 根拠のない情報は回答しない安全設計
- FastAPIによるREST API提供
- ローカル環境で実行可能(PoC向け)

## アーキテクチャ
```bash
Client
  ↓
FastAPI (/ask, /ingest)
  ↓
RAG Pipeline
  ├─ 文書分割（chunk）
  ├─ embedding（sentence-transformers）
  ├─ FAISS 検索（Multi-Query対応）
  ↓
OpenAI API
  ↓
回答生成（根拠付き）

```

## RAG設計のポイント

### チャンク分割
- 文書を 段落単位 で分割し、意味のまとまりを保持
- 文脈切れを防ぐため overlap を付与
- 極端に短いチャンクは前チャンクに統合し、ノイズを抑制

### 検索戦略
- 通常のベクトル検索に加え、質問文をキーワード化した
Multi-Query Retrieval を実装
- Recall を優先して広めに検索し、LLM投入前に context_k 件へ絞り込み
### 回答制御（安全設計）
- CONTEXT に存在しない情報は回答しない
- 使用した文書チャンク（chunk_id）を回答文中に引用
- 根拠が見つからない場合は
**「文書内に根拠が見つかりません」** と返却

## ディレクトリ構成
```bash
rag_app/
├ api.py            # FastAPI（入出力・ルーティング）
├ rag.py            # RAGロジック（検索・chunk・embedding）
├ llm_client.py     # OpenAI API 呼び出し（環境変数管理）
├ tests/            # pytest テストスイート（25件）
├ scripts/          # ingest / ベンチマーク用スクリプト
├ index/            # FAISS index / metadata（自動生成）
├ logs/             # JSONL形式のリクエストログ（自動生成）
├ .env.example      # 環境変数テンプレート
├ requirements.txt
├ Dockerfile
└ README.md
```
## セットアップ
1. 依存関係をインストール: `pip install -r requirements.txt`
2. 環境変数を設定: `.env.example` をコピーして `.env` を作成し、APIキーを設定します。
  ```
  cp .env.example .env
  # OPENAI_API_KEY を設定
  ```
※`.env`はGitHubにpushしません。

3. APIサーバーを起動: `uvicorn api:app --reload`
4. `http://127.0.0.1:8000/docs`を開く

## テスト
```bash
# モックテスト（OpenAI不要）
pytest tests/ -k "not e2e"

# E2Eテスト（OPENAI_API_KEY が必要）
pytest tests/test_e2e_openai.py
```

## 主なAPIエンドポイント
- `/health` ヘルスチェック
- `/stats` 登録済みチャンク数・source別内訳・使用モデルを返却
- `/ingest` テキスト文書をRAGに登録
- `/ingest_files` PDF/txt/md/docxファイルをアップロードして登録
- `/ask` 質問に対してRAGに基づく回答を返却
  - `retrieval_k`:検索候補数
  - `context_k`:LLMに渡すチャンク数
  - `use_multi`:Multi-Query Retrievalの有無
  - `debug`:取得チャンク情報の返却有無
  - `max_new_tokens`:生成する最大トークン数

## 想定ユースケース
- 社内文書検索・FAQ の PoC
- RAG設計（chunk / retrieval）の検証
- LLM活用可否の技術調査
- 本番導入前の精度・挙動確認


## 本実装の位置づけ
- ⭕ 技術検証・PoC用途
- ⭕ RAG設計の評価・改善
- ⭕ 社内データ活用の可否確認
- ❌ 高トラフィック本番運用
- ❌ 認証・認可を含む業務システム
- ❌ SLAを伴うサービス提供

## 注記(セキュリティー)
- Python 3.9以上が必須です
- APIキーは`.env`で管理し、コードには含めません
- `.env` は GitHub に push しません
- 本番運用では認証・ログ・アクセス制御を追加してください

## 今後の拡張案
- chunk戦略の高度化
- rerankingモデルの導入
- 推論速度の最適化
- ローカルLLM対応
- Structured OutputsによるJSON応答

## 補足
本プロジェクトは**ポートフォリオおよび案件向けPoC実装**として整理しています。コードの再利用・拡張を前提とした構成です。