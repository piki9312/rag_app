# RAG API 技術検証PoC

## 概要
本フォルダは、同一のRAG実装（`core/rag_app`）を用いて、
**RAGを業務利用できるか判断するための技術検証（PoC）**として整理したドキュメントです。

完成システム（FAQ）としての説明ではなく、以下のような設計要素が
回答品質・安全性に与える影響を確認することを目的とします。

---

## 検証対象アーキテクチャ（共通）
Document → Chunking → Embedding（sentence-transformers） → FAISS Retrieval
→ Context Selection → LLM（OpenAI API） → Answer

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

## 検証観点（PoCとして確認したい点）

- チャンク分割粒度（段落単位 + overlap）が
  回答の一貫性にどの程度寄与するか
- Multi-Query Retrieval による Recall 向上と
  ノイズ増加のトレードオフ
- retrieval_k / context_k の設定が
  回答品質と hallucination に与える影響
- 根拠提示ルールによる誤答抑制効果

## 検証から得られた知見（暫定）

- 段落単位のチャンク分割は、文単位よりも
  回答文の一貫性が高くなる傾向があった
- Multi-Query Retrieval を有効にすると
  Recall は向上するが、context_k を絞らないと
  ノイズが増加しやすい
- 根拠提示を強制することで誤答は減少したが、
  回答不可となるケースが増えた

## ディレクトリ構成
```bash
rag_app/
├ api.py            # FastAPI（入出力・ルーティング）
├ rag.py            # RAGロジック（検索・chunk・embedding）
├ llm_client.py     # OpenAI API 呼び出し（環境変数管理）
├ scripts/          # ingest / ベンチマーク用スクリプト
├ index/            # FAISS index / metadata（自動生成）
├ .env              # APIキー（Git管理しない）
├ requirements.txt
├ README.md
└ experiments.md

```
## セットアップ
1. 依存関係をインストール: `pip install -r requirements.txt`
2. 環境変数を設定: `.env` ファイルを作成し、以下を設定します。
  ```
  OPENAI_API_KEY=your_api_key
  OPENAI_MODEL=gpt-4o-mini
  ```
※`.env`はGitHubにpushしません。

## APIサーバーを起動: `uvicorn api:app --reload`

## 主なAPIエンドポイント
- `/health` ヘルスチェック
- `/stats` 登録済みチャンク数・source別内訳・使用モデルを返却
- `/ingest` テキスト文書をRAGに登録
- `/ingest_file` PDF/txt/md/docxファイルをアップロードして登録
- `/ask` 質問に対してRAGに基づく回答を返却
  - `retrieval_k`:検索候補数
  - `context_k`:LLMに渡すチャンク数
  - `use_multi`:Multi-Query Retrievalの有無
  - `debug`:取得チャンク情報の返却有無

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