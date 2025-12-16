# rag_app — RAG API (FAISS + sentence-transformers + OpenAI)

社内文書（TXT/PDF/DOCX）を取り込み、質問すると「根拠チャンクID付き」で回答するRAG APIです。  
FastAPIで `/ingest_file` で文書投入 → `/ask` でQA という最小構成になっています。

## 構成
- `api.py` : FastAPI本体（/ingest, /ingest_file, /ask, /stats, /reset）
- `rag.py` : FAISS + sentence-transformers の検索ストア
- `.env` : OpenAI APIキーなどの設定（**コミットしない**）
- `index/` : FAISS index と meta（永続化）
- `logs/events.jsonl` : 実行ログ（jsonl）
- `scripts/setup.bat` : セットアップ（venv作成 + 依存導入）
- `scripts/run.bat` : 起動（uvicorn）

## 必要要件
- Windows
- Python 3.10+（推奨: 3.12）

## セットアップ
1. `.env` を `rag_app` 直下に作成（例）

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
```


※APIキーはフルで1行に貼り付けてください。

2. 依存関係を導入
- `scripts\setup.bat` をダブルクリック  
  または cmd で：
  ```bat
  cd /d C:\LLM\python_lesson\rag_app
  scripts\setup.bat
  ```

## 起動
- `scripts\run.bat` をダブルクリック  
または cmd で：
```bat
cd /d C:\LLM\python_lesson\rag_app
scripts\run.bat
起動すると以下でSwagger UIが開けます：

http://127.0.0.1:8000/docs

使い方（最短）
1) 文書投入

Swagger UIで POST /ingest_file を開き、TXT/PDF/DOCXをアップロードします。
成功すると ingested_chunks と total_chunks が返ります。

2) 質問

POST /ask に以下を送ります：

{
  "question": "打刻漏れをした場合、いつまでに修正申請が必要ですか？",
  "top_k": 6,
  "debug": true
}


レスポンス例：

answer に [chunk_id] が引用され、末尾に 根拠: [id] が出ます

debug=true のとき retrieved に参照チャンクが返ります

便利なエンドポイント

GET /health : 生存確認

GET /stats : 取り込み状況（source別チャンク数など）

POST /reset : indexを全消しして作り直し（開発用）

よくあるトラブル
OPENAI_API_KEY が設定されていません

.env が rag_app 直下にあるか確認

OPENAI_API_KEY=... がフルで入っているか確認

サーバ再起動（Ctrl+C → scripts\run.bat）

PDFが空になる

スキャンPDFだとテキスト抽出できないことがあります（OCRが必要な場合あり）。

注意

.env は機密情報です。Gitにコミットしないでください（.gitignore 推奨）。


---

次のステップは **`.gitignore` を作って `.env`, `.venv`, `index/`, `logs/` を除外**するのが鉄板です。  
これも順番にやる？（すぐテンプレ出せる）
::contentReference[oaicite:0]{index=0}