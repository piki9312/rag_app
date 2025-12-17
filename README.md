# rag_app — RAG API (FAISS + sentence-transformers + OpenAI)

社内文書（TXT / PDF / DOCX）を取り込み、質問に対して  
**根拠チャンクID（[chunk_id]）を明示して回答する** RAG（Retrieval-Augmented Generation）API です。

FastAPI を用い、  
**文書投入 → 検索 → LLM回答** の最小構成を実装しています。

---

## 特徴
- FAISS + sentence-transformers による高速検索
- 回答には **必ず根拠チャンクIDを引用**
- TXT / PDF / DOCX 対応
- 再現性重視（venv / requirements / setup & run script）
- ローカル環境で完結（GPU不要）

---

## 構成

rag_app/
├─ api.py # FastAPI本体
├─ rag.py # RAGストア（FAISS + embedding）
├─ requirements.txt # 依存関係
├─ README.md
├─ .env # APIキー（※Git管理しない）
├─ index/ # FAISS index / meta（自動生成）
├─ logs/ # 実行ログ（jsonl）
└─ scripts/
├─ setup.bat # 環境構築
└─ run.bat # 起動


---

## 必要要件
- Windows
- Python 3.10+（推奨: 3.12）
- OpenAI API Key

---

## セットアップ

### 1. `.env` を作成（※コミットしない）
`rag_app` 直下に `.env` を作成：

**```env**
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
**```**

※ APIキーは フルで1行 に貼り付けてください。
2. 依存関係の導入

cd /d C:\LLM\python_lesson\rag_app
scripts\setup.bat

起動

cd /d C:\LLM\python_lesson\rag_app
scripts\run.bat

Swagger UI：

http://127.0.0.1:8000/docs

使い方（最短）
1) 文書投入

POST /ingest_file

TXT / PDF / DOCX をアップロードすると
文書は自動で chunk 分割され、検索用に登録されます。

成功時レスポンス例：

{
  "ok": true,
  "source": "company_rules.txt",
  "ingested_chunks": 6,
  "total_chunks": 6
}

2) 質問

POST /ask

{
  "question": "打刻漏れをした場合、いつまでに修正申請が必要ですか？",
  "top_k": 6,
  "debug": true
}

レスポンス例：

{
  "answer": "打刻漏れが発生した場合は当日中に修正申請が必要です。[11]",
  "retrieved": [
    {
      "chunk_id": 11,
      "source": "company_rules.txt",
      "text": "..."
    }
  ],
  "latency_ms": 312
}

評価方法（品質保証）

    ダミー社内規程（経費 / 勤怠 / 有給）を投入

    10問すべてで

        回答に [chunk_id] が含まれる

        根拠チャンクが一致することを確認済み

便利なエンドポイント

    GET /health : 生存確認

    GET /stats : 文書投入状況

    POST /reset : index 全削除（開発用）

注意

    .env / .venv / index / logs は Git 管理対象外

    APIキーは 絶対に公開しない

今後の拡張予定

    Docker対応

    クラウドデプロイ

    自動評価スクリプト

    メタ情報（ページ番号等）の強化


---

## 次にやること（これだけ）
README 保存できたら、cmdで👇

```bat
git add README.md
git commit -m "Fix README formatting"
git push