# RAG API サンプル

## 概要
このプロジェクトは、Retrieval-Augmented Generation (RAG) パターンを使用したAPI実装のサンプルです。外部データソースから関連情報を取得し、大規模言語モデルに統合して、より正確で文脈に基づいた応答を生成します。

## 機能
- ドキュメント検索と取得
- ベクトル埋め込み処理
- LLMを使用した応答生成
- RESTful API インターフェース
- キャッシング機能

## アーキテクチャ
```
クライアント → API エンドポイント → RAG パイプライン → LLM
                                        ↓
                                    ベクトルDB
                                    ドキュメントストア
```

## セットアップ
1. 依存関係をインストール: `pip install -r requirements.txt`
2. 環境変数を設定: `.env` ファイルを作成
3. ドキュメントを準備: `data/` ディレクトリに配置
4. APIサーバーを起動: `uvicorn api:app --reload`

## API使用方法
```bash
curl -X POST http://127.0.0.1:8000/docs/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "質問内容"}'
```

## 注記(セキュリティー)
- Python 3.9以上が必須です
- APIキーの設定を忘れずに行ってください
- 本番環境ではセキュリティ設定を確認してください
- .env と data/（文書・ベクトルDB等）は GitHub に push しません
- 本番運用では認証・ログ・アクセス制御を追加してください

