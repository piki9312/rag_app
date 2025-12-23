# Retrieval Cache Verification (PoC)

## 概要
本章では、RAG API に実装した **Retrieval キャッシュ**について、
PoCとして以下の点を検証した。

- Retrieval キャッシュが正しくヒットするか
- index 更新時にキャッシュが無効化されるか
- 推論全体のレイテンシに与える影響

本検証では、**LLM 生成はキャッシュ対象外**とし、
Embedding + FAISS 検索結果のみをキャッシュする。

---

## キャッシュ設計

### 対象
- Retrieval（embedding + FAISS 検索結果）
- LLM 生成結果はキャッシュしない

### キャッシュキー
以下の情報を正規化・結合し、SHA256 でハッシュ化した。

- embedding model
- index_version（faiss.index / meta.json の fingerprint）
- retrieval_k
- use_multi
- normalized question

```
key = hash(
  emb_model |
  index_version |
  retrieval_k |
  use_multi |
  normalized_question
)
```

### TTL

インメモリキャッシュ

TTL = 600 秒（PoC 設定）

## index_version による無効化

キャッシュの安全性を担保するため、
FAISS index および metadata の状態を表す
**index_version** をキャッシュキーに含めた。
- index_version = mtime + size の fingerprint
- /ingest 実行後に index_version が変化
- index_version 変更時は自動的に cache miss となる
これにより、
**検索対象データが更新された状態での誤キャッシュ利用を防止**できる。

## 検証方法

### 条件
- 同一 question / 同一パラメータ
- retrieval_k / context_k / use_multi / max_new_tokens 固定
- runs=10, warmup=2
### 確認方法
- /ask を連続実行
- 1リクエスト1行JSONログを確認
- 以下のフィールドを検証
```
{
  "cache_hit": true / false,
  "retrieval_ms": <int>,
  "index_version": "<fingerprint>"
}
```

## 検証結果

### 推論レイテンシ（/ask 全体）
```
runs=10, warmup=2
avg=2.172s, p50=2.158s, p95≈2.845s
raw: [2.251, 2.354, 1.901, 1.703, 1.623, 2.48, 2.845, 2.862, 2.064, 1.637]
```
### Observations
- 2回目以降のリクエストで cache_hit=true を確認
- cache hit 時は retrieval_ms が大幅に減少
- /ask 全体のレイテンシは LLM 生成時間が支配的なため、
キャッシュによる改善は retrieval 部分に局所的に現れる
- /ingest 実行後、index_version が変化し
次回リクエストが cache miss となることを確認

## 考察 / Implications
- Retrieval キャッシュは、
**同一・類似質問が繰り返される利用形態**において有効
- index_version をキーに含めることで、
PoC 環境でも安全にキャッシュを導入できる
- 推論全体の高速化には、
Retrieval キャッシュに加えて
 - max_new_tokens の最適化
 - Answer キャッシュ（FAQ用途）
 - Streaming 応答
   などの併用が有効と考えられる

## Summary
- Retrieval キャッシュを PoC として実装・検証した
- index 更新時の自動無効化が正しく機能することを確認
- キャッシュ効果は retrieval レイヤに明確に現れ、
RAG 運用における有効な最適化手法であることを確認した
