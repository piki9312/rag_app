# Inference Speed Benchmark (PoC)

## 概要
本フォルダは、RAG/LLM応答の体感品質に直結する **推論速度（レイテンシ / tokens/sec）** を、
PoCとして計測・比較するためのドキュメントです。

## 計測対象
- 生成（token-by-token）時の速度
- 設定差分による速度変化（例：max_new_tokens / context_k など）
- （将来）LoRA merge 前後の比較

## 指標
- 1リクエストあたりのレイテンシ（sec）
- tokens/sec（生成トークン数 ÷ 時間）
- p50 / p95（余裕があれば）

## ベンチ条件（固定するもの）
- 同一プロンプト / 同一入力
- max_new_tokens を固定
- サンプリング条件を固定（可能なら seed固定）
- 実行環境（CPU/GPU, OS）を記録

## 前提/環境
- model / model_version
- backend（例：vLLM / llama.cpp / transformers）
- quantization（有無と方式）
- hardware（GPU/CPU, VRAM/RAM）
- batch_size / context_length
- sampling（temperature / top_p / top_k）

## 入力条件（固定）
- 入力トークン数 or 文字数
- 検索件数（context_k）
- ドキュメント長（最大/平均）

## 測定手順
- ウォームアップ回数 / 計測回数 / 捨てるサンプル数
- 測定区間の定義（TTFT / 全生成時間 / 総レイテンシ）
- 出力形式（例：CSV）

## 実行方法
- （ここに `scripts/bench_generate.py` などを追加予定）

## results（Baseline）
- runs=5, warmup=1
- avg: 3.503 s
- p50(median): 3.157 s
- p95(rough): 3.447 s  ※ runsが少ないため参考値
- raw (sec): [3.447, 3.157, 5.94, 2.77, 2.198]

### Notes
- 1回だけ 5.94s の外れ値が出た（ネットワーク/初回負荷/キャッシュ等の影響の可能性）
- 以後の比較では debug=False、同一question、同一パラメータで測定する

## Results (context_k=3)
- runs=10, warmup=2
- avg: 2.642 s
- p50(median): 2.460 s
- raw (sec): [3.177, 3.169, 2.252, 2.867, 3.748, 2.194, 2.447, 1.982, 2.106, 2.473]

### Notes
- baseline(context_k=6) と比較して **速度が速く** なった
- context_k を削減しても、代表的な質問に対する回答品質に大きな劣化は見られなかった

## Results (use_multi=False, context_k=3)

- runs=10, warmup=2
- avg: 2.037 s
- p50(median): 1.896 s
- raw (sec): [1.952, 1.737, 2.17, 2.539, 1.746, 2.266, 1.722, 1.331, 3.072, 1.84]

### Notes
- Multi-Query Retrieval を無効化すると、速度が 改善 した
- 速度とRecallのトレードオフがあるため、用途により切替が必要

## Results (retrieval_k sweep, context_k=3, use_multi=False)

| retrieval_k | runs | avg (s) | p50 (s) | p95 (s) |
|---:|---:|---:|---:|---:|
| 10 | 10 | 2.076 | 2.058 | 2.648 |
| 30 | 10 | 2.500 | 1.949 | 3.071 |
| 60 | 10 | 1.741 | 1.743 | 1.994 |

### Observations

- retrieval_k を増加させても、context_k を固定しているため
  LLM に渡すトークン量は変化しない
- 本測定では、retrieval_k=60 が最も安定したレイテンシを示した
- retrieval_k が小さい場合、検索結果のばらつきにより
  生成時間が不安定になる可能性がある
- 推論全体のレイテンシは、検索処理よりも
  LLM生成の安定性に強く影響されると考えられる

## Results (max_new_tokens sweep, context_k=3, use_multi=False, retrieval_k=60)

| max_new_tokens | runs | avg (s) | p50 (s) | p95 (s) |
|---:|---:|---:|---:|---:|
| 64  | 10 | 2.589 | 2.041 | 3.897 |
| 128 | 10 | 1.752 | 1.688 | 1.897 |
| 256 | 10 | 2.477 | 1.920 | 3.087 |

### Observations (Generation)

- max_new_tokens は上限値であり、実際の生成トークン数は
  質問内容やモデルの収束挙動に依存する
- 本検証では max_new_tokens=128 が最も安定して高速だった（p50 기준）
- 上限が低すぎる場合（64）、生成の不安定化により
  外れ値が発生するケースが見られた
- 上限が高すぎる場合（256）、生成時間のばらつきが増加した
- 実務では「必要十分な上限」を設定することが
  レイテンシと安定性の両立に有効と考えられる

  
## Summary / Implications

- 推論レイテンシは、context_k と検索戦略の影響が大きい
- context_k を削減することで、回答品質を大きく損なわずに
  体感速度を改善できる可能性がある
- Multi-Query Retrieval は Recall 向上に有効だが、
  レイテンシとのトレードオフが明確に存在する
- 本番運用では、用途（精度優先 / 速度優先）に応じて
  パラメータを切り替える設計が有効と考えられる

