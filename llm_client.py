"""OpenAI API クライアント生成 — .env から API キーを読み込む。"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TypeVar

from dotenv import dotenv_values
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).with_name(".env")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# モデル別トークン単価 (USD / 1K tokens)  — 2024-12 時点の参考値
# 新モデル追加時はここに行を足す
# ---------------------------------------------------------------------------
_COST_PER_1K: dict[str, tuple[float, float]] = {
    # (input, output) per 1K tokens
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """トークン数からコスト (USD) を推定する。

    モデルが単価表にない場合は ``None`` を返す。
    """
    rate = _COST_PER_1K.get(model)
    if rate is None:
        return None
    in_cost, out_cost = rate
    return round(input_tokens / 1000 * in_cost + output_tokens / 1000 * out_cost, 8)


def get_openai_client() -> OpenAI:
    """`.env` から ``OPENAI_API_KEY`` を読み取り :class:`OpenAI` を返す。

    Raises:
        RuntimeError: API キーが見つからない場合。
    """
    env = dotenv_values(ENV_PATH)
    key = (env.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in .env or environment")
    return OpenAI(api_key=key, timeout=30.0)


def retry_with_backoff(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0
):
    """エクスポーネンシャルバックオフで関数をリトライするデコレータ。

    Args:
        max_retries: 最大リトライ回数（デフォルト: 3）。
        base_delay: 初回待機時間秒（デフォルト: 1.0）。
        max_delay: 最大待機時間秒（デフォルト: 10.0）。

    Raises:
        APIError: 最終的に失敗した場合。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_sec = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"rate limited (attempt {attempt+1}/{max_retries}), "
                            f"waiting {wait_sec:.1f}s: {e}"
                        )
                        time.sleep(wait_sec)
                    else:
                        logger.error(f"rate limit exceeded after {max_retries} retries")
                        raise
                except APITimeoutError as e:
                    if attempt < max_retries - 1:
                        wait_sec = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"timeout (attempt {attempt+1}/{max_retries}), "
                            f"waiting {wait_sec:.1f}s: {e}"
                        )
                        time.sleep(wait_sec)
                    else:
                        logger.error(f"timeout after {max_retries} retries")
                        raise
                except APIError as e:
                    # 他の API エラー（4xx など）はリトライしない
                    logger.error(f"API error (no retry): {e}")
                    raise

        return wrapper

    return decorator
