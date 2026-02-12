import os
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

ENV_PATH = Path(__file__).with_name(".env")


def get_openai_client() -> OpenAI:
    env = dotenv_values(ENV_PATH)
    key = (env.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=key)
