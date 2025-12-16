from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

from openai import OpenAI

client = OpenAI()  # .env が読めていればここでOK
resp = client.responses.create(
    model="gpt-4o-mini",
    input="Say OK in Japanese."
)
print(resp.output_text)

