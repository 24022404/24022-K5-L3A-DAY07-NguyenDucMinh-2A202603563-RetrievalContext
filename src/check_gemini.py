"""Check the Gemini key and list models that support generateContent. Run once before the agent."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY missing in .env")
client = genai.Client(api_key=key)
names = [m.name for m in client.models.list() if "generateContent" in (m.supported_actions or [])]
print("flash models:", [n for n in names if "flash" in n])
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
resp = client.models.generate_content(model=model, contents="Reply with the single word OK.")
print(f"{model} ->", resp.text)
