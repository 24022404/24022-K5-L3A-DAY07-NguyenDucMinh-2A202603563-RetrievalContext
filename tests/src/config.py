"""Shared settings for Day 7. Everything reuses the Day 6 index (same chunks, model, store)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # read GEMINI_MODEL / GEMINI_RPM before using them

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def _find_day6() -> Path:
    env = os.getenv("DAY6_DIR")
    if env:
        return Path(env).resolve()
    for p in sorted(ROOT.parent.iterdir()):
        if p.is_dir() and "day06" in p.name.lower() and (p / "chroma_db").exists():
            return p
    raise FileNotFoundError("Day 6 folder with chroma_db/ not found next to this repo; set DAY6_DIR")


DAY6_DIR = _find_day6()
CHROMA_DIR = DAY6_DIR / "chroma_db"
DAY6_COLLECTION = "eos_C"            # Day 6 chosen config C (paragraph chunks, chapter filter)
DAY6_QUESTIONS = DAY6_DIR / "benchmark" / "questions.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # same as Day 6, 384 dims
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # local, free

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_RPM = float(os.getenv("GEMINI_RPM", "8"))           # stay under free-tier rate limit

CHUNKS_PATH = DATA_DIR / "chunks.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"   # answerable questions with gold
CASES_PATH = DATA_DIR / "cases.json"           # all cases incl. unanswerable (no gold)

CANDIDATES = 20   # candidates per retriever before fusion / rerank
RRF_K = 60

# Retrieval configurations compared on dev. Selection rule (fixed before test):
# highest dev Context Evidence Rate, then Evidence Hit@3, then Hit@1, then lower latency.
CONFIGS = {
    "dense":         {"method": "dense",  "top_k": 3, "budget": 800},   # baseline (Day 6 vector store)
    "bm25":          {"method": "bm25",   "top_k": 3, "budget": 800},
    "hybrid":        {"method": "hybrid", "top_k": 3, "budget": 800},
    "hybrid_rerank": {"method": "rerank", "top_k": 3, "budget": 800},   # improvement 1
    "hybrid_rerank_1600": {"method": "rerank", "top_k": 3, "budget": 1600},  # improvement 2
}
BASELINE = "dense"
MAX_SEARCHES = 3
