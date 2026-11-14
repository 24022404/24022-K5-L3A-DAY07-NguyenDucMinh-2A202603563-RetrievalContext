"""Retrieval methods on the same Day 6 chunks: dense (Chroma), BM25, hybrid RRF, hybrid + rerank.

Every method returns chunks in the starter format plus `score`, `rank` and `method`.
A filter (e.g. {"chapter": 5}) comes only from the user's question context, never from gold.
"""
from __future__ import annotations

import re
import time

import chromadb

from config import CANDIDATES, CHROMA_DIR, DAY6_COLLECTION, EMBED_MODEL, RERANK_MODEL, RRF_K


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _match(chunk: dict, filters: dict | None) -> bool:
    return not filters or all(chunk["metadata"].get(k) == v for k, v in filters.items())


class Retrievers:
    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(DAY6_COLLECTION)
        if self.col.count() != len(chunks):
            raise RuntimeError("Chroma collection and chunks.json differ; re-run export_day6.py")
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self._reranker = None

    # --- method 1: dense on the Day 6 vector store -------------------------------------------
    def dense(self, query: str, filters: dict | None, k: int) -> list[dict]:
        vec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=vec, n_results=k, where=filters or None,
                             include=["distances"])
        return [{**self.by_id[i], "score": round(1 - d, 4)} for i, d in zip(res["ids"][0], res["distances"][0])]

    # --- method 2: lexical BM25 --------------------------------------------------------------
    def bm25_search(self, query: str, filters: dict | None, k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted((i for i, c in enumerate(self.chunks) if _match(c, filters)), key=lambda i: -scores[i])
        return [{**self.chunks[i], "score": round(float(scores[i]), 4)} for i in order[:k] if scores[i] > 0]

    # --- method 3: hybrid = Reciprocal Rank Fusion of dense + BM25 ----------------------------
    def hybrid(self, query: str, filters: dict | None, k: int) -> list[dict]:
        fused: dict[str, float] = {}
        for ranked in (self.dense(query, filters, CANDIDATES), self.bm25_search(query, filters, CANDIDATES)):
            for r, c in enumerate(ranked):
                fused[c["id"]] = fused.get(c["id"], 0.0) + 1.0 / (RRF_K + r + 1)
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{**self.by_id[i], "score": round(fused[i], 5)} for i in order]

    # --- method 4: hybrid candidates re-ranked by a cross-encoder -----------------------------
    def rerank(self, query: str, filters: dict | None, k: int) -> list[dict]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL, device="cpu")
        cands = self.hybrid(query, filters, CANDIDATES)
        scores = self._reranker.predict([(query, c["text"]) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -scores[i])[:k]
        return [{**cands[i], "score": round(float(scores[i]), 4)} for i in order]

    def search(self, method: str, query: str, filters: dict | None, top_k: int) -> tuple[list[dict], float]:
        fn = {"dense": self.dense, "bm25": self.bm25_search, "hybrid": self.hybrid, "rerank": self.rerank}[method]
        t = time.perf_counter()
        results = fn(query, filters, top_k)
        ms = (time.perf_counter() - t) * 1000
        return [{**c, "rank": r + 1, "method": method} for r, c in enumerate(results)], ms


def pack_context(results: list[dict], budget: int) -> tuple[str, list[str]]:
    """Same rule as starter/baseline.py: keep rank order, skip duplicates, skip a chunk that
    does not fit in the remaining character budget (header with ID/source/page included)."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"] and g["evidence"] in chunk["text"]
               for g in gold)
