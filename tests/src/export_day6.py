"""Step 1: export Day 6 chunks (from the Chroma collection itself) and build the Day 7 benchmark.

- chunks.json   : every chunk of Day 6 collection eos_C, in the starter data format
- questions.json: 21 Day 6 answerable questions; evidence mapped to the exact chunk text
- cases.json    : the 21 answerable + 4 unanswerable cases (2 dev, 2 test)
"""
from __future__ import annotations

import json
import re

import chromadb

from config import CASES_PATH, CHROMA_DIR, CHUNKS_PATH, DAY6_COLLECTION, DAY6_QUESTIONS, QUESTIONS_PATH

UNANSWERABLE = [
    {"id": "u01", "split": "dev",
     "question": "When should a writer use a dash instead of a comma or a semicolon?",
     "absence_reason": "The book gives rules for comma, semicolon, colon (quotations) and parentheses, but no rule on dashes. Checked: the word 'dash' occurs 0 times in the normalized corpus; chapters II and IV (punctuation, form) read in full."},
    {"id": "u02", "split": "dev",
     "question": "According to the list of words commonly misused, is 'irregardless' acceptable?",
     "absence_reason": "Chapter V has no entry for 'irregardless'; the word occurs 0 times in the corpus (chapter V entries read in full)."},
    {"id": "u03", "split": "test",
     "question": "Should numbers under ten be spelled out in words?",
     "absence_reason": "Chapter IV 'Numerals' only says not to spell out dates or other serial numbers; nothing about small numbers. 'under ten' occurs 0 times; 'ten' occurs only inside unrelated examples."},
    {"id": "u04", "split": "test",
     "question": "How should a website or an online article be cited in the references?",
     "absence_reason": "1920 book: chapter IV 'References' covers acts, scenes, lines, books and pages only. 'website', 'online', 'internet' occur 0 times."},
]


def exact_span(evidence: str, text: str) -> str | None:
    """Return the exact substring of `text` matching `evidence` up to whitespace."""
    pattern = r"\s+".join(re.escape(w) for w in evidence.split())
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(0) if m else None


def main() -> None:
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(DAY6_COLLECTION)
    got = col.get(include=["documents", "metadatas"])
    chunks = []
    for cid, text, m in zip(got["ids"], got["documents"], got["metadatas"]):
        chunks.append({"id": cid, "text": text, "source_id": m["document_id"], "source": m["source"],
                       "page": m["page"], "metadata": {k: m[k] for k in ("chapter", "chapter_title", "page_id", "chunk_index")}})
    chunks.sort(key=lambda c: (c["page"], c["metadata"]["chunk_index"]))

    questions, cases, problems = [], [], []
    for line in DAY6_QUESTIONS.open(encoding="utf-8"):
        q = json.loads(line)
        gold = []
        for c in chunks:
            if c["page"] in q["gold_pages"]:
                span = exact_span(q["evidence_quote"], c["text"])
                if span:
                    gold.append({"source_id": c["source_id"], "page": c["page"], "evidence": span})
        if not gold:
            problems.append(q["id"])
        gold = [dict(t) for t in {tuple(sorted(g.items())) for g in gold}]
        item = {"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
                "expected_answer": q["expected_answer"], "gold": gold}
        if q.get("filter"):
            item["filter"] = q["filter"]
        questions.append(item)
        cases.append({"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
                      "expected_behavior": "search_and_answer", **({"filter": q["filter"]} if q.get("filter") else {})})
    for u in UNANSWERABLE:
        cases.append({**u, "answerable": False, "expected_behavior": "search_and_abstain"})
    if problems:
        raise RuntimeError(f"evidence not inside a single chunk for: {problems}")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path, obj in ((CHUNKS_PATH, chunks), (QUESTIONS_PATH, questions), (CASES_PATH, cases)):
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n = {s: (sum(c["split"] == s and c["answerable"] for c in cases), sum(c["split"] == s and not c["answerable"] for c in cases)) for s in ("dev", "test")}
    print(f"chunks={len(chunks)} answerable/unanswerable dev={n['dev']} test={n['test']}")


if __name__ == "__main__":
    main()
