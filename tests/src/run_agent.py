"""Step 3: run the real agent on a split for one retrieval config.

Usage:  python src/run_agent.py --config dense --split dev
Writes results/agent/<config>_<split>.json (full trace per case) and prints automatic stats.
Manual labels go in results/agent/reviews_<config>_<split>.json (template is created).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from agent import GeminiAgent
from config import CASES_PATH, CHUNKS_PATH, CONFIGS, QUESTIONS_PATH, RESULTS_DIR
from retrievers import Retrievers, is_hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in chunks}
    gold = {q["id"]: q["gold"] for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}  # scoring only
    cases = [c for c in json.loads(CASES_PATH.read_text(encoding="utf-8")) if c["split"] == args.split]
    agent = GeminiAgent(Retrievers(chunks))

    out_dir = RESULTS_DIR / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    for i, case in enumerate(cases, 1):
        try:
            r = agent.run(case["question"], method=cfg["method"], top_k=cfg["top_k"],
                          budget=cfg["budget"], filters=case.get("filter"))
        except Exception as e:  # provider / store error: recorded, never replaced by a fake answer
            r = {"status": "error", "error": f"{type(e).__name__}: {e}", "answer": "", "citations": [],
                 "events": [], "context_ids": [], "n_searches": 0, "tokens": {}, "latency_s": None}
        ctx_hit = None
        if case["answerable"] and r["status"] != "error":
            ctx_hit = any(is_hit(by_id[cid], gold[case["id"]]) for cid in r["context_ids"])
        traces.append({"query_id": case["id"], "split": case["split"], "answerable": case["answerable"],
                       "config": args.config, **cfg, "question": case["question"],
                       "context_evidence": ctx_hit, **r})
        print(f"[{i}/{len(cases)}] {case['id']} {r['status']:8s} searches={r['n_searches']} "
              f"ctx_evidence={ctx_hit} {r.get('error', '')[:120]}")
        (out_dir / f"{args.config}_{args.split}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")

    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    una = [t for t in ok if not t["answerable"]]
    stats = {
        "config": args.config, "split": args.split, "model": agent.model, "cases": len(traces),
        "errors": len(traces) - len(ok),
        "tool_use": f"{sum(t['n_searches'] > 0 for t in ok)}/{len(ok)}",
        "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "context_evidence_rate": f"{sum(bool(t['context_evidence']) for t in ans)}/{len(ans)}",
        "answered_answerable": f"{sum(t['status'] == 'answered' for t in ans)}/{len(ans)}",
        "abstained_unanswerable": f"{sum(t['status'] == 'abstain' for t in una)}/{len(una)}",
        "answers_with_citation": f"{sum(bool(t['citations']) for t in ok if t['status'] == 'answered')}/{sum(t['status'] == 'answered' for t in ok)}",
        "median_latency_s": statistics.median(t["latency_s"] for t in ok) if ok else None,
        "tokens_prompt": sum(t["tokens"].get("prompt", 0) for t in ok),
        "tokens_output": sum(t["tokens"].get("output", 0) for t in ok),
        "cost_usd": "0 (Gemini API free tier)",
    }
    (out_dir / f"stats_{args.config}_{args.split}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))

    review_path = out_dir / f"reviews_{args.config}_{args.split}.json"
    if not review_path.exists():
        reviews = [{"query_id": t["query_id"], "answerable": t["answerable"], "status": t["status"],
                    "answer": t["answer"], "citations": t["citations"],
                    "citation_support": [None] * len(t["citations"]),
                    **({"answer_grade": None} if t["answerable"] else {"abstention_correct": None}),
                    "reason": ""} for t in ok]
        review_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
