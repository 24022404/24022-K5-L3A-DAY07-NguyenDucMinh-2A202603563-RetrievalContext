"""Step 2: compare retrieval configs without the LLM (original query, same corpus/index/labels).

Usage:  python src/retrieval_eval.py --split dev
        python src/retrieval_eval.py --split test --configs dense hybrid_rerank_1600
Metrics per config: Hit@1, Evidence Hit@3, Context Evidence Rate (evidence inside the packed
context), mean context chars, median latency. Unanswerable cases are not in the denominator.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics

from config import CHUNKS_PATH, CONFIGS, QUESTIONS_PATH, RESULTS_DIR
from retrievers import Retrievers, is_hit, pack_context


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    qs = [q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if q["split"] == args.split]
    R = Retrievers(chunks)
    R.search("rerank", "warm up models", None, 3)  # load models before timing

    out_dir = RESULTS_DIR / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.configs:
        cfg = CONFIGS[name]
        traces = []
        for q in qs:
            results, ms = R.search(cfg["method"], q["question"], q.get("filter"), cfg["top_k"])
            context, ids = pack_context(results, cfg["budget"])
            flags = [is_hit(c, q["gold"]) for c in results]
            traces.append({
                "query_id": q["id"], "split": q["split"], "config": name, **cfg,
                "question": q["question"], "filter": q.get("filter"),
                "results": [{"rank": c["rank"], "id": c["id"], "page": c["page"],
                             "chapter": c["metadata"]["chapter"], "score": c["score"], "hit": f,
                             "text": c["text"]} for c, f in zip(results, flags)],
                "context_ids": ids, "context_chars": len(context),
                "hit1": bool(flags[:1] and flags[0]), "hit3": any(flags[:3]),
                "context_hit": any(f and c["id"] in ids for c, f in zip(results, flags)),
                "latency_ms": round(ms, 1)})
        (out_dir / f"{name}_{args.split}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
        N = len(traces)
        row = {"config": name, "split": args.split, **cfg, "N": N,
               "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
               "context_evidence": sum(t["context_hit"] for t in traces),
               "mean_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
               "median_latency_ms": round(statistics.median(t["latency_ms"] for t in traces), 1),
               "cost_usd": 0,
               "misses_hit3": " ".join(t["query_id"] for t in traces if not t["hit3"]),
               "lost_in_packing": " ".join(t["query_id"] for t in traces if t["hit3"] and not t["context_hit"])}
        rows.append(row)
        print(f"{name:20s} {args.split} Hit@1 {row['hit1']}/{N}  Hit@3 {row['hit3']}/{N}  "
              f"CtxEvidence {row['context_evidence']}/{N}  ctx~{row['mean_context_chars']}ch  "
              f"{row['median_latency_ms']}ms  miss3=[{row['misses_hit3']}] lost=[{row['lost_in_packing']}]")
    path = out_dir / f"summary_{args.split}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
