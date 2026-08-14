"""
eval/run_eval.py
-----------------
Phase 1 retrieval evaluation harness (see /docs or the project chat log
for the "Phase 0/1/2/3/4" improvement plan this belongs to).

Loads eval/eval_dataset.json, runs each query through the *current*
SemanticSearchEngine (FAISS + dense embeddings only -- this is the
pre-hybrid-retrieval baseline), and reports:

  - Recall@1 / Recall@3 / Recall@5 / Recall@10
  - MRR (mean reciprocal rank)
  - Per-type breakdown (exact_reference vs semantic vs unanswerable)
  - An "unanswerable leakage" check: for unanswerable queries, what score
    does the top hit get? A high score there means the system would
    confidently retrieve garbage for a question it has no answer to --
    that's the number to watch, since this repo's chatbot.py is the
    layer that's supposed to catch this via the similarity threshold.

Ground-truth matching note: this corpus's FAISS index has MANY chunks
per (doc_id, article_number) pair (confirmed up to ~79 for one article --
inspect data/index/metadata.json yourself to see this). So a "hit" is
counted whenever ANY retrieved chunk in the top-k matches the expected
doc_id (article_number checked when present, but doc-level match alone
still counts as a partial credit "doc_hit" -- see the two recall columns
below).

Usage:
    python -m eval.run_eval                      # top_k=10, all queries
    python -m eval.run_eval --top-k 5
    python -m eval.run_eval --type semantic       # filter by query type
    python -m eval.run_eval --save results.json   # dump raw per-query hits

Run this BEFORE and AFTER any retrieval change (e.g. adding BM25 hybrid
search in Phase 2) and diff the summary numbers -- that's the entire
point of having this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search_engine.search import SemanticSearchEngine

EVAL_SET_PATH = Path(__file__).parent / "eval_dataset.json"
MAX_K = 10


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"]


def run_query(
    engine: SemanticSearchEngine, entry: dict, top_k: int = MAX_K, hybrid: bool = True
) -> dict:
    """Runs one eval entry through the search engine and scores it."""
    results = engine.search(entry["query"], top_k=top_k, hybrid=hybrid)

    expected_doc = entry.get("expected_doc_id")
    expected_art = entry.get("expected_article")

    doc_rank = None       # first rank (1-indexed) where doc_id matches
    doc_art_rank = None   # first rank where BOTH doc_id and article match

    for i, r in enumerate(results, start=1):
        if expected_doc and r["doc_id"] == expected_doc:
            if doc_rank is None:
                doc_rank = i
            if expected_art is not None and r.get("article_number") == expected_art:
                if doc_art_rank is None:
                    doc_art_rank = i

    top_score = results[0]["score"] if results else None

    # Scores cosinus bruts (échelle du seuil de chatbot.py) : le top du
    # classement, et le meilleur chunk du doc attendu (tous articles /
    # article exact) — pour vérifier que le filtre de contexte de
    # chatbot.py (cosine_score >= 0.75) ne rejette pas un hit attendu.
    top_cosine = results[0].get("cosine_score") if results else None
    expected_max_cosine = None
    expected_max_cosine_art = None
    if expected_doc:
        cos = [r.get("cosine_score") for r in results
               if r["doc_id"] == expected_doc and r.get("cosine_score") is not None]
        if cos:
            expected_max_cosine = max(cos)
        if expected_art is not None:
            cos_art = [r.get("cosine_score") for r in results
                       if r["doc_id"] == expected_doc
                       and r.get("article_number") == expected_art
                       and r.get("cosine_score") is not None]
            if cos_art:
                expected_max_cosine_art = max(cos_art)

    return {
        "id": entry["id"],
        "type": entry.get("type", "unknown"),
        "lang": entry.get("lang", "?"),
        "query": entry["query"],
        "expected_doc_id": expected_doc,
        "doc_rank": doc_rank,
        "doc_art_rank": doc_art_rank,
        "top_score": top_score,
        "top_cosine": top_cosine,
        "expected_max_cosine": expected_max_cosine,
        "expected_max_cosine_art": expected_max_cosine_art,
        "top_hit_doc_id": results[0]["doc_id"] if results else None,
        "n_results": len(results),
    }


def recall_at_k(rows: list[dict], k: int, key: str = "doc_rank") -> float:
    answerable = [r for r in rows if r["expected_doc_id"] is not None]
    if not answerable:
        return float("nan")
    hits = sum(1 for r in answerable if r[key] is not None and r[key] <= k)
    return hits / len(answerable)


def mrr(rows: list[dict], key: str = "doc_rank") -> float:
    answerable = [r for r in rows if r["expected_doc_id"] is not None]
    if not answerable:
        return float("nan")
    total = 0.0
    for r in answerable:
        if r[key] is not None:
            total += 1.0 / r[key]
    return total / len(answerable)


def summarize(rows: list[dict]) -> dict:
    ks = [1, 3, 5, 10]
    summary = {
        "n_queries": len(rows),
        "n_answerable": sum(1 for r in rows if r["expected_doc_id"] is not None),
        "n_unanswerable": sum(1 for r in rows if r["expected_doc_id"] is None),
        "doc_level": {
            f"recall@{k}": round(recall_at_k(rows, k, "doc_rank"), 3) for k in ks
        },
        "doc_article_level": {
            f"recall@{k}": round(recall_at_k(rows, k, "doc_art_rank"), 3) for k in ks
        },
        "mrr_doc_level": round(mrr(rows, "doc_rank"), 3),
        "mrr_doc_article_level": round(mrr(rows, "doc_art_rank"), 3),
    }

    unans = [r for r in rows if r["expected_doc_id"] is None]
    if unans:
        scored = [r["top_score"] for r in unans if r["top_score"] is not None]
        summary["unanswerable_top_score_mean"] = (
            round(sum(scored) / len(scored), 3) if scored else None
        )
        summary["unanswerable_top_score_max"] = (
            round(max(scored), 3) if scored else None
        )

    by_type = {}
    for t in sorted(set(r["type"] for r in rows)):
        sub = [r for r in rows if r["type"] == t]
        if any(r["expected_doc_id"] is not None for r in sub):
            by_type[t] = {
                "n": len(sub),
                "recall@5_doc": round(recall_at_k(sub, 5, "doc_rank"), 3),
                "recall@5_doc_art": round(recall_at_k(sub, 5, "doc_art_rank"), 3),
            }
        else:
            scored = [r["top_score"] for r in sub if r["top_score"] is not None]
            by_type[t] = {
                "n": len(sub),
                "top_score_mean": round(sum(scored) / len(scored), 3) if scored else None,
            }
    summary["by_type"] = by_type

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=MAX_K)
    parser.add_argument("--type", type=str, default=None,
                         help="Only run queries of this type (semantic, exact_reference, unanswerable)")
    parser.add_argument("--lang", type=str, default=None, help="Only run queries in this lang (fr, ar)")
    parser.add_argument("--save", type=str, default=None, help="Path to dump raw per-query results as JSON")
    parser.add_argument("--eval-set", type=str, default=str(EVAL_SET_PATH))
    parser.add_argument(
        "--engine", type=str, choices=["hybrid", "semantic"], default="hybrid",
        help="hybrid = FAISS + BM25 RRF (default), semantic = dense only",
    )
    args = parser.parse_args()

    entries = load_eval_set(Path(args.eval_set))
    if args.type:
        entries = [e for e in entries if e.get("type") == args.type]
    if args.lang:
        entries = [e for e in entries if e.get("lang") == args.lang]

    if not entries:
        print("No queries matched the given filters.")
        return

    print(f"Loading SemanticSearchEngine (this loads the FAISS index + embedder)...")
    engine = SemanticSearchEngine()
    print(f"Running {len(entries)} eval queries at top_k={args.top_k}...\n")

    rows = [run_query(engine, e, top_k=args.top_k, hybrid=(args.engine == "hybrid")) for e in entries]

    for r in rows:
        status = "?"
        if r["expected_doc_id"] is None:
            status = f"(unanswerable) top_score={r['top_score']}"
        elif r["doc_rank"] is not None:
            status = f"doc hit @ rank {r['doc_rank']}" + (
                f", article hit @ rank {r['doc_art_rank']}" if r["doc_art_rank"] else ", article NOT matched"
            )
        else:
            status = "MISS (expected doc never retrieved)"
        print(f"  [{r['id']:>10}] {status}")

    summary = summarize(rows)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save:
        Path(args.save).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved raw results to {args.save}")


if __name__ == "__main__":
    main()
