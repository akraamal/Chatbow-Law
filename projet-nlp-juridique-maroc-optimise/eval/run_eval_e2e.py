"""
eval/run_eval_e2e.py
---------------------
Phase 2 eval harness: tests the FULL answer pipeline (chatbot.answer()),
not just retrieval. This is the layer that actually decides whether an
unanswerable question gets refused, and whether an answerable one comes
back with a citation that traces to the RIGHT document -- neither of
which eval/run_eval.py (Phase 1, retrieval-only) can tell you.

Requires a working LLMClient (GROQ_API_KEY set, groq package installed)
-- unlike run_eval.py this makes real LLM calls, so it costs API quota
and takes much longer. Run it far less often than the retrieval-only
harness: after a chatbot.py/prompt_builder.py/citation_verifier.py
change, or before a report/demo, not on every retrieval tweak.

What it measures, per query type:
  answerable (semantic / exact_reference / multi_doc):
    - answered_rate      : did the pipeline produce a real answer at all
                            (vs NO_RESULT_MESSAGE / UNSUPPORTED_SENTENCE
                            / REFUSAL_SENTENCE)?
    - citation_doc_match : of the ones answered, did at least one VERIFIED
                            citation's source doc_id match expected_doc_id?
                            (this is the number that actually matters --
                            an answer with zero traceable-correct citations
                            is not a good answer even if it "looks" right)
    - citation_precision : verified / claimed, averaged over answered queries

  unanswerable:
    - refusal_rate        : did the pipeline correctly refuse?
    - false_answer_texts  : logged verbatim so you can read exactly what
                             it said instead of refusing, if it didn't

A "refusal" is detected structurally (empty sources + citation_stats
claimed=0, OR the canonical refusal/unsupported sentence appears in the
answer), not by string-matching alone -- LLM output phrasing can vary
even when the underlying guard fired correctly.

Usage:
    python -m eval.run_eval_e2e                        # all queries
    python -m eval.run_eval_e2e --type unanswerable     # just the refusal test
    python -m eval.run_eval_e2e --limit 10              # smoke test before a full run
    python -m eval.run_eval_e2e --save eval/e2e_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EVAL_SET_PATH = Path(__file__).parent / "eval_dataset.json"


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"]


def _looks_like_refusal(result: dict, lang: str) -> bool:
    """Structural refusal check, not just string match -- covers both the
    pre-LLM NO_RESULT_MESSAGE path and the post-LLM unsupported/refusal
    sentence paths in chatbot.answer()."""
    from src.rag.chatbot import NO_RESULT_MESSAGE, NO_RESULT_MESSAGE_AR
    from src.rag.prompt_builder import (
        REFUSAL_SENTENCE_FR, REFUSAL_SENTENCE_AR,
        UNSUPPORTED_SENTENCE_FR, UNSUPPORTED_SENTENCE_AR,
    )

    answer = result.get("answer", "")
    canonical = (
        NO_RESULT_MESSAGE_AR if lang == "ar" else NO_RESULT_MESSAGE,
        REFUSAL_SENTENCE_AR if lang == "ar" else REFUSAL_SENTENCE_FR,
        UNSUPPORTED_SENTENCE_AR if lang == "ar" else UNSUPPORTED_SENTENCE_FR,
    )
    if any(c in answer for c in canonical if c):
        return True

    stats = result.get("citation_stats", {})
    if not result.get("sources") and stats.get("claimed", 0) == 0:
        return True

    return False


def run_query(bot, entry: dict) -> dict:
    lang = entry.get("lang", "fr")
    t0 = time.time()
    try:
        result = bot.answer(entry["query"], lang=lang)
        error = None
    except Exception as e:
        result = {"answer": "", "sources": [], "citations": [], "citation_stats": {}}
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    refused = _looks_like_refusal(result, lang)
    expected_doc = entry.get("expected_doc_id")

    cited_doc_ids = {c.get("doc_id") for c in result.get("citations", []) if isinstance(c, dict)}
    # Fall back to sources if citation dicts don't carry doc_id directly --
    # adapt this line if your verify_citations() return shape differs.
    if not cited_doc_ids:
        cited_doc_ids = {s.get("doc_id") for s in result.get("sources", []) if isinstance(s, dict)}

    citation_doc_match = bool(expected_doc) and expected_doc in cited_doc_ids

    stats = result.get("citation_stats", {}) or {}
    claimed = stats.get("claimed", 0)
    verified = stats.get("verified", 0)
    precision = (verified / claimed) if claimed else None

    return {
        "id": entry["id"],
        "type": entry.get("type", "unknown"),
        "lang": lang,
        "query": entry["query"],
        "expected_doc_id": expected_doc,
        "refused": refused,
        "answer_text": result.get("answer", ""),
        "cited_doc_ids": sorted(d for d in cited_doc_ids if d),
        "citation_doc_match": citation_doc_match,
        "claimed": claimed,
        "verified": verified,
        "citation_precision": precision,
        "elapsed_s": round(elapsed, 2),
        "error": error,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {"n_queries": len(rows)}

    unans = [r for r in rows if r["expected_doc_id"] is None]
    ans = [r for r in rows if r["expected_doc_id"] is not None]

    if unans:
        summary["unanswerable"] = {
            "n": len(unans),
            "refusal_rate": round(sum(r["refused"] for r in unans) / len(unans), 3),
            "false_answers": [
                {"id": r["id"], "query": r["query"], "answer": r["answer_text"]}
                for r in unans if not r["refused"]
            ],
        }

    if ans:
        answered = [r for r in ans if not r["refused"]]
        precisions = [r["citation_precision"] for r in answered if r["citation_precision"] is not None]
        summary["answerable"] = {
            "n": len(ans),
            "answered_rate": round(len(answered) / len(ans), 3),
            "citation_doc_match_rate": round(
                sum(r["citation_doc_match"] for r in answered) / len(answered), 3
            ) if answered else None,
            "mean_citation_precision": round(sum(precisions) / len(precisions), 3) if precisions else None,
            "unexpectedly_refused": [r["id"] for r in ans if r["refused"]],
        }

        by_type = {}
        for t in sorted(set(r["type"] for r in ans)):
            sub = [r for r in ans if r["type"] == t]
            sub_answered = [r for r in sub if not r["refused"]]
            by_type[t] = {
                "n": len(sub),
                "answered_rate": round(len(sub_answered) / len(sub), 3) if sub else None,
                "citation_doc_match_rate": round(
                    sum(r["citation_doc_match"] for r in sub_answered) / len(sub_answered), 3
                ) if sub_answered else None,
            }
        summary["answerable"]["by_type"] = by_type

    errors = [r["id"] for r in rows if r.get("error")]
    if errors:
        summary["errors"] = errors

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N matching queries (smoke test)")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--eval-set", type=str, default=str(EVAL_SET_PATH))
    args = parser.parse_args()

    entries = load_eval_set(Path(args.eval_set))
    if args.type:
        entries = [e for e in entries if e.get("type") == args.type]
    if args.lang:
        entries = [e for e in entries if e.get("lang") == args.lang]
    if args.limit:
        entries = entries[: args.limit]

    if not entries:
        print("No queries matched the given filters.")
        return

    from src.rag.chatbot import LegalRAGChatbot

    print(f"Loading LegalRAGChatbot (index + LLM client)...")
    bot = LegalRAGChatbot()
    print(f"Running {len(entries)} end-to-end queries (this calls the real LLM -- may take a while)...\n")

    rows = []
    for i, e in enumerate(entries, start=1):
        r = run_query(bot, e)
        rows.append(r)
        tag = "REFUSED" if r["refused"] else "ANSWERED"
        extra = ""
        if r["expected_doc_id"] is not None and not r["refused"]:
            extra = " [doc match]" if r["citation_doc_match"] else " [NO DOC MATCH]"
        if r["error"]:
            extra += f" ERROR: {r['error']}"
        print(f"  [{i}/{len(entries)}] {r['id']:>10} -> {tag}{extra} ({r['elapsed_s']}s)")

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
