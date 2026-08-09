"""
run_rag_pipeline.py
--------------------
End-to-end RAG orchestration: load enriched JSONs → consolidate → build FAISS
index → (optional) run a query or start an interactive chat.

Usage:
    # Build index from all enriched JSONs in data/annotated/
    python -m scripts.run_rag_pipeline --build-index

    # Query
    python -m scripts.run_rag_pipeline --query "Qui délivre le permis de construire ?"

    # Interactive chat
    python -m scripts.run_rag_pipeline --chat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ANNOTATED_DIR = Path("data/annotated")
INDEX_DIR = Path("data/index")


def _load_all_enriched_jsons() -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Load all enriched JSONs from data/annotated/ into a flat article list
    and a parallel doc_unlinked_tables dict keyed by doc_id.
    """
    articles = []
    doc_unlinked: dict[str, list[dict]] = {}
    for p in sorted(ANNOTATED_DIR.glob("**/*_entities.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data.get("doc_id", p.stem)
        bo_number = data.get("bo_number", "")
        date_pub = data.get("date_publication", "")
        lang = data.get("lang", "fr")

        # Capture document-level unlinked tables
        ul = data.get("unlinked_tables", [])
        if ul:
            doc_unlinked[doc_id] = ul

        for art in data.get("articles", []):
            articles.append({
                "article_id": art.get("article_id", ""),
                "doc_id": doc_id,
                "bo_number": bo_number,
                "date_publication": date_pub,
                "lang": lang,
                "article_number": art.get("number", ""),
                "text": art.get("text", ""),
                "text_clean": art.get("text_clean", ""),
                "instrument_type": art.get("instrument_type", ""),
                "reference": art.get("reference", ""),
                "pdf_page": art.get("pdf_page"),
                "printed_page": art.get("printed_page"),
                "extracted_tables": art.get("extracted_tables", []),
                "entities": art.get("entities", []),
                "dates": art.get("dates", []),
                "citations": art.get("citations", []),
            })

        # Also add articles nested inside instruments (if present)
        for instr in data.get("instruments", []):
            instr_type = instr.get("instrument_type", "")
            instr_ref = instr.get("reference", "")
            for art in instr.get("articles", []):
                articles.append({
                    "article_id": art.get("article_id", ""),
                    "doc_id": doc_id,
                    "bo_number": bo_number,
                    "date_publication": date_pub,
                    "lang": lang,
                    "article_number": art.get("number", ""),
                    "text": art.get("text", ""),
                    "text_clean": art.get("text_clean", ""),
                    "instrument_type": instr_type,
                    "reference": instr_ref,
                    "pdf_page": art.get("pdf_page"),
                    "printed_page": art.get("printed_page"),
                    "extracted_tables": art.get("extracted_tables", []),
                    "entities": art.get("entities", []),
                    "dates": art.get("dates", []),
                    "citations": art.get("citations", []),
                })

    return articles, doc_unlinked


def build_index():
    """Build FAISS index from enriched JSON data."""
    from src.search_engine.index_builder import build_index as _build
    from src.search_engine.index_builder import DEFAULT_INDEX_DIR

    print("Loading enriched JSONs ...")
    articles, doc_unlinked = _load_all_enriched_jsons()
    print(f"  {len(articles)} articles loaded from {ANNOTATED_DIR}")

    index_dir = Path(DEFAULT_INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)

    _build(articles, str(index_dir))
    print(f"  Index saved to {index_dir}/")
    n_ul = sum(len(v) for v in doc_unlinked.values())
    if n_ul:
        print(f"  {n_ul} unlinked tables loaded from {len(doc_unlinked)} document(s)")


def run_query(query: str, top_k: int = 5, lang: str | None = None):
    """Run a single query against the RAG pipeline."""
    from src.rag.chatbot import LegalRAGChatbot

    print(f"Query: {query}")
    bot = LegalRAGChatbot()
    result = bot.answer(query, top_k=top_k, lang=lang)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for src in result["sources"]:
        print(f"  [{src['doc_id']}] Art. {src.get('article_number','?')} "
              f"(score={src['score']:.2f}, page={src.get('pdf_page','?')})")
    if result.get("sources") and result.get("query_used") != query:
        print(f"\n(reformulé depuis : {result['query_used']})")


def interactive_chat():
    """Start an interactive RAG chat session."""
    from src.rag.chatbot import LegalRAGChatbot

    bot = LegalRAGChatbot()
    history = []
    print("RAG Chatbot Juridique Marocain (tape 'quit' pour quitter)")
    print("-" * 50)
    while True:
        try:
            query = input("\nQuestion > ").strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        result = bot.answer(query, history=history if history else None)
        print(f"\n{result['answer']}")
        if result["sources"]:
            print(f"\nSources:")
            for s in result["sources"]:
                print(f"  [{s['doc_id']}] Art. {s.get('article_number','?')} "
                      f"(score={s['score']:.2f})")
        history.append({"question": query, "answer": result["answer"]})


def main():
    parser = argparse.ArgumentParser(
        description="RAG pipeline: index enriched JSONs and answer queries."
    )
    parser.add_argument("--build-index", action="store_true",
                        help="Build FAISS index from enriched JSONs")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to retrieve (default: 5)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Filter by language: 'fr' or 'ar'")
    args = parser.parse_args()

    if args.build_index:
        build_index()
    elif args.query:
        run_query(args.query, top_k=args.top_k, lang=args.lang)
    elif args.chat:
        interactive_chat()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
