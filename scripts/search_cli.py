"""
search_cli.py
----------------
Étape 6 : interroge l'index de recherche sémantique déjà construit
(voir scripts/build_search_index.py).

Usage :
    python -m scripts.search_cli "licence de télécommunications"
    python -m scripts.search_cli "رخصة البناء" --lang ar --top-k 3
    python -m scripts.search_cli   # sans argument : mode interactif
"""
import argparse

from src.search_engine.search import SemanticSearchEngine


def print_results(results: list[dict]) -> None:
    if not results:
        print("Aucun résultat.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.3f} | {r['doc_id']} — article {r['article_number']} ({r['lang']})")
        snippet = r["text"][:300].replace("\n", " ")
        print(f"    {snippet}...")


def run_query(engine: SemanticSearchEngine, query: str, top_k: int, lang: str | None) -> None:
    results = engine.search(query, top_k=top_k, lang=lang)
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interroge l'index de recherche sémantique (étape 6).")
    parser.add_argument("query", type=str, nargs="?", default=None,
                         help="Requête à rechercher. Si absente, passe en mode interactif.")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de résultats (défaut: 5)")
    parser.add_argument("--lang", type=str, choices=["fr", "ar"], default=None,
                         help="Filtrer par langue (défaut: les deux)")
    parser.add_argument("--index-dir", type=str, default="data/index",
                         help="Répertoire de l'index (défaut: data/index)")
    args = parser.parse_args()

    engine = SemanticSearchEngine(index_dir=args.index_dir)
    print(f"Index chargé : {engine.index.ntotal} article(s), modèle {engine.embedder.model_name}\n")

    if args.query:
        run_query(engine, args.query, args.top_k, args.lang)
    else:
        print("Mode interactif — tape une requête (ou 'quit' pour sortir) :")
        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            run_query(engine, query, args.top_k, args.lang)
