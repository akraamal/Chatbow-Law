"""
rag_chat_cli.py
-------------------
Teste LegalRAGChatbot (src/rag/chatbot.py) en terminal, avant de le
brancher à l'interface Streamlit.

Prérequis :
  - Index FAISS déjà construit : python -m scripts.build_search_index
  - Variable d'environnement GEMINI_API_KEY définie (clé gratuite sur
    https://aistudio.google.com/apikey)

Usage :
    python -m scripts.rag_chat_cli "Qui délivre le permis de construire ?"
    python -m scripts.rag_chat_cli --lang ar
    python -m scripts.rag_chat_cli   # mode interactif, avec historique
"""
import argparse
import sys

from dotenv import load_dotenv
load_dotenv()

from src.rag.chatbot import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K, LegalRAGChatbot


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("Sources utilisées :")
        for i, src in enumerate(result["sources"], start=1):
            snippet = src["text"][:150].replace("\n", " ")
            print(
                f"  [{i}] {src['doc_id']} — article {src['article_number']} "
                f"({src['lang']}, score={src['score']:.3f})"
            )
            print(f"      {snippet}...")


def run_single_query(bot: LegalRAGChatbot, query: str, lang: str | None) -> None:
    result = bot.answer(query, lang=lang)
    print_result(result)


def run_interactive(bot: LegalRAGChatbot, lang: str | None) -> None:
    print("Mode interactif — tape une question (ou 'quit' pour sortir).")
    print("L'historique est conservé pour permettre les questions de suivi.\n")
    history: list[dict] = []
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        result = bot.answer(query, history=history, lang=lang)
        print_result(result)

        # N'accumule dans l'historique que les échanges avec une vraie
        # réponse (pas les "pas d'info trouvée"), pour ne pas polluer les
        # reformulations suivantes avec un tour sans contenu utile.
        if result["sources"]:
            history.append({"question": query, "answer": result["answer"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste le chatbot RAG juridique en terminal.")
    parser.add_argument("query", type=str, nargs="?", default=None,
                         help="Question à poser. Si absente, passe en mode interactif.")
    parser.add_argument("--lang", type=str, choices=["fr", "ar"], default=None,
                         help="Filtrer la recherche par langue (défaut: les deux)")
    parser.add_argument("--index-dir", type=str, default="data/index",
                         help="Répertoire de l'index (défaut: data/index)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                         help=f"Nombre d'articles récupérés (défaut: {DEFAULT_TOP_K})")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                         help=f"Seuil anti-hallucination (défaut: {DEFAULT_SCORE_THRESHOLD})")
    args = parser.parse_args()

    try:
        bot = LegalRAGChatbot(
            index_dir=args.index_dir,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Erreur : {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Chatbot prêt — modèle génération : {bot.llm.model_name}, "
          f"index : {bot.search_engine.index.ntotal} article(s)\n")

    if args.query:
        run_single_query(bot, args.query, args.lang)
    else:
        run_interactive(bot, args.lang)