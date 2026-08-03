"""
src/rag/chatbot.py
-----------------------
Étape RAG : orchestrateur. Enchaîne retrieval (search_engine existant),
construction du prompt (prompt_builder) et génération (llm_client), avec
un garde-fou anti-hallucination basé sur le score de similarité.

Usage minimal :
    from src.rag.chatbot import LegalRAGChatbot

    bot = LegalRAGChatbot()  # charge data/index/ + clé GEMINI_API_KEY
    result = bot.answer("Qui délivre le permis de construire ?")
    print(result["answer"])
    for src in result["sources"]:
        print(src["doc_id"], src["article_number"], src["score"])
"""
from __future__ import annotations

from pathlib import Path

from src.rag.llm_client import LLMClient
from src.rag.prompt_builder import MAX_CONTEXT_CHARS, build_prompt
from src.search_engine.search import DEFAULT_INDEX_DIR, SemanticSearchEngine

DEFAULT_TOP_K = 3

# Seuil de similarité cosinus (embeddings E5 normalisés, cf. embedder.py) en
# dessous duquel on considère qu'aucun résultat n'est assez pertinent pour
# servir de base à une réponse — à recalibrer empiriquement sur ton corpus
# (regarder la distribution des scores sur quelques dizaines de requêtes
# test réelles/hors-sujet pour ajuster cette valeur).
DEFAULT_SCORE_THRESHOLD = 0.55

NO_RESULT_MESSAGE = (
    "Je n'ai pas trouvé d'information suffisamment pertinente dans le corpus "
    "indexé pour répondre à cette question. Essaie de la reformuler, ou "
    "vérifie que le domaine concerné est bien couvert par les documents indexés."
)

NO_RESULT_MESSAGE_AR = (
    "لم أتمكن من العثور على معلومات كافية وذات صلة في الوثائق المفهرسة "
    "للإجابة على هذا السؤال. حاول إعادة صياغته، أو تأكد من أن المجال "
    "المعني مغطى بالوثائق المفهرسة."
)

# Utilisé uniquement quand un historique de conversation est fourni, pour
# reformuler une question de suivi ("et pour les décrets ?") en requête
# autonome exploitable par le retrieval sémantique seul.
REFORMULATION_SYSTEM_INSTRUCTION = (
    "Tu reformules la dernière question d'une conversation en une question "
    "autonome et complète, compréhensible sans le reste de l'historique. "
    "Ne réponds pas à la question, reformule-la uniquement. Renvoie "
    "uniquement la question reformulée, sans commentaire ni guillemets."
)


def _load_doc_unlinked(annotated_dir: str | Path = "data/annotated") -> dict[str, list[dict]]:
    """Load document-level unlinked_tables from enriched JSONs."""
    import json
    doc_unlinked: dict[str, list[dict]] = {}
    for p in sorted(Path(annotated_dir).glob("**/*_entities.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            ul = data.get("unlinked_tables", [])
            if ul:
                doc_id = data.get("doc_id", p.stem)
                doc_unlinked[doc_id] = ul
        except Exception:
            pass
    return doc_unlinked


class LegalRAGChatbot:
    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        embedding_model_name: str | None = None,
        llm_model_name: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ):
        self.search_engine = SemanticSearchEngine(index_dir=index_dir, model_name=embedding_model_name)
        self.llm = LLMClient(model_name=llm_model_name) if llm_model_name else LLMClient()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.doc_unlinked = _load_doc_unlinked()

    def _standalone_query(self, query: str, history: list[dict]) -> str:
        """
        Reformule `query` en question autonome à partir de l'historique
        (liste de {"question": ..., "answer": ...}), pour que le retrieval
        sémantique fonctionne même sur une question de suivi elliptique.
        N'est appelée que si `history` est non vide.
        """
        try:
            # History vient du client (app/chat.py) sans validation de
            # forme : on ne garde que les items valides, et on ne construit
            # la chaîne QUE dans le try — un historique malformé replie sur
            # la question brute au lieu de crasher avant le garde-fou.
            turns = "\n".join(
                f"Q: {t['question']}\nR: {t['answer']}"
                for t in history
                if isinstance(t, dict)
                and isinstance(t.get("question"), str)
                and isinstance(t.get("answer"), str)
                and t["question"]
            )
            if not turns:
                return query
            prompt = (
                f"Historique :\n{turns}\n\n"
                f"Dernière question : {query}\n\nQuestion reformulée :"
            )
            reformulated = self.llm.generate(
                REFORMULATION_SYSTEM_INSTRUCTION, prompt
            ).strip()
        except Exception:
            # Panne LLM (ou historique malformé) : on replie sur la question
            # brute plutôt que de casser toute la requête (l'historique est
            # un bonus, pas une condition).
            return query
        return reformulated or query  # repli sur la question brute si la reformulation échoue

    def answer(
        self,
        query: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
        lang: str | None = None,
    ) -> dict:
        """
        Renvoie {"answer": str, "sources": list[dict], "query_used": str}.
        `sources` est la liste des articles effectivement utilisés comme
        contexte (vide si le garde-fou anti-hallucination s'est déclenché).
        """
        search_query = self._standalone_query(query, history) if history else query

        results = self.search_engine.search(search_query, top_k=top_k or self.top_k, lang=lang)

        # Garde-fou anti-hallucination : appliqué à TOUS les résultats, pas
        # seulement au premier — un hit sous le seuil n'a pas sa place dans
        # le contexte du prompt.
        results = [r for r in results if r["score"] >= self.score_threshold]

        if not results:
            no_result = NO_RESULT_MESSAGE_AR if lang == "ar" else NO_RESULT_MESSAGE
            return {"answer": no_result, "sources": [], "query_used": search_query}

        system_instruction, user_prompt = build_prompt(
            search_query, results, doc_unlinked=self.doc_unlinked, max_context_chars=MAX_CONTEXT_CHARS
        )
        answer_text = self.llm.generate(system_instruction, user_prompt)

        return {"answer": answer_text, "sources": results, "query_used": search_query}