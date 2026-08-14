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

from src.rag.citation_verifier import parse_citations, verify_citations
from src.rag.llm_client import LLMClient
from src.rag.prompt_builder import (
    REFUSAL_SENTENCE_AR,
    REFUSAL_SENTENCE_FR,
    UNSUPPORTED_SENTENCE_AR,
    UNSUPPORTED_SENTENCE_FR,
    MAX_CONTEXT_CHARS,
    build_prompt,
    build_catalog_prompt,
)
from src.rag.query_routing import route_query
from src.search_engine.search import DEFAULT_INDEX_DIR, SemanticSearchEngine

DEFAULT_TOP_K = 3

# Seuil de similarité cosinus (embeddings E5 normalisés, cf. embedder.py) en
# dessous duquel on considère qu'un résultat est trop éloigné pour figurer
# dans le contexte du prompt (filtre de QUALITÉ du contexte, pas garde-fou
# anti-hallucination).
#
# Historique de la calibration (2026-08-03, 24 requêtes labelisées, index
# 1161 docs, intfloat/multilingual-e5-base) :
#   - scores top-1 des requêtes pertinentes : min 0.819 / médiane 0.833 / max 0.844
#   - scores top-1 des requêtes hors-sujet  : min 0.777 / médiane 0.801 / max 0.818
#   - seuil 0.82  → recall 11/12, fp 0/12 (F1 0.957) MAIS bloc dés aussi la
#     plupart des questions réelles (scores saturant 0.78-0.82 sur un corpus
#     juridique homogène, même pour des questions pertinentes).
#
# Politique actuelle : le garde-fou anti-hallucination est le VÉRIFICATEUR
# DE CITATIONS (a) sortie) — une réponse sans citation vérifiée mécaniquement
# est refusée (voir answer()). Le seuil n'est donc plus qu'un filtre de bruit
# de contexte : 0.75 coupe les chunks manifestement hors-sujet tout en
# laissant passer les questions réelles (recall élevé), le refus explicite
# en aval gardant le contrôle du risque.
DEFAULT_SCORE_THRESHOLD = 0.75

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


def _load_catalog(index_dir: str) -> list[dict] | None:
    """
    Charge le catalogue d'instruments (data/index/catalog.json) ; s'il
    n'existe pas, le construit depuis data/annotated et le persiste.
    Retourne None (et ne bloque jamais le chatbot) en cas de problème.
    """
    try:
        from src.search_engine.catalog import build_catalog, load_catalog, save_catalog

        catalog = load_catalog(index_dir)
        if catalog is None:
            catalog = build_catalog()
            if catalog:
                save_catalog(catalog, index_dir)
        return catalog or None
    except Exception:
        return None


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
        self.catalog = _load_catalog(index_dir)
        self.catalog_top_n = 8

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
        Renvoie {"answer": str, "sources": list[dict], "citations": list[dict],
        "citation_stats": dict, "query_used": str, "mode": "catalog"|None}.

        Les questions agrégées (« les dahirs les plus importants », « les
        décrets de 2024 », « combien d'articles comporte le décret n° X ? »)
        sont aiguillées vers le catalogue d'instruments (mode "catalog") ;
        les autres suivent la recherche sémantique classique. Dans les
        deux cas, les citations du LLM sont extraites du bloc [[CITATIONS]]
        et VÉRIFIÉES mécaniquement contre le texte des sources récupérées
        (src/rag/citation_verifier.py). Une citation qui ne se retrouve pas
        mot à mot dans sa source est silencieusement supprimée — rien
        d'invérifiable n'est jamais montré. Si le LLM a produit des
        citations mais qu'aucune ne passe la vérification, la réponse est
        remplacée par un refus explicite.
        """
        search_query = self._standalone_query(query, history) if history else query

        # Chemin « catalogue » : questions agrégées / par référence
        # (« les dahirs les plus importants », « les décrets de 2024 »,
        # « combien d'articles comporte le décret n° 2-25-1080 ? »).
        # Se replie silencieusement sur le chemin sémantique si le catalogue
        # n'est pas disponible ou ne donne rien.
        route = route_query(search_query, lang)
        catalog_hits = None
        if route.get("catalog") and self.catalog:
            from src.search_engine.catalog import search_catalog

            catalog_hits = search_catalog(
                self.catalog,
                search_query,
                type_filter=route.get("type") or None,
                year=route.get("year") or None,
                lang=lang,
                top_n=self.catalog_top_n,
            )

        if catalog_hits:
            system_instruction, user_prompt = build_catalog_prompt(search_query, catalog_hits)
            answer_text = self.llm.generate_with_citation_guarantee(system_instruction, user_prompt)

            clean_answer, cited_spans = parse_citations(answer_text)
            verified_citations, citation_stats = verify_citations(cited_spans, catalog_hits)

            if cited_spans and not verified_citations and clean_answer.strip():
                unsupported = UNSUPPORTED_SENTENCE_AR if lang == "ar" else UNSUPPORTED_SENTENCE_FR
                return {
                    "answer": unsupported,
                    "sources": [],
                    "citations": [],
                    "citation_stats": citation_stats,
                    "query_used": search_query,
                    "mode": "catalog",
                }

            return {
                "answer": clean_answer,
                "sources": catalog_hits,
                "citations": verified_citations,
                "citation_stats": citation_stats,
                "query_used": search_query,
                "mode": "catalog",
            }

        results = self.search_engine.search(search_query, top_k=top_k or self.top_k, lang=lang)

        # Filtre de QUALITÉ du contexte : appliqué à TOUS les résultats, pas
        # seulement au premier — un hit sous le seuil n'a pas sa place dans
        # le contexte du prompt.
        #
        # IMPORTANT : on filtre sur cosine_score, PAS sur score. Depuis le
        # passage à la recherche hybride (FAISS+BM25 fusionnés par RRF),
        # "score" est un score RRF (échelle ~0.01-0.03, dérivé des rangs,
        # non comparable à un cosinus) utilisé uniquement pour le classement
        # des résultats. Le filtre de qualité du contexte doit rester sur la
        # similarité cosinus brute, seule échelle pour laquelle
        # self.score_threshold (0.75) a été calibré. Un hit trouvé
        # uniquement par BM25 (jamais présent dans le pool dense) a
        # cosine_score=None : on le rejette aussi, faute de signal de
        # similarité pour le juger. Le vrai garde-fou anti-hallucination
        # reste le vérificateur de citations en aval (cf. answer()).
        results = [
            r for r in results
            if r.get("cosine_score") is not None and r["cosine_score"] >= self.score_threshold
        ]

        if not results:
            no_result = NO_RESULT_MESSAGE_AR if lang == "ar" else NO_RESULT_MESSAGE
            return {
                "answer": no_result,
                "sources": [],
                "citations": [],
                "citation_stats": {"claimed": 0, "verified": 0, "failed": 0},
                "query_used": search_query,
            }

        system_instruction, user_prompt = build_prompt(
            search_query, results, doc_unlinked=self.doc_unlinked, max_context_chars=MAX_CONTEXT_CHARS
        )
        # Garantit l'émission d'un bloc [[CITATIONS]] vérifiable : si le
        # modèle configuré est un "reasoning" qui n'émet jamais ce bloc
        # (ex. qwen/qwen3.6-27b), on régénère une fois avec le modèle citant
        # fallback — sinon le garde-fou anti-hallucination reste aveugle.
        answer_text = self.llm.generate_with_citation_guarantee(system_instruction, user_prompt)

        # 1. Extraire le bloc de citations du texte brut du LLM
        clean_answer, cited_spans = parse_citations(answer_text)

        # 2. Vérifier mécaniquement chaque citation contre la source réelle
        verified_citations, citation_stats = verify_citations(cited_spans, results)

        # 3. Garde-fou anti-hallucination « verrou à la sortie » : avec un
        #    seuil de retrieval bas (DEFAULT_SCORE_THRESHOLD = 0.75), la
        #    réponse n'est conservée QUE si elle est adossée à au moins une
        #    citation VÉRIFIÉE mécaniquement. Une réponse de fond sans aucune
        #    citation vérifiée est remplacée par un refus explicite — sauf si
        #    le LLM a lui-même refusé avec la phrase canonique (déjà sûre).
        if clean_answer.strip() and not verified_citations:
            refusal = REFUSAL_SENTENCE_AR if lang == "ar" else REFUSAL_SENTENCE_FR
            if refusal not in clean_answer:
                unsupported = UNSUPPORTED_SENTENCE_AR if lang == "ar" else UNSUPPORTED_SENTENCE_FR
                return {
                    "answer": unsupported,
                    "sources": [],
                    "citations": [],
                    "citation_stats": citation_stats,
                    "query_used": search_query,
                }

        return {
            "answer": clean_answer,
            "sources": results,
            "citations": verified_citations,
            "citation_stats": citation_stats,
            "query_used": search_query,
        }