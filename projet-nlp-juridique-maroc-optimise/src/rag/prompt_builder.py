"""
src/rag/prompt_builder.py
------------------------------
Étape RAG (augmentation) : assemble le contexte récupéré par
SemanticSearchEngine.search() (voir src/search_engine/search.py) en un
prompt exploitable par le LLM.

Ne fait aucun appel réseau — pure construction de texte, facilement
testable de façon unitaire (contrairement à llm_client.py).
"""
from __future__ import annotations

SYSTEM_INSTRUCTION = """\
Tu es un assistant juridique spécialisé dans le droit marocain. Tu réponds \
UNIQUEMENT à partir des extraits du Bulletin Officiel fournis dans le \
contexte ci-dessous. Tu n'utilises JAMAIS tes connaissances propres.

RÈGLES ABSOLUES (leur non-respect rend la réponse irrecevable) :

1. [SOURCES] Chaque affirmation doit être suivie de son numéro de source \
entre crochets, comme [Source 1]. Exemple correct : « Le décret n° 2.25.559 \
concerne la reconnaissance d'État de l'école ESSEM [Source 3]. »

2. [DATES] Ne mentionne JAMAIS une date qui n'apparaît pas textuellement \
dans le contexte. Si un décret mentionne sa date dans le texte (par ex. \
« Fait à Rabat, le 14 chaoual 1447 (2 avril 2026) »), utilise cette date \
textuelle. N'invente jamais une date, même partiellement.

3. [NUMÉROS] Quand tu cites un numéro de décret, de dahir ou d'arrêté, \
vérifie qu'il est exactement identique dans le texte source. Ne confonds \
jamais deux numéros différents : chaque numéro de décret correspond à un \
contenu spécifique.

4. [HONNÊTETÉ] Si le contexte ne contient pas l'information demandée, \
réponds : « Le contexte fourni ne permet pas de répondre à cette question. » \
N'invente jamais une information qui ne figure pas textuellement dans les \
extraits.

5. [LANGUE] Réponds dans la même langue que la question posée.
"""

# Budget grossier en caractères (≈4 caractères/token pour du FR/AR mixte).
# Objectif : rester nettement sous les limites TPM des providers (ex. Groq
# on_demand tier = 8000 TPM), marge gardée pour system_instruction, la
# question, et les tokens de sortie (max_output_tokens dans llm_client.py).
MAX_CONTEXT_CHARS = 9000
# Plancher par article : même si on doit couper fort pour tenir le budget
# total, on garde au moins ce nombre de caractères par source pour que la
# réponse reste exploitable (citations, numéros de décret, etc.).
MIN_ARTICLE_CHARS = 600


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[...texte tronqué...]"


def format_context(
    articles: list[dict],
    doc_unlinked: dict[str, list[dict]] | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Formate les articles récupérés (résultats de SemanticSearchEngine.search
    ou articles enrichis avec pages/instruments/tables) en un bloc de
    contexte numéroté, dans l'ordre de pertinence.
    
    Inclut les champs enrichis quand ils sont disponibles : instrument_type,
    reference, pdf_page, printed_page, extracted_tables.
    
    doc_unlinked: document-level unlinked_tables (tables that couldn't be
    linked to a specific article) keyed by doc_id.

    max_context_chars: budget total (caractères) pour le bloc de contexte.
    Le texte de chaque article est tronqué à parts égales (plancher
    MIN_ARTICLE_CHARS) pour tenir dans ce budget — évite les 413 "request
    too large" côté LLM quand plusieurs articles longs sont récupérés.
    """
    if not articles:
        return "(Aucun extrait pertinent trouvé.)"

    # Budget par article, réparti équitablement, jamais sous le plancher.
    per_article_budget = max(MIN_ARTICLE_CHARS, max_context_chars // max(len(articles), 1))

    blocks = []
    seen_doc_ids: set[str] = set()
    for i, article in enumerate(articles, start=1):
        # Enriched fields
        instr_type = article.get("instrument_type", "")
        ref = article.get("reference", "")
        pdf_page = article.get("pdf_page")
        printed_page = article.get("printed_page")
        tables = article.get("extracted_tables", [])
        art_num = article.get("article_number", article.get("number", "?"))

        parts = [f"[Source {i}]"]
        bo = article.get("bo_number", "?")
        parts.append(f"Bulletin Officiel n°{bo}")
        doc = article.get("doc_id", "?")
        parts.append(f"— {doc}")
        parts.append(f", article {art_num}")
        if instr_type:
            parts.append(f" ({instr_type}")
            if ref:
                parts.append(f" n°{ref}")
            parts.append(")")
        if pdf_page:
            page_str = f", pdf_page={pdf_page}"
            if printed_page:
                page_str += f" (BO p.{printed_page})"
            parts.append(page_str)

        meta = ", ".join(parts)
        text = _truncate(article.get("text", ""), per_article_budget)

        # Append table summary if tables are linked
        table_summary = ""
        if tables:
            n_tables = len(tables)
            total_rows = sum(t.get("n_rows", 0) for t in tables)
            total_cols = max((t.get("n_cols", 0) for t in tables), default=0)
            table_summary = (
                f"\n[Ce document contient {n_tables} tableau(x) "
                f"({total_rows} lignes × {total_cols} colonnes) sur "
                f"la même page. Extrait des premières lignes :\n"
            )
            for ti, t in enumerate(tables[:2], 1):
                header = t.get("rows", [[]])[0]
                table_summary += f"  Tableau {ti}: {' | '.join(str(c) for c in header[:4])}\n"
            if n_tables > 2:
                table_summary += f"  ... et {n_tables - 2} tableau(x) supplémentaire(s).\n"
            table_summary += "]"

        # Track seen doc_ids so we can append unlinked tables per doc
        if doc:
            seen_doc_ids.add(doc)

        blocks.append(f"{meta}\n{text}{table_summary}")

    # Append unlinked tables from any doc_id that appeared in results.
    # Cap le nombre de documents traités ici : avec top_k élevé, les articles
    # récupérés peuvent venir de plusieurs BO différents, et ce bloc est en
    # plus du texte des articles eux-mêmes — sans cap il peut à lui seul
    # dépasser le budget total sur une requête avec beaucoup de tableaux.
    MAX_UNLINKED_DOCS = 2
    if doc_unlinked:
        for doc_id in list(seen_doc_ids)[:MAX_UNLINKED_DOCS]:
            ul = doc_unlinked.get(doc_id, [])
            if not ul:
                continue
            ul_summary = (
                f"\n[Tableaux non liés du document {doc_id} — ils figurent dans "
                f"le PDF mais n'ont pu être rattachés à un article spécifique :\n"
            )
            for ti, t in enumerate(ul[:3], 1):
                header = t.get("rows", [[]])[0]
                ul_summary += f"  Tableau {ti}: page {t.get('page_number','?')}, "
                ul_summary += f"{' | '.join(str(c)[:60] for c in header[:3])}\n"
            n_remaining = len(ul) - 3
            if n_remaining > 0:
                ul_summary += f"  ... et {n_remaining} tableau(x) supplémentaire(s).\n"
            ul_summary += "]"
            blocks.append(ul_summary)

    full_context = "\n\n---\n\n".join(blocks)
    # Filet de sécurité final : même avec les budgets par article ci-dessus,
    # des métadonnées verbeuses (tableaux, references) sur beaucoup de
    # sources pourraient dépasser le budget total. On coupe en dernier
    # recours plutôt que d'envoyer un prompt trop gros au LLM.
    hard_cap = max_context_chars + 2000  # marge pour métadonnées/tableaux
    if len(full_context) > hard_cap:
        full_context = full_context[:hard_cap].rstrip() + "\n\n[...contexte tronqué...]"
    return full_context


def build_user_prompt(
    query: str,
    articles: list[dict],
    doc_unlinked: dict[str, list[dict]] | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Assemble la question de l'utilisateur avec le contexte récupéré, prêt à
    être envoyé comme `user_prompt` à LLMClient.generate().
    """
    context = format_context(articles, doc_unlinked=doc_unlinked, max_context_chars=max_context_chars)
    return (
        f"Contexte (extraits du Bulletin Officiel) :\n\n{context}\n\n"
        f"---\n\n"
        f"Question : {query}"
    )


def build_prompt(
    query: str,
    articles: list[dict],
    doc_unlinked: dict[str, list[dict]] | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, str]:
    """
    Point d'entrée principal, utilisé par chatbot.py.
    Renvoie (system_instruction, user_prompt) prêts pour LLMClient.generate().
    """
    return SYSTEM_INSTRUCTION, build_user_prompt(
        query, articles, doc_unlinked=doc_unlinked, max_context_chars=max_context_chars
    )