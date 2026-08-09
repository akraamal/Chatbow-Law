"""
test_context_table_clean.py
----------------------------
Tests du "gap text_clean-in-context" :
1. format_context doit utiliser text_clean (texte + tableaux linéarisés)
   quand disponible, sinon le LLM ne voit pas le contenu des tableaux.
2. citation_verifier.verify_citations doit vérifier les citations du tableau
   contre text_clean (le texte brut ne contient pas les valeurs).

Usage :
    python -m pytest tests/test_context_table_clean.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.citation_verifier import verify_citations
from src.rag.prompt_builder import format_context

RAW_TEXT = (
    "Les montants de la taxe sur le gaz sont fixés à l'article 936 du CGI.\n"
    "Les redevables s'acquittent du montant indiqué."
)

CLEAN_TEXT = RAW_TEXT + (
    "\n\n[Tableau : Tarifs de la taxe gaz]\n"
    "1. Conditionnement: Frais et marge des sociétés de distribution | "
    "Charges supérieure à 5 kg (DH/TM): 553,00\n"
    "2. Conditionnement: Ventes directes | Charges inférieures à 5 kg (DH/TM): "
    "619,00"
)

ARTICLE = {
    "article_id": "art_936_26_1",
    "doc_id": "ARRETE_936_26",
    "article_number": "936",
    "bo_number": "7510",
    "lang": "fr",
    "reference": "936-26",
    "instrument_type": "Arrêté",
    "text": RAW_TEXT,
    "text_clean": CLEAN_TEXT,
    "extracted_tables": [
        {
            "table_id": "art_936_26_1::tbl_0",
            "page": 31,
            "caption": "Tarant gaz",
            "headers": ["Conditionnement", "DH/TM"],
            "rows": [["Frais et marge sociétés de distribution", "553,00"],
                     ["Ventes", "619,00"]],
        }
    ],
}


def test_format_context_uses_text_clean_with_table_block():
    ctx = format_context([ARTICLE], max_context_chars=4096)
    assert "[Tableau : Tarifs de la taxe gaz]" in ctx
    assert "553,00" in ctx
    assert "619,00" in ctx


def test_format_context_keeps_table_block_when_narrative_truncated():
    big_narrative = RAW_TEXT + "\n" + ("détail sans fin. " * 500)
    art = {**ARTICLE, "text_clean": big_narrative + CLEAN_TEXT[len(RAW_TEXT):]}
    ctx = format_context([art], max_context_chars=2048)
    # la partie narrative est tronquée mais le bloc tableau est conservé
    assert "[Tableau : Tarifs de la taxe gaz]" in ctx
    assert "619,00" in ctx


def test_format_context_no_double_table_summary_when_clean():
    ctx = format_context([ARTICLE], max_context_chars=4096)
    # avec text_clean, le résumé produit par extracted_tables ne doit plus
    # être dupliqué
    assert ctx.count("Ce document contient") == 0


def test_verify_citations_uses_text_clean_for_table_values():
    chunk_dict = {
        "chunk_id": "c1",
        "doc_id": "ARRETE_936_26",
        "article_number": "936",
        "lang": "fr",
        "text": RAW_TEXT,          # pas la valeur du tableau
        "text_clean": CLEAN_TEXT,  # contient la valeur
    }
# citation reproduite VERBATIM par le LLM depuis text_clean
    # (le LLM omet le préfixe "1. " de la ligne de tableau linéarisée)
    spans = [
        {
            "quote": "Conditionnement: Frais et marge des sociétés de "
                     "distribution | Charges supérieure à 5 kg (DH/TM): 553,00",
            "source": 1,
        }
    ]
    verified, stats = verify_citations(spans, [chunk_dict])
    assert stats["verified"] == 1
    assert stats["failed"] == 0
    assert len(verified) == 1
    assert verified[0]["matched_quote"] == spans[0]["quote"]