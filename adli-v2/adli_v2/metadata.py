"""
adli_v2.metadata
----------------
Bloc de métadonnées par document et compteurs de mots-clés (étape v2
exécutée APRÈS l'enrichissement v1).  Distingue explicitement :
  - date_parution   → date de parution du BO (en-tête du bulletin) ;
  - decree_date_*   → dates propres à chaque instrument (déjà dans v1).
"""

from __future__ import annotations

import json
from pathlib import Path

from adli_v2.keyword_counter import count_keywords


def build_document_metadata(data: dict) -> dict:
    """Bloc metadata : {doc_name, lang, bo_number (+fiabilité), date_parution,
    edition_label, n_articles, n_instruments, total_pdf_pages}."""
    instruments = data.get("instruments") or []
    return {
        "doc_id": data.get("doc_id"),
        "doc_name": Path(data.get("doc_id", "")).stem,
        "lang": data.get("lang"),
        "bo_number": data.get("bo_number"),
        "bo_number_source": data.get("bo_number_source"),
        "bo_number_confidence": data.get("bo_number_confidence"),
        "date_parution": (
            data.get("bo_date_publication") or data.get("date_publication")
        ),
        "edition_label": data.get("edition_label"),
        "n_articles": len(data.get("articles") or []),
        "n_instruments": len(instruments),
        "total_pdf_pages": data.get("total_pdf_pages"),
    }


def document_text(data: dict) -> str:
    """Texte entier du document (préambule + tous les articles) pour le
    comptage au niveau bulletin."""
    parts = [data.get("preamble_text") or ""]
    parts += [a.get("text") or "" for a in data.get("articles") or []]
    return "\n".join(parts)


def instrument_text(instr: dict, articles: list[dict]) -> str:
    """Texte d'un instrument : son champ `content` (préambule + articles),
    sinon la concaténation de ses articles via article_indices."""
    if instr.get("content"):
        return instr["content"]
    indices = instr.get("article_indices") or []
    return "\n".join(
        articles[i].get("text", "") for i in indices if 0 <= i < len(articles)
    )


def add_keyword_counts(data: dict) -> dict:
    """Ajoute data['keyword_counts'] (niveau bulletin) et, pour chaque
    instrument, instr['keyword_counts'] (niveau instrument)."""
    lang = data.get("lang", "fr")
    data["keyword_counts"] = count_keywords(document_text(data), lang)
    for instr in data.get("instruments") or []:
        instr["keyword_counts"] = count_keywords(
            instrument_text(instr, data.get("articles") or []), lang
        )
    return data


def post_enrich(json_path: Path) -> dict:
    """Charge un JSON enrichi v1, ajoute metadata + keyword_counts, et le
    réécrit en place.  Retourne le data dict."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data["metadata"] = build_document_metadata(data)
    add_keyword_counts(data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
