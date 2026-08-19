"""
adli_v2.catalog
---------------
Catalogue des instruments de tous les JSON enrichis v2, trié DÉCRETS EN
PREMIER : chaque entrée porte le type d'instrument, la référence, le titre,
les métadonnées du bulletin, les index des articles complets et les
compteurs de mots-clés de l'instrument.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Rang de tri « décret-first » : les décrets (et décrets-lois) passent
# avant tout le reste ; l'ordre secondaire est le numéro de BO décroissant.
TYPE_RANK = {
    "DECRET": 0,
    "DECRET_LOI": 0,
    "LOI": 2,
    "DAHIR": 3,
    "ARRETE": 4,
    "DECISION": 5,
    "BULLETIN_OFFICIEL": 6,
}
DEFAULT_RANK = 10

_BO_NUM_RE = re.compile(r"(\d{3,5})")


def _type_rank(instrument_type) -> int:
    return TYPE_RANK.get(str(instrument_type or "").upper(), DEFAULT_RANK)


def _bo_sort_key(bo_number) -> int:
    if not bo_number:
        return -1
    m = _BO_NUM_RE.search(str(bo_number))
    return int(m.group(1)) if m else -1


def entry_from_json(data: dict) -> list[dict]:
    """Une entrée par instrument du document enrichi v2."""
    meta = data.get("metadata") or {}
    entries = []
    for instr in data.get("instruments") or []:
        entries.append({
            "doc_id": data.get("doc_id"),
            "doc_name": meta.get("doc_name"),
            "lang": data.get("lang"),
            "bo_number": meta.get("bo_number"),
            "date_parution": meta.get("date_parution"),
            "instrument_type": instr.get("instrument_type"),
            "reference": instr.get("reference"),
            "title": instr.get("title") or instr.get("reference_label"),
            "decree_date_gregorian": instr.get("decree_date_gregorian"),
            "n_articles": instr.get("n_articles"),
            "article_indices": instr.get("article_indices"),
            "keyword_counts": instr.get("keyword_counts", {}),
        })
    return entries


def build_catalog(annotated_dir: Path) -> list[dict]:
    """Catalogue complet, trié décret-first puis BO décroissant."""
    annotated_dir = Path(annotated_dir)
    entries: list[dict] = []
    for path in sorted(annotated_dir.glob("*_entities.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        entries.extend(entry_from_json(data))
    entries.sort(key=lambda e: (_type_rank(e["instrument_type"]),
                                -_bo_sort_key(e["bo_number"])))
    return entries


def save_catalog(entries: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_entries": len(entries), "entries": entries}, f,
                  ensure_ascii=False, indent=2)
    return path


def load_catalog(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["entries"]
