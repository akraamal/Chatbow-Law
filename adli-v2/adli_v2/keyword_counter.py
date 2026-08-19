"""
adli_v2.keyword_counter
-----------------------
Comptage de fréquences de mots-clés (aucun modèle, aucun GPU) : regex
bornée par mot en français, sous-chaîne en arabe.  La liste des mots-clés
est STATIQUE (config/keywords_fr.json, keywords_ar.json) — catégories
héritées de l'ancien classifieur de domaine (v1), dont elles constituent
le remplaçant déterministe.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "config"

_cache: dict[str, dict[str, list[str]]] = {}


def load_keywords(lang: str = "fr") -> dict[str, list[str]]:
    """Charge la liste statique : {catégorie: [mots-clés,...]}, mise en cache."""
    key = lang.strip().lower()
    if key not in _cache:
        path = CONFIG_DIR / f"keywords_{key}.json"
        with open(path, encoding="utf-8") as f:
            _cache[key] = json.load(f)["categories"]
    return _cache[key]


def _all_terms(lang: str) -> list[str]:
    """Tous les mots-clés de la langue, dédupliqués, dans l'ordre des catégories."""
    return list(dict.fromkeys(
        term for terms in load_keywords(lang).values() for term in terms
    ))


def _fr_pattern(term: str) -> re.Pattern:
    # Borne de mot + pluriel en -s (« impôt » match « impôts », pas « assiette »).
    return re.compile(rf"(?<!\w){re.escape(term)}s?(?!\w)", re.IGNORECASE)


def count_terms(text: str, lang: str = "fr") -> dict[str, int]:
    """Occurrences de chaque mot-clé dans *text* (FR : casse insensible et
    borne de mot ; AR : sous-chaîne, l'écriture arabe ne tolère pas de
    bornes de mot fiables)."""
    lang = lang.strip().lower()
    if not text:
        return {term: 0 for term in _all_terms(lang)}
    if lang == "ar":
        return {term: text.count(term) for term in _all_terms("ar")}
    return {
        term: len(_fr_pattern(term).findall(text)) for term in _all_terms("fr")
    }


def count_by_category(term_counts: dict[str, int], lang: str = "fr") -> dict[str, int]:
    """Total par catégorie (somme des occurrences de ses mots-clés)."""
    return {
        cat: sum(term_counts.get(term, 0) for term in terms)
        for cat, terms in load_keywords(lang).items()
    }


def count_keywords(text: str, lang: str = "fr") -> dict:
    """Point d'entrée : {'per_term': {...}, 'per_category': {...}}."""
    per_term = count_terms(text, lang)
    return {
        "per_term": per_term,
        "per_category": count_by_category(per_term, lang),
    }
