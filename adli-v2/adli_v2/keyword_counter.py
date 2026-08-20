"""
adli_v2.keyword_counter
-----------------------
Comptage de fréquences de mots-clés (aucun modèle, aucun GPU) : regex
bornée par mot en français (pluriel en -s), correspondance sur TOKENS en
arabe (normalisation : suppression des diacritiques et des préfixes
d'article définis ال/وال/بال/كال/فال/لال/لل — « رسم » ne matche plus
« رسمي »).  Les termes multi-mots (ex. « الضريبة على القيمة المضافة »)
se matchent comme une suite consécutive de tokens.  La liste des
mots-clés est STATIQUE (config/keywords_fr.json, keywords_ar.json) —
catégories héritées de l'ancien classifieur de domaine (v1), dont elles
constituent le remplaçant déterministe.

Les formes irrégulières du pluriel français (travaux, tribunaux, eaux...)
ne se dérivent pas par règle : elles sont déclarées telles quelles dans
keywords_fr.json (« edit JSON, pas de code »).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "config"

_cache: dict[str, dict[str, list[str]]] = {}

# --- Normalisation arabe ----------------------------------------------------
# « رسم » ≠ « رسمي » : la correspondance se fait sur des TOKENS (suite de
# lettres arabes), après suppression des diacritiques et des préfixes
# d'article défini (ال, وال, بال, كال, فال, لال, لل).
_AR_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لال", "لل", "ال")
_AR_TOKEN_RE = re.compile(r"[\u0600-\u06FF]+")


def _ar_norm_token(token: str) -> str:
    """Token arabe normalisé : diacritiques retirés, préfixe d'article coupé
    (« الضَّريبة » → « ضريبة », « بالضرائب » → « ضرائب »)."""
    token = _AR_DIACRITICS_RE.sub("", token)
    for prefix in _AR_PREFIXES:
        if token.startswith(prefix):
            return token[len(prefix):]
    return token


def _ar_count_tokens(tokens: list[str], term: str) -> int:
    """Occurrences de *term* dans la liste de tokens : égalité exacte pour un
    mot unique, suite consécutive de tokens pour un terme multi-mots."""
    parts = term.split()
    if len(parts) == 1:
        expected = _ar_norm_token(parts[0])
        return sum(1 for t in tokens if _ar_norm_token(t) == expected)
    expected = [_ar_norm_token(p) for p in parts]
    n = len(expected)
    count = 0
    for i in range(len(tokens) - n + 1):
        if all(_ar_norm_token(tokens[i + j]) == expected[j] for j in range(n)):
            count += 1
    return count


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
    borne de mot ; AR : correspondance sur tokens normalisés — préfixe
    d'article et diacritiques ignorés, multi-mots en suite consécutive)."""
    lang = lang.strip().lower()
    if not text:
        return {term: 0 for term in _all_terms(lang)}
    if lang == "ar":
        tokens = _AR_TOKEN_RE.findall(text)
        return {term: _ar_count_tokens(tokens, term) for term in _all_terms("ar")}
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
    """Point d'entrée : comptages BRUTS ('per_term', 'per_category') et
    NORMALISÉS par 1 000 mots ('per_term_normalized',
    'per_category_normalized', 'n_words') — pour comparer des documents
    de longueurs différentes (densité de termes)."""
    lang = lang.strip().lower()
    per_term = count_terms(text, lang)
    per_category = count_by_category(per_term, lang)
    n_words = len(re.findall(r"[\w\u0600-\u06FF]+", text or ""))
    norm = (lambda v: round(1000.0 * v / n_words, 2)) if n_words else (lambda v: 0.0)
    return {
        "per_term": per_term,
        "per_category": per_category,
        "per_term_normalized": {t: norm(v) for t, v in per_term.items()},
        "per_category_normalized": {c: norm(v) for c, v in per_category.items()},
        "n_words": n_words,
    }
