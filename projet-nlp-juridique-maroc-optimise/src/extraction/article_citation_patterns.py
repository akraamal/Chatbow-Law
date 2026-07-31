"""
src/extraction/article_citation_patterns.py
Étape 4a (FR) — Détection des citations d'articles.

Même famille d'outils que loi_decrets_patterns.py / dates_patterns.py :
regex sur texte brut -> (start, end, texte) -> char_span() dans entities.py.
Nouveau label : ARTICLE_CITATION.
"""
import re

LABEL = "ARTICLE_CITATION"

_NUM = r"\d+(?:[-–.]\d+)?"
ARTICLE_CITATION_PATTERNS = [
    rf"\bl['’]?article\s+{_NUM}\b",                          # "l'article 5", "l'article 5-2"
    rf"\barticles\s+{_NUM}(?:\s*,\s*{_NUM})*(?:\s+et\s+{_NUM})?\b",  # "articles 3 et 4", "articles 64-5, 64-7"
    rf"\barticles?\s+{_NUM}\s+à\s+{_NUM}\b",                  # "articles 3 à 7"
]


def find_article_citations(text: str):
    """
    Retourne une liste de (start, end, texte_matché), triée et dédoublonnée
    (le match le plus long l'emporte sur un match plus court qui le chevauche).
    """
    matches = []
    for pattern in ARTICLE_CITATION_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((m.start(), m.end(), m.group()))

    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    filtered, last_end = [], -1
    for start, end, txt in matches:
        if start >= last_end:
            filtered.append((start, end, txt))
            last_end = end
    return filtered


if __name__ == "__main__":
    sample = "Conformément à l'article 5 de la loi n° 03-25, les articles 3 à 7 ci-dessus s'appliquent."
    for m in find_article_citations(sample):
        print(m)