"""
src/extraction/article_citation_patterns_ar.py
Étape 4a (AR) — Détection des citations d'articles (version améliorée).

Couvre :
- المادة 5, المادة الأولى, المادة 5 مكرر
- المواد 3 إلى 7, المواد 3 و 4, المواد 3-7
- avec préfixes : للمادة, بالمادة, كالمادة, فالمادة, والمادة
- chiffres latins (0-9) et arabo‑indiens (٠-٩)
"""

import re

LABEL = "ARTICLE_CITATION"

# Chiffres latins et arabo‑indiens
_DIGITS = r"[\d٠-٩]+"

# Nombres en lettres (pour les premières)
_NUM_WORDS = r"(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة)"

# Préfixes possibles devant "مادة" (ال, لل, بال, كال, فال, وال)
_PREFIX = r"(?:ال|لل|بال|كال|فال|وال)?"

# Pattern principal : "المادة X" ou "مادة X" (avec préfixe optionnel)
PATTERN_ARTICLE_SINGLE = re.compile(
    rf"{_PREFIX}مادة\s+(?:{_DIGITS}|{_NUM_WORDS})(?:\s+مكرر)?"
)

# Pattern plurielles : "المواد X إلى Y" ou "المواد X و Y" ou "المواد X-Y"
PATTERN_ARTICLES_PLURAL = re.compile(
    rf"{_PREFIX}مواد\s+{_DIGITS}\s*(?:[-–إلىو]\s*{_DIGITS})?"
)


def find_article_citations(text: str):
    """
    Retourne une liste de (start, end, texte_matché), triée et dédoublonnée.
    Les matches les plus longs sont prioritaires en cas de chevauchement.
    """
    matches = []

    # Appliquer les deux patterns
    for pattern in (PATTERN_ARTICLE_SINGLE, PATTERN_ARTICLES_PLURAL):
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), m.group()))

    # Trier par position de début, puis par longueur décroissante
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Dédoublonnage : on ne garde que les matches qui ne chevauchent aucun match déjà retenu
    filtered = []
    for start, end, txt in matches:
        overlap = False
        for fs, fe, _ in filtered:
            if not (end <= fs or start >= fe):  # chevauchement
                overlap = True
                break
        if not overlap:
            filtered.append((start, end, txt))

    return filtered


if __name__ == "__main__":
    samples = [
        "طبقا للمادة 5 من القانون رقم 03-25، تطبق المواد 3 إلى 7 أعلاه.",
        "المادة الأولى تنص على ...",
        "المواد 5 و 6 و 7",
        "المادة 3 مكرر",
        "وفقاً للمادة ١٢ من الظهير.",
    ]
    for txt in samples:
        print(f"\n--- Texte : {txt}")
        for m in find_article_citations(txt):
            print(m)