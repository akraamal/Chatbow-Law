"""
loi_decrets_patterns_ar.py
----------------------------
Équivalent arabe de loi_decrets_patterns.py : patterns regex pour repérer
les références légales dans le texte ARABE du Bulletin Officiel marocain.

Termes juridiques couverts :
    ظهير              (dahir)
    قانون             (loi)
    مرسوم             (décret)
    قرار              (arrêté)
    منشور             (circulaire)
    الجريدة الرسمية    (Bulletin Officiel)
    DATE_HIJRI         (dates hégiriennes, avec exclusion des numéros de BO)

Améliorations majeures :
    - Le pattern DATE_HIJRI n'accepte plus les mots "عدد" (exclut les numéros de BO).
    - ARRETE_PATTERN tolère les erreurs OCR courantes (الفالحة/الفلاحة).
    - Tous les patterns utilisent _OPT_NUM pour accepter ou non le mot "رقم".
"""

import re

# ============================================================================
# PRIMITIVES
# ============================================================================

# Numéro type "1.09.20" ou "2.08.562" (séparateur point, parfois tiret)
_NUM = r"[\d٠-٩]+(?:[.\-–][\d٠-٩]+){1,2}"
_OPT_NUM = rf"(?:رقم\s*)?{_NUM}"

# Mois hégiriens réels (réutilisés de dates_patterns_ar.py) : le pattern
# DATE_HIJRI ne doit PAS accepter n'importe quel mot comme mois — sinon
# les dates grégoriennes ("28 فبراير 2025") étaient étiquetées DATE_HIJRI.
from src.extraction.dates_patterns_ar import MOIS_GREGORIEN_AR, MOIS_HIJRI_AR
_HIJRI_MONTHS_SORTED = sorted(MOIS_HIJRI_AR.keys(), key=len, reverse=True)
_GREG_MONTHS_SORTED = sorted(MOIS_GREGORIEN_AR.keys(), key=len, reverse=True)

# Date hégirienne : jour (1‑2 chiffres) + mois hégirien (éventuellement
# précédé de "من" : « 24 من ذي القعدة 1446 ») + année 3‑4 chiffres.
_HIJRI_DATE = (
    rf"[\d٠-٩]{{1,2}}\s+(?:من\s+)?"
    rf"(?:{'|'.join(map(re.escape, _HIJRI_MONTHS_SORTED))})"
    rf"\s+[\d٠-٩]{{3,4}}"
)

# Date grégorienne : jour + mois grégorien + année 4 chiffres.
_GREG_DATE = (
    rf"[\d٠-٩]{{1,2}}\s+"
    rf"(?:{'|'.join(map(re.escape, _GREG_MONTHS_SORTED))})"
    rf"\s+[\d٠-٩]{{4}}"
)

# Date entre parenthèses type "(18 فبراير 2009)" ou hégirien "27 chaoual 1393"
_DATE_PARENS = r"(?:\s*\([^)]{4,40}\))?"

# ============================================================================
# PATTERNS SPÉCIFIQUES
# ============================================================================

# --- ظهير (dahir), y compris "ظهير شريف" ---
DAHIR_PATTERN = re.compile(
    rf"ظهير(?:\s+(?:ال)?شريف)?\s*{_OPT_NUM}"
    rf"(?:\s+(?:ال)?صادر\s+في\s+{_HIJRI_DATE}){{0,1}}{_DATE_PARENS}"
)

# --- قانون (loi) ---
LOI_PATTERN = re.compile(
    rf"قانون\s*{_OPT_NUM}(?:\s+يتعلق\s+ب[^,،؛.\n]{{0,80}})?"
)

# --- مرسوم (décret) ---
DECRET_PATTERN = re.compile(
    rf"مرسوم\s*{_OPT_NUM}"
)

# --- قرار (arrêté) : tolère les erreurs OCR courantes ---
#   "قرار لوزير الفالحة" au lieu de "الفلاحة" (OCR)
ARRETE_PATTERN = re.compile(
    rf"قرار(?:\s+رقم\s*{_NUM})?(?:\s+مشترك)?\s+لوزير"
    rf"(?:[^,،؛.\n]{{0,100}})"
)

# --- منشور (circulaire) ---
CIRCULAIRE_PATTERN = re.compile(
    rf"منشور\s*{_OPT_NUM}"
)

# --- الجريدة الرسمية (référence au Bulletin Officiel) ---
BULLETIN_OFFICIEL_PATTERN = re.compile(
    rf"الجريدة\s+الرسمية\s*عدد\s*[\d٠-٩]{{3,4}}"
    rf"(?:\s+بتاريخ[^,،؛.\n)]{{4,40}}){{0,1}}{_DATE_PARENS}"
)

# ============================================================================
# DICTIONNAIRE CENTRAL
# ============================================================================

LEGAL_REFERENCE_PATTERNS_AR = {
    "DAHIR": DAHIR_PATTERN,
    "LOI": LOI_PATTERN,
    "DECRET": DECRET_PATTERN,
    "ARRETE": ARRETE_PATTERN,
    "CIRCULAIRE": CIRCULAIRE_PATTERN,
    "BULLETIN_OFFICIEL": BULLETIN_OFFICIEL_PATTERN,
    "DATE_HIJRI": re.compile(_HIJRI_DATE),       # uniquement les dates hégiriennes réelles
    "DATE_GREGORIAN": re.compile(_GREG_DATE),    # dates grégoriennes (mois arabes)
}


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    sample = (
        "بناء على الظهير الشريف رقم 1.09.20 الصادر في 22 صفر 1430 "
        "الموافق طبقا لقانون رقم 03.25 يتعلق بهيئات التوظيف الجماعي، "
        "وعلى مرسوم رقم 2.08.562، وعلى قرار لوزير الصناعة والتجارة، "
        "المنشور بالجريدة الرسمية عدد 7499 بتاريخ 25 شوال 1447."
        "كما جاء في منشور رقم 12/2022."
    )

    for label, pattern in LEGAL_REFERENCE_PATTERNS_AR.items():
        matches = pattern.findall(sample)
        print(f"{label}: {matches}")