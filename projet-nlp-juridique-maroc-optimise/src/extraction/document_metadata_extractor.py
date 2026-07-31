"""
src/extraction/document_metadata_extractor.py
Extrait les métadonnées d'en-tête d'un Bulletin Officiel à partir du texte
brut (même famille regex que dates_patterns.py) : numéro du BO, date
grégorienne de publication, année d'édition.

Exemple d'en-tête réel (FR) :
    "Cent-quinzième année – N° 7500"
    "28 chaoual 1447 (16 avril 2026)"
"""
import re
from datetime import datetime

# --- Patterns FRANÇAIS ---
# "N° 7500", "N°7500", "n° 7500"
BO_NUMBER_PATTERN_FR = r"[Nn]°\s*(\d+(?:[-–]bis)?)"

# Date grégorienne entre parenthèses : "(16 avril 2026)", "(1er janvier 2026)"
GREGORIAN_DATE_PATTERN_FR = (
    r"\((\d{1,2})(?:er)?\s+"
    r"(janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[ûu]t|"
    r"septembre|octobre|novembre|d[ée]cembre)\s+(\d{4})\)"
)

_MONTHS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}

# Année d'édition en toutes lettres : "Cent-quinzième année"
EDITION_YEAR_PATTERN_FR = r"([A-Za-zÀ-ÿ\-]+)\s+ann[ée]e"

# --- Patterns ARABE ---
# Format 1: "عدد 4 - 7350" (issue N - BO serial)
# Format 2: "عدد7360" or "عدد 7360" (BO serial directly)
# Format 3: "العدد 4 - 7350" with definite article
BO_NUMBER_PATTERN_AR = r"(?:عدد|العدد)\s*(?:\d+\s*[-–])?\s*(\d{3,5})"

# Mois grégoriens en arabe (transcrits, tels qu'utilisés dans les BO)
_MONTHS_AR = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ماي": 5, "يونيو": 6,
    "يوليوز": 7, "غشت": 8, "شتنبر": 9, "أكتوبر": 10, "نونبر": 11, "دجنبر": 12,
    # orthographes alternatives sans alif
    "براير": 2, "ابريل": 4, "اكتوبر": 10,
}

GREGORIAN_DATE_PATTERN_AR = (
    r"\((\d{1,2})\s*"
    r"(يناير|فبراير|مارس|أبريل|ابريل|ماي|يونيو|يوليوز|غشت|شتنبر|أكتوبر|اكتوبر|نونبر|دجنبر)"
    r"\s*(\d{4})\)"
)


def extract_bo_number(text: str, window: int = 500, lang: str = "fr") -> str | None:
    """Cherche le numéro du BO dans les `window` premiers caractères (en-tête)."""
    pattern = BO_NUMBER_PATTERN_AR if lang == "ar" else BO_NUMBER_PATTERN_FR
    m = re.search(pattern, text[:window])
    return m.group(1) if m else None


def extract_publication_date(text: str, window: int = 500, lang: str = "fr") -> str | None:
    """Retourne la date de publication au format ISO (YYYY-MM-DD), ou None."""
    if lang == "ar":
        pattern, months = GREGORIAN_DATE_PATTERN_AR, _MONTHS_AR
    else:
        pattern, months = GREGORIAN_DATE_PATTERN_FR, _MONTHS_FR

    m = re.search(pattern, text[:window], flags=re.IGNORECASE)
    if not m:
        return None
    day, month_name, year = m.groups()
    month = months.get(month_name.strip())
    if month is None:
        return None
    try:
        return datetime(int(year), month, int(day)).date().isoformat()
    except ValueError:
        return None


def extract_edition_label(text: str, window: int = 500, lang: str = "fr") -> str | None:
    """Retourne le libellé d'année d'édition en toutes lettres."""
    pattern = EDITION_YEAR_PATTERN_FR  # l'arabe n'utilise pas ce format
    m = re.search(pattern, text[:window], flags=re.IGNORECASE)
    return m.group(1) if m else None


def extract_document_metadata(text: str, doc_id: str, lang: str = "fr") -> dict:
    """Point d'entrée : regroupe les trois extractions ci-dessus dans un dict."""
    return {
        "doc_id": doc_id,
        "lang": lang,
        "bo_number": extract_bo_number(text, lang=lang),
        "date_publication": extract_publication_date(text, lang=lang),
        "edition_label": extract_edition_label(text, lang=lang),
    }


if __name__ == "__main__":
    sample = "Cent-quinzième année – N° 7500\n28 chaoual 1447 (16 avril 2026)\nROYAUME DU MAROC"
    print(extract_document_metadata(sample, doc_id="BO_7500_Fr", lang="fr"))