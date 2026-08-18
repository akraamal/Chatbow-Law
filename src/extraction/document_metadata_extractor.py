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
import warnings
from datetime import datetime

# --- Patterns FRANÇAIS ---
# "N° 7500", "N°7500", "n° 7500", "NO 4804" (l'OCR confond °, O et o),
# "N° 7460 bis" (édition bis/ter — le suffixe est volontairement capturé)
BO_NUMBER_PATTERN_FR = r"[Nn]\s*[°ºoO]\s*(\d{3,5}(?:\s*[-–]?\s*(?:bis|ter))?)"

# Numéro du BO extrait du nom de fichier : "BO_6804_Fr_abc123",
# "BO_7460-bis Fr" (doc_id / stem du fichier)
BO_NUMBER_FILENAME_PATTERN = r"BO_(\d{3,5})(?:\s*[-–]?\s*(?:bis|ter))?"

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

# Partie numérique d'un numéro de BO (supprime un suffixe bis/ter éventuel)
_BO_NUM_ONLY_RE = re.compile(r"^(\d{3,5})")


def _bo_num_only(value: str) -> str | None:
    """Partie numérique de '6804', '7460 bis', '7460-bis' → '7460'."""
    m = _BO_NUM_ONLY_RE.match((value or "").strip())
    return m.group(1) if m else None


def extract_bo_number(text: str, window: int = 500, lang: str = "fr") -> str | None:
    """Cherche le numéro du BO dans les `window` premiers caractères (en-tête)."""
    pattern = BO_NUMBER_PATTERN_AR if lang == "ar" else BO_NUMBER_PATTERN_FR
    m = re.search(pattern, text[:window])
    return m.group(1) if m else None


def extract_bo_number_from_filename(doc_id: str) -> str | None:
    """
    Numéro du BO déduit du nom de fichier (doc_id, ex. 'BO_6804_Fr_1a2b3c').

    Retourne le numéro AVEC le suffixe bis/ter éventuel ('7460 bis'),
    ou None si le nom ne suit pas le format BO_<numéro>_<lang>_<hash>.
    """
    if not doc_id:
        return None
    m = re.search(BO_NUMBER_FILENAME_PATTERN, doc_id)
    return m.group(1).strip() if m else None


def extract_bo_number_cross_validated(
    text: str,
    doc_id: str | None = None,
    window: int = 500,
    lang: str = "fr",
) -> dict:
    """
    Numéro du BO validé par deux sources indépendantes :

    1. le nom de fichier (doc_id, ex. 'BO_6804_Fr_1a2b3c') ;
    2. l'en-tête du document (les `window` premiers caractères du texte,
       ex. 'N° 6804' / 'NO 4804' / 'عدد 7506').

    Retourne un dict :
        bo_number            — valeur retenue (nom de fichier en priorité)
        bo_number_source     — 'filename' | 'header' | 'filename+header'
        bo_number_confidence — 'high' (les deux sources concordent),
                               'low' (une seule source disponible),
                               'mismatch' (les deux diffèrent — signaler)
        bo_number_header     — valeur lue dans l'en-tête (None si absente)

    Un désaccord est signalé (warnings.warn) et la valeur du nom de fichier
    est retenue : un mismatch indique un nom de fichier incorrect, un
    en-tête mal OCR-isé, ou une édition bis/ter dont le suffixe est porté
    par une seule des deux sources.
    """
    filename_num = extract_bo_number_from_filename(doc_id)
    header_raw = extract_bo_number(text, window=window, lang=lang)
    header_num = _bo_num_only(header_raw) if header_raw else None
    filename_num_only = _bo_num_only(filename_num) if filename_num else None

    result = {
        "bo_number": None,
        "bo_number_source": None,
        "bo_number_confidence": None,
        "bo_number_header": header_raw,
    }

    if filename_num and header_num:
        if filename_num_only == header_num:
            result.update(bo_number=filename_num,
                          bo_number_source="filename+header",
                          bo_number_confidence="high")
        else:
            warnings.warn(
                f"bo_number mismatch: filename '{filename_num}' "
                f"vs header '{header_raw}' (doc_id={doc_id!r})"
            )
            result.update(bo_number=filename_num,
                          bo_number_source="filename",
                          bo_number_confidence="mismatch")
    elif filename_num:
        result.update(bo_number=filename_num,
                      bo_number_source="filename",
                      bo_number_confidence="low")
    elif header_raw:
        result.update(bo_number=header_raw,
                      bo_number_source="header",
                      bo_number_confidence="low")
    return result


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
    """Point d'entrée : regroupe les extractions ci-dessus dans un dict.

    ``bo_number`` est désormais validé par recoupement nom-de-fichier /
    en-tête (voir extract_bo_number_cross_validated) : les champs
    bo_number_source et bo_number_confidence documentent la fiabilité.
    """
    return {
        "doc_id": doc_id,
        "lang": lang,
        **extract_bo_number_cross_validated(text, doc_id=doc_id, lang=lang),
        "date_publication": extract_publication_date(text, lang=lang),
        "edition_label": extract_edition_label(text, lang=lang),
    }


if __name__ == "__main__":
    sample = "Cent-quinzième année – N° 7500\n28 chaoual 1447 (16 avril 2026)\nROYAUME DU MAROC"
    print(extract_document_metadata(sample, doc_id="BO_7500_Fr", lang="fr"))
