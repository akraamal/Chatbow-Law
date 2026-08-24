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
from pathlib import Path

from src.extraction.dates_patterns_ar import (  # noqa: F401  (réexport utile)
    MOIS_GREGORIEN_AR,
    _GREG_MONTHS_SORTED,
)

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

# Mois grégoriens en arabe : source UNIQUE = dates_patterns_ar.py (couvre
# les orthographes maghrébines ET MSA : أغسطس, مايو, سبتمبر, نوفمبر,
# ديسمبر…). L'ancienne copie locale était incomplète et a dérivé.
_MONTHS_AR = MOIS_GREGORIEN_AR

GREGORIAN_DATE_PATTERN_AR = (
    r"\((\d{1,2})\s*"
    r"(" + "|".join(_GREG_MONTHS_SORTED) + r")"
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


# --- Repli OCR ciblé (en-tête page 1) --------------------------------------
# Cas d'usage : couche texte native corrompue au niveau caractère
# (BO_7430_Ar : l'en-tête se rend correctement à l'écran mais PyMuPDF en
# extrait un texte brouillé même trié par position x des glyphes — corruption
# du PDF source, pas un bug d'ordre de lecture). On relit donc l'IMAGE du
# bandeau supérieur uniquement (HEADER_BAND_FRACTION ≈ 8 %), pas la page.

_HEADER_OCR_DPI = 200          # suffisant pour une date, reste rapide
_HEADER_OCR_LANGS = ("ar", "fr")


def extract_publication_date_ocr(
    pdf_path: str | Path,
    page_number: int = 1,
) -> str | None:
    """Date de publication relue par OCR sur le seul bandeau d'en-tête.

    Ne doit être appelée que lorsque l'extraction native a échoué. Toute
    erreur (Tesseract absent, pack de langue manquant, PDF illisible)
    dégrade en warning + None — jamais d'exception vers le pipeline.
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract

        from src.ingestion.ocr_extractor import HEADER_BAND_FRACTION
    except ImportError as e:                       # dépendance non installée
        warnings.warn(f"OCR header indisponible (import): {e}")
        return None

    try:
        with fitz.open(str(pdf_path)) as doc:
            page = doc[max(0, page_number - 1)]
            zoom = _HEADER_OCR_DPI / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        header = img.crop((0, 0, img.width,
                           int(img.height * HEADER_BAND_FRACTION)))

        for lang in _HEADER_OCR_LANGS:
            tesseract_lang = "ara+fra" if lang == "ar" else "fra"
            text = pytesseract.image_to_string(
                header, lang=tesseract_lang, config="--oem 3 --psm 6")
            iso = extract_publication_date(text or "", lang=lang,
                                           window=len(text or "") + 10)
            if iso:
                return iso
        return None
    except Exception as e:                         # noqa: BLE001
        warnings.warn(f"OCR header en échec pour {pdf_path!r}: {e}")
        return None


def extract_publication_date_cross_validated(
    text: str,
    lang: str = "fr",
    pdf_path: str | Path | None = None,
    page_number: int = 1,
) -> dict:
    """Date de publication + provenance/confiance, même pattern que
    extract_bo_number_cross_validated :

        date_publication_source     — 'text' | 'ocr_header' | None
        date_publication_confidence — 'high' (texte natif) |
                                      'low' (repli OCR) | None
    """
    native = extract_publication_date(text, lang=lang)
    result = {
        "date_publication": native,
        "date_publication_source": "text" if native else None,
        "date_publication_confidence": "high" if native else None,
    }
    if native or not pdf_path:
        return result                              # dégradation identique à avant
    ocr_iso = extract_publication_date_ocr(pdf_path, page_number)
    if ocr_iso:
        result.update(date_publication=ocr_iso,
                      date_publication_source="ocr_header",
                      date_publication_confidence="low")
    return result


def resolve_raw_pdf_path(stem: str, raw_dir: str | Path) -> str | None:
    """
    Retrouve le PDF brut correspondant à un stem de fichier interim/processed
    (ex. 'BO_7430_Ar', 'BO_7522_Fr_3af5da6a', 'BO_7460-bis_Fr').

    Le pipeline d'ingestion nomme systématiquement interim/processed/annotated
    d'après ``pdf_path.stem`` tel quel (voir run_pipeline_complet.run_ingestion,
    ``stem = pdf_path.stem``) : aucun hash n'est ajouté par le pipeline lui-même,
    donc '<stem>.pdf' est le nom exact à chercher sous raw_dir (recherche
    récursive, au cas où raw/ contiendrait des sous-dossiers).

    Retourne le chemin en str (compatible avec extract_document_metadata,
    pdf_path=...), ou None si rien n'est trouvé (le fallback OCR est alors
    simplement ignoré par extract_publication_date_cross_validated — pas
    d'erreur).
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None
    match = next(raw_dir.glob(f"**/{stem}.pdf"), None)
    return str(match) if match else None


def extract_edition_label(text: str, window: int = 500, lang: str = "fr") -> str | None:
    """Retourne le libellé d'année d'édition en toutes lettres."""
    pattern = EDITION_YEAR_PATTERN_FR  # l'arabe n'utilise pas ce format
    m = re.search(pattern, text[:window], flags=re.IGNORECASE)
    return m.group(1) if m else None


def extract_document_metadata(
    text: str,
    doc_id: str,
    lang: str = "fr",
    pdf_path: str | None = None,
    page_number: int = 1,
) -> dict:
    """Point d'entrée : regroupe les extractions ci-dessus dans un dict.

    ``bo_number`` est désormais validé par recoupement nom-de-fichier /
    en-tête (voir extract_bo_number_cross_validated) : les champs
    bo_number_source et bo_number_confidence documentent la fiabilité.

    ``pdf_path`` / ``page_number`` (optionnels, rétrocompatibles) : quand un
    PDF brut est fourni ET que l'extraction native de la date échoue (couche
    texte corrompue), un repli OCR ciblé sur le bandeau d'en-tête est tenté —
    voir extract_publication_date_cross_validated. Sans pdf_path, le
    comportement est strictement identique à avant.
    """
    return {
        "doc_id": doc_id,
        "lang": lang,
        **extract_bo_number_cross_validated(text, doc_id=doc_id, lang=lang),
        **extract_publication_date_cross_validated(
            text, lang=lang, pdf_path=pdf_path, page_number=page_number),
        "edition_label": extract_edition_label(text, lang=lang),
    }


if __name__ == "__main__":
    sample = "Cent-quinzième année – N° 7500\n28 chaoual 1447 (16 avril 2026)\nROYAUME DU MAROC"
    print(extract_document_metadata(sample, doc_id="BO_7500_Fr", lang="fr"))
