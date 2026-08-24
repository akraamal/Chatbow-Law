"""Tests du repli OCR de date_publication et du résolveur de PDF brut.

Aucun de ces tests n'exige Tesseract : le fallback OCR lui-même dégrade en
warning + None quand la dépendance manque (comportement couvert par
test_metadata_degrades_without_pdf / test_ocr_missing_tesseract_warns).
"""

import warnings

import pytest

from src.extraction.document_metadata_extractor import (
    extract_document_metadata,
    extract_publication_date,
    resolve_raw_pdf_path,
)

HEADER_FR_CLEAN = (
    "Cent-quinzième année – N° 7500\n"
    "28 chaoual 1447 (16 avril 2026)\n"
    "ROYAUME DU MAROC"
)


# ── Fix liste de mois arabes (MSA) ────────────────────────────────────────

@pytest.mark.parametrize("month,expected", [
    ("أغسطس", "2025-08-14"),   # MSA août — était ignoré avant la dédup
    ("مايو", "2025-05-14"),     # MSA mai
    ("سبتمبر", "2025-09-14"),   # MSA septembre
    ("نوفمبر", "2025-11-14"),   # MSA novembre
    ("ديسمبر", "2025-12-14"),   # MSA décembre
])
def test_extract_publication_date_ar_msa_months(month, expected):
    assert extract_publication_date(
        f"(14 {month} 2025)", lang="ar") == expected


def test_extract_publication_date_ar_maghrebi_still_works():
    """Non-régression : les orthographes maghrébines d'origine restent OK."""
    assert extract_publication_date("(14 غشت 2025)", lang="ar") == "2025-08-14"
    assert extract_publication_date("(14 يناير 2025)", lang="ar") == "2025-01-14"


def test_extract_publication_date_fr_clean():
    assert extract_publication_date(HEADER_FR_CLEAN) == "2026-04-16"


# ── resolve_raw_pdf_path ──────────────────────────────────────────────────

def test_resolve_raw_pdf_path_finds_exact_stem(tmp_path):
    (tmp_path / "BO_9999_Fr.pdf").write_bytes(b"%PDF-1.4 fake")
    assert resolve_raw_pdf_path("BO_9999_Fr", tmp_path) == \
        str(tmp_path / "BO_9999_Fr.pdf")


def test_resolve_raw_pdf_path_recursive(tmp_path):
    sub = tmp_path / "sous-dossier"
    sub.mkdir()
    (sub / "BO_7430_Ar.pdf").write_bytes(b"%PDF-1.4 fake")
    found = resolve_raw_pdf_path("BO_7430_Ar", tmp_path)
    assert found == str(sub / "BO_7430_Ar.pdf")


def test_resolve_raw_pdf_path_returns_none_when_absent(tmp_path):
    assert resolve_raw_pdf_path("BO_9999_Fr", tmp_path) is None


def test_resolve_raw_pdf_path_none_when_dir_missing(tmp_path):
    assert resolve_raw_pdf_path("BO_9999_Fr", tmp_path / "inexistant") is None


def test_resolve_raw_pdf_path_no_partial_match(tmp_path):
    """Pas de fuzzy matching : 'BO_99.pdf' ne doit pas matcher 'BO_9999_Fr.pdf'."""
    (tmp_path / "BO_9999_Fr.pdf").write_bytes(b"%PDF")
    assert resolve_raw_pdf_path("BO_99", tmp_path) is None


# ── Dégradation sans pdf_path (comportement inchangé) ────────────────────

def test_metadata_degrades_without_pdf_when_native_fails():
    """Texte sans date lisible + pas de PDF : source/confidence None, et
    surtout AUCUNE exception ni tentative OCR."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # tout warning ferait échouer le test
        meta = extract_document_metadata(
            "En-tête brouillée sans date exploitable.", doc_id="BO_X",
            lang="ar", pdf_path=None)
    assert meta["date_publication"] is None
    assert meta["date_publication_source"] is None
    assert meta["date_publication_confidence"] is None
    assert meta["bo_number"] is None


def test_metadata_native_text_keeps_high_confidence_and_ignores_pdf():
    """Texte propre + pdf_path fourni : la source reste 'text'/'high', l'OCR
    n'est jamais déclenché (aucun accès disque nécessaire)."""
    meta = extract_document_metadata(
        HEADER_FR_CLEAN, doc_id="BO_7500_Fr", lang="fr",
        pdf_path="Z:/chemin/inexistant/BO_7500_Fr.pdf")
    assert meta["date_publication"] == "2026-04-16"
    assert meta["date_publication_source"] == "text"
    assert meta["date_publication_confidence"] == "high"


def test_ocr_missing_tesseract_warns_and_returns_none(tmp_path):
    """Sans Tesseract installé, extract_publication_date_ocr prévient puis
    retourne None au lieu de lever — et la cross_validated retombe sur
    source=None (couche texte vide)."""
    from src.extraction import document_metadata_extractor as m

    fake_pdf = tmp_path / "BO_FAKE.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")   # fitz ouvrira… ou échouera peu importe

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        iso = m.extract_publication_date_ocr(fake_pdf, page_number=1)
    assert iso is None
    assert any("OCR" in str(w.message) for w in caught), \
        f"un warning explicite est attendu, eu: {[str(w.message) for w in caught]}"

    # cross_validated ne doit pas lever non plus avec un chemin fourni
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        res = m.extract_document_metadata(
            "texte sans date", doc_id="BO_FAKE", lang="fr",
            pdf_path=str(fake_pdf))
    assert res["date_publication"] is None
    assert res["date_publication_source"] is None
