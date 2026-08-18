"""
test_interim_provenance.py
---------------------------
Régression : la provenance des fichiers data/interim/ (sidecar .meta.json,
pipeline.stamp_interim_provenance) doit détecter un cache stale — texte
extrait par une version antérieure de pdf_extractor.py ou depuis un PDF
modifié — et bloquer la régénération silencieuse de JSON sur du texte
obsolète (cas BO_7510 : l'ancienne extraction empilait les titres de deux
arrêtés côte à côte ; le correctif de lecture colonne par colonne existait
déjà, mais rien n'invalidait l'interim, et le JSON avait été régénéré sur
le texte stale → fusion n=2).

Usage:
    python -m pytest tests/test_interim_provenance.py -v
"""

import pytest

from src.ingestion import pipeline
from src.ingestion.pipeline import (
    stamp_interim_provenance,
    interim_freshness,
    ensure_interim_fresh,
)


def test_stale_extractor_version_detected(tmp_path, monkeypatch):
    """Une extraction faite par une version antérieure de l'extracteur doit
    être refusée tant que l'ingestion n'a pas été relancée."""
    interim = tmp_path / "BO_X_Fr.txt"
    pdf = tmp_path / "src.pdf"
    pdf.write_bytes(b"abc")

    monkeypatch.setattr(pipeline, "EXTRACTOR_VERSION", "2")
    stamp_interim_provenance(interim, pdf)

    monkeypatch.setattr(pipeline, "EXTRACTOR_VERSION", "3")
    fresh, reason = interim_freshness(interim)
    assert not fresh
    assert "extracteur" in reason

    with pytest.raises(RuntimeError, match="stale"):
        ensure_interim_fresh(interim)

    monkeypatch.setenv("ALLOW_STALE_INGESTION", "1")
    ensure_interim_fresh(interim)  # ne lève pas (saut de garantie explicite)


def test_missing_provenance_is_stale(tmp_path):
    """Un interim sans sidecar .meta.json (fichiers antérieurs au suivi de
    provenance) doit être traité comme stale, jamais comme frais."""
    interim = tmp_path / "BO_X_Fr.txt"
    interim.write_text("x", encoding="utf-8")
    fresh, reason = interim_freshness(interim)
    assert not fresh
    assert "provenance inconnue" in reason


def test_changed_pdf_detected(tmp_path, monkeypatch):
    """Un PDF source modifié après l'extraction rend l'interim stale."""
    interim = tmp_path / "BO_X_Fr.txt"
    pdf = tmp_path / "src.pdf"
    pdf.write_bytes(b"abc")

    monkeypatch.setattr(pipeline, "EXTRACTOR_VERSION", "1")
    stamp_interim_provenance(interim, pdf)

    pdf.write_bytes(b"abcd")
    fresh, reason = interim_freshness(interim)
    assert not fresh
    assert "changé" in reason


def test_fresh_interim_passes(tmp_path, monkeypatch):
    """Extraite par l'extracteur actuel depuis le PDF actuel : rien à
    signaler."""
    interim = tmp_path / "BO_X_Fr.txt"
    pdf = tmp_path / "src.pdf"
    pdf.write_bytes(b"abc")

    monkeypatch.setattr(pipeline, "EXTRACTOR_VERSION", "1")
    stamp_interim_provenance(interim, pdf)

    fresh, reason = interim_freshness(interim)
    assert fresh
    ensure_interim_fresh(interim)  # ne lève pas
