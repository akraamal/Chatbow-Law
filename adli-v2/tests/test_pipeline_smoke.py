"""
test_pipeline_smoke.py
----------------------
Test de bout en bout du pipeline v2 sur un vrai PDF du corpus (BO_7480_Fr),
sans toucher aux répertoires v1 (dirs pointés vers tmp_path).  Vérifie que
le JSON produit porte le bloc metadata, les keyword_counts (document +
instrument) et les instruments.

Lent (chargement des modèles NLP) : 1-2 minutes.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from adli_v2.pipeline import process_pdf

FR_PDF = Path("data/raw/fr/BO_7480_Fr.pdf")


@pytest.mark.skipif(
    not FR_PDF.exists(), reason="BO_7480_Fr.pdf absent du dépôt (données gitignorées)"
)
def test_v2_pipeline_end_to_end(tmp_path):
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    annotated = tmp_path / "annotated"
    md = tmp_path / "annotated-MD"
    uploads = tmp_path / "uploads"

    out = process_pdf(
        FR_PDF,
        interim_dir=interim,
        processed_dir=processed,
        annotated_dir=annotated,
        md_dir=md,
        uploads_dir=uploads,
        classify_domain=False,
    )

    assert out, "aucun JSON annoté produit"
    for json_path in out:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data, "bloc metadata manquant"
        assert data["metadata"]["doc_name"], "doc_name manquant"
        assert data["metadata"]["date_parution"], "date_parution manquante"
        assert "keyword_counts" in data, "keyword_counts document manquant"
        assert data["keyword_counts"]["per_category"], "catégories vides"
        instruments = data.get("instruments") or []
        assert instruments, "aucun instrument détecté"
        for instr in instruments:
            assert "keyword_counts" in instr, "keyword_counts instrument manquant"
            assert instr.get("n_articles", 0) >= 1, "instrument sans articles"


def test_v2_pipeline_restores_v1_dir_constants(tmp_path):
    """Le pipeline v2 restaure les constantes du module v1 après son run."""
    import scripts.run_pipeline_complet as rpc

    before = {n: getattr(rpc, n) for n in
              ("INTERIM_DIR", "PROCESSED_DIR", "ANNOTATED_DIR", "ANNOTATED_MD_DIR")}
    tmp = tmp_path / "x"
    tmp.mkdir()
    try:
        # Pas de PDF : le pipeline s'arrête tôt mais le finally doit passer.
        process_pdf(tmp / "missing.pdf", interim_dir=tmp / "i",
                    processed_dir=tmp / "p", annotated_dir=tmp / "a",
                    md_dir=tmp / "m", uploads_dir=tmp / "u")
    except Exception:
        pass
    after = {n: getattr(rpc, n) for n in
             ("INTERIM_DIR", "PROCESSED_DIR", "ANNOTATED_DIR", "ANNOTATED_MD_DIR")}
    assert before == after, "constantes v1 non restaurées"