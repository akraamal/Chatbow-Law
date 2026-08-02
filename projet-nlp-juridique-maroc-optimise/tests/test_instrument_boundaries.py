"""
test_instrument_boundaries.py
-----------------------------
Régression : les frontières d'instruments (loi/décret/arrêté) du BO ne sont
pas des artefacts de regex mais des décisions structurelles du segmenter
(get_per_decree_preamble_map, src/preprocessing/segmenter.py) : chaque
frontière est détectée dans le « gap » entre deux articles via
DOCUMENT_TITLE_PATTERN_FR + _is_doc_title_match (tolérance aux titres
minusculisés par l'extraction PDF).

Cas gelé (BO_7480_Fr.pdf, vérifié à la main) : l'entrée « Dahir » décimait
en un seul groupe les 5 articles de la loi n° 44-22 (experts judiciaires)
ET les 7 articles du décret n° 2-20-716 (performances énergétiques) —
le titre minuscule « décret n° 2-20-716 ... » n'était pas reconnu comme
frontière.  Depuis le correctif, decrees[1] est la loi 44-22 et decrees[2]
le décret 2-20-716, avec le partage exact 5/7.

Usage:
    python -m pytest tests/test_instrument_boundaries.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

FR_PDF = Path("data/raw/fr/BO_7480_Fr.pdf")


@pytest.mark.skipif(
    not FR_PDF.exists(), reason="BO_7480_Fr.pdf absent du dépôt (données gitignorées)"
)
def test_bo7480_dahir_decret_boundary(tmp_path, monkeypatch):
    """Le Dahir n° 1-23-61 (loi 44-22, 5 articles) et le décret n° 2-20-716
    (7 articles) doivent être deux instruments séparés, pas un seul groupe."""
    import scripts.run_pipeline_complet as rpc

    for name in ("INTERIM_DIR", "PROCESSED_DIR", "ANNOTATED_DIR", "ANNOTATED_MD_DIR"):
        monkeypatch.setattr(rpc, name, tmp_path / name)

    rpc.process_single_pdf(FR_PDF)

    out = tmp_path / "ANNOTATED_DIR" / "fr_BO_7480_Fr_entities.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    decs = data["decrees"]

    dahir = decs[1]
    assert dahir["first_article_idx"] == 105, "frontière Dahir 1-23-61 déplacée"
    assert "dahir" in dahir["title"].lower(), f"titre inattendu : {dahir['title']!r}"
    assert "1-23-61" in dahir["preamble"], "préambule du Dahir 1-23-61 inattendu"

    decreet = decs[2]
    assert decreet["first_article_idx"] == 110, \
        "frontière décret 2-20-716 manquée (articles avalés par le Dahir)"
    assert "décret" in decreet["title"].lower(), f"titre inattendu : {decreet['title']!r}"
    assert "2-20-716" in decreet["preamble"], "préambule du décret 2-20-716 inattendu"

    arr = decs[3]
    assert arr["first_article_idx"] == 117, "frontière Arrêté 1529-24 déplacée"

    arts = data["articles"]
    loi_nums = [arts[j]["number"] for j in range(105, 110)]
    dec_nums = [arts[j]["number"] for j in range(110, 117)]
    assert loi_nums == ["premier", "2", "3", "4", "5"], f"articles loi 44-22 : {loi_nums}"
    assert dec_nums == ["PREMIER", "2", "3", "4", "5", "6", "7"], \
        f"articles décret 2-20-716 : {dec_nums}"
