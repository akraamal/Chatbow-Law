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


FR_PDF_7510 = Path("data/raw/fr/BO_7510_Fr.pdf")


@pytest.mark.skipif(
    not FR_PDF_7510.exists(), reason="BO_7510_Fr.pdf absent du dépôt (données gitignorées)"
)
def test_bo7510_arrete_pairs_not_merged(tmp_path, monkeypatch):
    """Cas n=2 (BO_7510_Fr.pdf) : les courts arrêtés « équivalences de
    diplômes » (2 articles chacun : ARTICLE PREMIER + ART. 2) ne doivent PAS
    être fusionnés deux à deux.

    Cause racine vérifiée : page 18 du PDF, deux corps côte à côte (405-26
    colonne gauche, 406-26 colonne droite) ; une ancienne extraction
    empilait leurs deux titres avant le premier corps, si bien que le titre
    406-26 était consommé comme préambule du 405-26 et le second arrêté se
    retrouvait sans frontière.  Depuis l'ordre de lecture colonne par
    colonne (_order_blocks_for_reading), chaque titre précède son corps :
    405-26, 406-26, 407-26... sont des instruments séparés.
    """
    import scripts.run_pipeline_complet as rpc

    for name in ("INTERIM_DIR", "PROCESSED_DIR", "ANNOTATED_DIR", "ANNOTATED_MD_DIR"):
        monkeypatch.setattr(rpc, name, tmp_path / name)

    rpc.process_single_pdf(FR_PDF_7510)

    out = tmp_path / "ANNOTATED_DIR" / "fr_BO_7510_Fr_entities.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    decs = data["decrees"]

    ensup = [d for d in decs if d["preamble"].startswith(
        "Arrêté du ministre de l'enseignement supérieur")]
    nums = [d["preamble"].split("n° ")[1].split(" ")[0] for d in ensup]
    assert "405-26" in nums and "406-26" in nums, \
        f"arrêtés équivalences de diplômes manquants : {nums}"

    d405 = next(d for d in ensup if "405-26" in d["preamble"])
    d406 = next(d for d in ensup if "406-26" in d["preamble"])
    assert d406["first_article_idx"] - d405["first_article_idx"] == 2, \
        "arrêté 406-26 fusionné avec 405-26 (cas n=2)"

    indices = [d["first_article_idx"] for d in ensup]
    counts = [b - a for a, b in zip(indices, indices[1:])]
    counts.append(len(data["articles"]) - indices[-1])
    assert counts[:3] == [2, 2, 2], \
        f"arrêtés non découpés en 2 articles : {list(zip(nums, counts))}"


FR_PDF_7492 = Path("data/raw/fr/BO_7492_Fr.pdf")


@pytest.mark.skipif(
    not FR_PDF_7492.exists(), reason="BO_7492_Fr.pdf absent du dépôt (données gitignorées)"
)
def test_bo7492_arrete_pairs_not_merged(tmp_path, monkeypatch):
    """Cas n=2 (BO_7492_Fr.pdf) : les arrêtés « équivalences de diplômes »
    (168-26 chirurgie générale, 349-26 et 350-26 architecte) font chacun
    2 articles (ARTICLE PREMIER + ART. 2) et doivent rester trois
    instruments séparés, pas être fusionnés.
    """
    import scripts.run_pipeline_complet as rpc

    for name in ("INTERIM_DIR", "PROCESSED_DIR", "ANNOTATED_DIR", "ANNOTATED_MD_DIR"):
        monkeypatch.setattr(rpc, name, tmp_path / name)

    rpc.process_single_pdf(FR_PDF_7492)

    out = tmp_path / "ANNOTATED_DIR" / "fr_BO_7492_Fr_entities.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    decs = data["decrees"]

    ensup = [d for d in decs if d["preamble"].startswith(
        "Arrêté du ministre de l'enseignement supérieur")]
    nums = [d["preamble"].split("n° ")[1].split(" ")[0] for d in ensup]
    for expected in ("168-26", "349-26", "350-26"):
        assert expected in nums, \
            f"arrêté {expected} manquant : {nums}"

    positions = [decs.index(d) for d in ensup]
    counts = []
    for p in positions:
        end = (decs[p + 1]["first_article_idx"] if p + 1 < len(decs)
               else len(data["articles"]))
        counts.append(end - decs[p]["first_article_idx"])
    assert counts == [2, 2, 2], \
        f"arrêtés non découpés en 2 articles : {list(zip(nums, counts))}"
