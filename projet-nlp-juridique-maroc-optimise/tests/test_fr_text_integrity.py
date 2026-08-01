"""
test_fr_text_integrity.py
--------------------------
Régression : l'intégrité du TEXTE français BO_7522 (ce qui est embeddé et
montré au LLM) est le produit critique du pipeline :

- les séquences de points 4+ en pleine ligne (convention d'élision des
  amendements BO : « texte inchangé, omis ici ») sont remplacées par un
  marqueur explicite, jamais réduites à un espace ;
- la clause du cahier des charges SNRT citée en arabe est CONSERVÉE dans le
  préambule de son décret (aucune perte silencieuse) ;
- le collecteur possible_embedded_arabic ne garde que le vrai résidu
  (["؛", "؛"] — les points-virgules hors guillemets) ;
- les entités : ACLAB capturé via le déclencheur "site « ... »", et la
  sparsité article reste honnête (50 articles zéro-entité = prose).

Le test pipeline écrit ses sorties dans tmp_path (constantes du module
monkeypatchées) pour ne pas toucher à data/.

Usage:
    python -m pytest tests/test_fr_text_integrity.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

FR_PDF = Path("data/raw/fr/BO_7522_Fr.pdf")

ELISION_MARKER = " […texte non modifié…] "
SNRT_CLAUSE = "تمتنع الشركة"


def test_elision_dots_preserved():
    """Les points d'élision en pleine ligne deviennent un marqueur explicite ;
    les lignes de sommaire à points de suite sont retirées."""
    from preprocessing.cleaner_fr import clean_french_text

    sample = (
        "Edition générale...................790\n"
        "L'opération ................................ recours est maintenue.\n"
        "Texte normal."
    )
    out = clean_french_text(sample)

    assert "Edition générale" not in out, "ligne de sommaire non retirée"
    assert ELISION_MARKER in out, "marqueur d'élision absent"
    assert "L'opération" in out and "recours est maintenue" in out


def test_bo7522_pipeline_integrity(tmp_path, monkeypatch):
    """Pipeline complet canonique sur BO_7522_Fr.pdf, sorties écrites dans
    tmp_path : intégrité du texte (SNRT, élision), collecteur arabe,
    découpage en articles, ACLAB capturé, sparsité honnête."""
    if not FR_PDF.exists():
        pytest.skip("BO_7522_Fr.pdf absent du dépôt (données gitignorées)")

    import scripts.run_pipeline_complet as rpc

    for name in ("INTERIM_DIR", "PROCESSED_DIR", "ANNOTATED_DIR", "ANNOTATED_MD_DIR"):
        monkeypatch.setattr(rpc, name, tmp_path / name)

    rpc.process_single_pdf(FR_PDF)

    out = tmp_path / "ANNOTATED_DIR" / "fr_BO_7522_Fr_entities.json"
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["n_articles"] == 138, "découpage en articles modifié"

    assert data["possible_embedded_arabic"] == ["؛", "؛"], \
        "collecteur d'arabe cité inattendu"

    assert any(SNRT_CLAUSE in dec.get("preamble", "") for dec in data["decrees"]), \
        "clause SNRT perdue du préambule de décret"

    assert any(ELISION_MARKER in a["text"] for a in data["articles"]), \
        "marqueur d'élision absent des articles"

    orgs = [
        e["text"]
        for a in data["articles"]
        for e in a["entities"]
        if e["label"] == "ORG"
    ]
    assert any("Analysis and Control Laboratory (ACLAB)" in t for t in orgs), \
        "ACLAB non capturé (déclencheur « site ... »)"

    zero = sum(1 for a in data["articles"] if not a["entities"])
    assert zero == 50, f"nombre de zéro-entité inattendu : {zero}"
