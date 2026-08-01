"""
test_sommaire_ordering.py
---------------------------
Régression : la page de garde des éditions FR (couverture/abonnements +
sommaire) n'est PAS ordonnée avec le même chemin que les pages de corps.

Le sommaire de BO_7522 (page 1) a 2 colonnes de contenu + un bandeau
d'abonnements 4 colonnes qui s'intercale en x : un split global de
colonnes échouait et intercalait les deux colonnes du sommaire
("Arrêté conjoint du ministre de l'industrie et du Cour
constitutionnelle. commerce et de la ministre...").

Usage:
    python -m pytest tests/test_sommaire_ordering.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PDF = Path("data/raw/fr/BO_7522_Fr.pdf")

pytestmark = pytest.mark.skipif(
    not PDF.exists(), reason="BO_7522_Fr.pdf absent du dépôt (données gitignorées)"
)


def test_sommaire_page_not_interleaved():
    """L'entrée « Arrêté conjoint ... n° 1164-26 ... » du sommaire doit
    être lisible sans bloc d'une autre colonne intercalé au milieu."""
    import fitz

    from ingestion.pdf_extractor import _extract_blocks_via_rawdict, _order_sommaire_page

    with fitz.open(str(PDF)) as doc:
        blocks = _extract_blocks_via_rawdict(doc[0])

    ordered = _order_sommaire_page(blocks)
    assert ordered is not None, "page 1 : sommaire non détecté"

    text = "\n".join(b[4] for b in ordered)
    # Texte brut du rawdict : normaliser apostrophes typographiques et
    # espaces insécables avant comparaison (le nettoyeur le fait aussi).
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\xa0", " ")

    # Les trois fragments de l'entrée « Arrêté conjoint n° 1164-26 » (colonne
    # droite du sommaire) doivent se suivre dans le bon ordre, SANS bloc de
    # la colonne gauche intercalé (régression observée avant le correctif :
    # "Arrêté conjoint du ministre de l'industrie et du Cour
    # constitutionnelle. commerce et de la ministre...").
    i = text.find("Arrêté conjoint du ministre de l")
    assert i != -1
    fragment = text[i:i + 500]
    assert "commerce et de la ministre de l'économie" in fragment
    assert "1164-26" in fragment
    assert "Cour constitutionnelle" not in fragment[:fragment.find("1164-26")], \
        "colonne gauche intercalée dans l'entrée de la colonne droite"

    # La colonne gauche (entrées Dahir) reste lisible avant la droite.
    i_cour = text.find("Cour constitutionnelle.")
    i_arr = text.find("Arrêté conjoint")
    assert i_cour != -1 and i_arr != -1
    assert i_cour < i_arr, "la colonne gauche du sommaire doit précéder la droite"


def test_quoted_arabic_preserved_in_cleaner():
    """Les citations arabes légitimes (entre guillemets français) sont
    conservées dans le texte nettoyé ; seuls les artefacts non cités sont
    retirés et collectés."""
    from preprocessing.cleaner_fr import clean_french_text

    sample = (
        "dispose que :\n"
        "«)...( تمتنع الشركة عن تقديم أي شكل من أشكال العرض )...(»\n"
        "et «الحقيقة في 90 دقيقة» sont diffusées.\n"
        "Ligne parasite. الحقي artefact de fin.\n"
    )
    runs = []
    out = clean_french_text(sample, arabic_runs=runs)

    assert "تمتنع الشركة" in out, "clause SNRT citée supprimée"
    assert "الحقيقة في 90 دقيقة" in out, "titre d'émission cité supprimé"
    assert "الحقي artefact" not in out, "artefact non cité conservé"
    assert runs == ["الحقي"], f"collecteur inattendu : {runs}"
