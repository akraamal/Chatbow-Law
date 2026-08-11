"""
test_sommaire_ar.py
--------------------
Régression : le sommaire des éditions arabes du BO (en-tête « فهرست »)
doit être extrait comme celui des éditions FR (« SOMMAIRE »).

Avant le correctif, get_sommaire cherchait uniquement "SOMMAIRE" en dur
→ les éditions arabes renvoyaient toujours "".

Usage:
    python -m pytest tests/test_sommaire_ar.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

AR_PDF = Path("data/raw/ar/BO_7408_Ar.pdf")

pytestmark = pytest.mark.skipif(
    not AR_PDF.exists(), reason="BO_7408_Ar.pdf absent du dépôt (données gitignorées)"
)


def test_ar_sommaire_extracted():
    """Le sommaire arabe (« فهرست ») est détecté et restitue les entrées
    de la table des matières ; le contenu réel ne commence pas dedans."""
    from ingestion.pipeline import run_ingestion_pipeline
    from preprocessing.segmenter import (
        _filter_article_matches,
        _skip_sommaire,
        get_preamble,
        get_sommaire,
    )

    result = run_ingestion_pipeline(str(AR_PDF))
    text = result.text_ar
    assert text, "aucun texte arabe extrait"

    sommaire = get_sommaire(text, lang="ar")
    assert sommaire, "sommaire arabe non extrait (marqueur فهرست manquant ?)"

    # Le sommaire commence bien par l'en-tête « فهرست » et contient une
    # entrée réelle du BO_7408 (accord de prêt avec la Banque mondiale).
    assert sommaire.startswith("فهرست"), \
        f"le sommaire doit commencer par « فهرست » : {sommaire[:40]!r}"
    assert "البنك الدولي" in sommaire, "entrée du sommaire absente"

    # Le préambule (texte avant le premier article) ne doit pas inclure
    # le sommaire : il commence au premier marqueur de section
    # (« نصوص عامة ») situé hors de la liste de la table des matières.
    preamble = get_preamble(text, lang="ar")
    assert preamble, "préambule arabe vide"
    assert "فهرست" not in preamble[:200], \
        "le préambule ne doit pas contenir la table des matières"

    article_matches = _filter_article_matches(text, lang="ar")
    skip = _skip_sommaire(
        text,
        lang="ar",
        limit=article_matches[0].start() if article_matches else None,
    )
    assert skip > 0, "aucun marqueur de section trouvé après le sommaire"
    assert "نصوص عامة" in text[skip - 15:skip + 15], \
        "le contenu réel doit s'ouvrir sur un marqueur de section arabe"


def test_fr_sommaire_still_extracted():
    """Non-régression : le sommaire FR (« SOMMAIRE ») continue de
    fonctionner après le passage des marqueurs en mode bilingue.
    Le marqueur de section est cherché à 500+ chars du titre (même
    logique que _skip_sommaire) : on reproduit ici la longueur d'une
    vraie table des matières."""
    from preprocessing.segmenter import get_sommaire

    entries = "\n".join(
        f"Dahir n° 1-18-0{i} du … \u2026{300 + i:03d}"
        for i in range(80)
    )
    sample = (
        "SOMMAIRE\n"
        "Dahirs.- Décrets et arrêtés ministériels.\n"
        f"{entries}\n"
        "TEXTES GÉNÉRAUX\n"
        "Arrêté n° 1197-26 du ministre …\n"
    )
    sommaire = get_sommaire(sample, lang="fr")
    assert sommaire.startswith("SOMMAIRE")
    assert "Dahirs" in sommaire
    assert "Arrêté n° 1197-26" not in sommaire, \
        "le contenu réel ne doit pas faire partie du sommaire"
