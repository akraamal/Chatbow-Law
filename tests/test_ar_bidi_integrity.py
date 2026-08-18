"""
test_ar_bidi_integrity.py
--------------------------
Régression round 4 (corrigée) : dans le rawdict des PDF arabes, les
chiffres sont physiquement LTR — les runs chiffres/ponctuation purs des
lignes RTL doivent être re-séquencés en ordre logique arabe, les runs
contenant des lettres latines (« NM 01.4.510 ») laissés tels quels, et les
parenthèses d'une ligne RTL ne doivent PAS être miroirées.

Vérifié à la main contre le PDF BO_7515_Ar.pdf (pages 7-11) : avant le
correctif, "943.26" sortait inversé en "62.349", "855.26" en "62.558",
"165.009,80" en "08,900.561", et "(2 avril 2026)" en ")2 أبريل 6202(".

Usage:
    python -m pytest tests/test_ar_bidi_integrity.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

AR_PDF = Path("data/raw/ar/BO_7515_Ar.pdf")

pytestmark = pytest.mark.skipif(
    not AR_PDF.exists(), reason="BO_7515_Ar.pdf absent du dépôt (données gitignorées)"
)

# Formes logiques correctes présentes dans le document…
CORRECT_FORMS = [
    "943.26",          # montant, anciennement inversé en "62.349"
    "855.26",          # anciennement "62.558"
    "2.26.324",        # référence de décret, anciennement "62.558"-style
    "165.009,80",      # montant avec virgule décimale
    "(11 ماي 2026)",   # date grégorienne : parenthèses non miroirées
    "NM 01.4.510",     # run mixte latin+chiffres : laissé tel quel
]
# …et leurs formes inversées (le bug) doivent rester absentes.
REVERSED_FORMS = ["62.349", "62.558", "08,900.561"]


def test_ar_digits_in_logical_order():
    """L'extraction du PDF arabe doit produire les nombres dans l'ordre
    logique (chiffres LTR physiques re-rangés), sans miroir des
    parenthèses, et préserver les runs mixtes latin/chiffres."""
    from ingestion.pipeline import run_ingestion_pipeline

    result = run_ingestion_pipeline(str(AR_PDF))
    text = result.text_ar
    assert text, "aucun texte arabe extrait"

    for s in CORRECT_FORMS:
        assert s in text, f"forme logique absente du texte extrait : {s!r}"

    for s in REVERSED_FORMS:
        assert s not in text, f"forme inversée (bug RTL) encore présente : {s!r}"
