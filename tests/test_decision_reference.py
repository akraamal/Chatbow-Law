"""
tests/test_decision_reference.py
---------------------------------
Régression (BO_6758) : les « Décision du Wali de Bank Al-Maghrib n° 79,
80, 81 … » portent un numéro simple sans séparateur (pas « X-Y-Z »).
_extract_reference renvoyait None pour ces 5 instruments ; le préfixe
« n° » doit suffire à capturer le numéro propre.

Usage:
    python -m pytest tests/test_decision_reference.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _ref(text: str, itype: str):
    from enrich_json_with_pages import _extract_reference
    return _extract_reference(text, itype)


def test_decision_bam_single_number_reference():
    cases = [
        ("Décision du Wali de Bank Al-Maghrib n° 79 du 22 chaoual 1440 "
         "(27 février 2019) portant octroi d'un agrément à la société "
         "« Centre monétique interbancaire »", "79"),
        ("Décision du Wali de Bank Al-Maghrib n° 80 du 22 chaoual 1440 "
         "(27 février 2019) portant octroi d'un agrément en qualité "
         "d'établissement de paiement", "80"),
        ("Décision du Wali de Bank Al-Maghrib n° 81 du 22 chaoual 1440 "
         "(27 février 2019) portant octroi d'un agrément", "81"),
    ]
    for heading, expected in cases:
        got = _ref(heading, "DECISION")
        assert got == expected, f"{heading[:60]!r}: {got!r} != {expected!r}"


def test_decision_without_number_returns_none():
    """Sans préfixe « n° », un simple nombre isolé (jour, montant) ne doit
    pas être pris pour une référence."""
    heading = "Décision du Wali de Bank Al-Maghrib du 22 chaoual 1440 portant octroi d'un agrément"
    assert _ref(heading, "DECISION") is None


def test_decision_2_part_number_unchanged():
    """Non-régression : les numéros composés restent capturés par les
    patterns 2 parties existants."""
    heading = "Décision du directeur général de l'ANRT n° 1-2023 du 5 janvier 2023 portant approbation"
    assert _ref(heading, "DECISION") == "1-2023"


def test_arrete_reference_unchanged():
    """Non-régression : arrêtés/décrets/dahirs (numéros 2-3 parties) inchangés."""
    assert _ref("Arrêté de la ministre de l'économie et des finances n° 168-26 du 25 juin 2026", "ARRETE") == "168-26"
    assert _ref("Décret n° 2-25-1062 du 13 avril 2026 portant application de la loi n° 03-25", "DECRET") == "2-25-1062"
    assert _ref("Dahir n° 1-19-19 du 21 joumada II 1440 (27 février 2019) portant promulgation de la loi n° 20-19", "DAHIR") == "1-19-19"
