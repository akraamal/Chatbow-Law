"""
test_hijri_date_digit_variants.py
---------------------------------
Régression : les mois hégiriens à deux parties « joumada 1/2 » et
« rabii 1/2 » (chiffres, variante OCR/corpus fréquente — 68 occurrences
« joumada N » et 72 « rabii N » dans data/processed/) doivent être
reconnus à l'égal des formes en chiffres romains (« joumada i/ii »).

Avant le correctif, seuls les chiffres romains étaient reconnus :
« décret n° 2-19-40 du 17 joumada 1 1440 » (BO_6758_Fr.txt) restait
sans conversion hégirienne.

Usage:
    python -m pytest tests/test_hijri_date_digit_variants.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _parse_fr(text):
    from src.utils.hijri_calendar import parse_hijri_date_fr
    return parse_hijri_date_fr(text)


def _to_gregorian(triplet):
    from src.utils.hijri_calendar import hijri_to_gregorian
    return hijri_to_gregorian(*triplet)


def test_joumada_digit_variants():
    """« joumada 1 » = mois 5, « joumada 2 » = mois 6 (comme i/ii)."""
    assert _parse_fr("17 joumada 1 1440 (24 janvier 2019)") == (1440, 5, 17)
    assert _parse_fr("20 joumada 2 1440") == (1440, 6, 20)
    # Formes romaines inchangées
    assert _parse_fr("17 joumada I 1440") == (1440, 5, 17)
    assert _parse_fr("21 joumada II 1440") == (1440, 6, 21)


def test_rabii_digit_variants():
    """« rabii 1 » = mois 3, « rabii 2 » = mois 4 (comme i/ii)."""
    assert _parse_fr("7 rabii 1 1440 (15 novembre 2018)") == (1440, 3, 7)
    assert _parse_fr("5 rabii 2 1440") == (1440, 4, 5)
    assert _parse_fr("5 rabii ii 1440") == (1440, 4, 5)


def test_ocr_double_digit_ii_forms():
    """« joumada 11 »/« rabii 11 » = variante OCR du « II » romain
    (255 occurrences « joumada 11 » et 10 « rabii 11 » dans data/processed/,
    ex. « 6 joumada 11 1439 (23 février 2018) » de BO_6718_Fr)."""
    assert _parse_fr("6 joumada 11 1439 (23 février 2018)") == (1439, 6, 6)
    assert _parse_fr("26 rabii 11 1440 (3 janvier 2019)") == (1440, 4, 26)
    # « 11 » ne doit pas être lu comme jour : « 11 joumada 11 1439 »
    assert _parse_fr("11 joumada 11 1439") == (1439, 6, 11)


def test_gregorian_crosscheck_matches_bo():
    """La conversion tabulaire du cas du bug (BO_6758_Fr.txt, ligne 250)
    colle au millésime entre parenthèses du BO lui-même : 17 joumada 1
    1440 = 24 janvier 2019."""
    from datetime import date

    assert _to_gregorian(_parse_fr("17 joumada 1 1440")) == date(2019, 1, 24)


def test_dates_patterns_fr_digit_variants():
    """L'extraction d'entités de dates (dates_patterns) reconnaît aussi
    les formes chiffrées."""
    from src.extraction.dates_patterns import extract_dates_fr

    found = extract_dates_fr("du 17 joumada 1 1440 (24 janvier 2019)")
    hijri = [d for d in found if d.label == "DATE_HIJRI"]
    assert len(hijri) == 1
    assert hijri[0].meta["month"] == 5
    assert hijri[0].meta["month_name"] == "joumada 1"

    found2 = extract_dates_fr("du 7 rabii 1 1440")
    hijri2 = [d for d in found2 if d.label == "DATE_HIJRI"]
    assert len(hijri2) == 1 and hijri2[0].meta["month"] == 3

    found3 = extract_dates_fr("du 6 joumada 11 1439")
    hijri3 = [d for d in found3 if d.label == "DATE_HIJRI"]
    assert len(hijri3) == 1 and hijri3[0].meta["month"] == 6


def test_single_part_months_unchanged():
    """Non-régression : les mois à une partie et les autres formes ne
    sont pas affectés."""
    assert _parse_fr("20 kaada 1440") == (1440, 11, 20)
    assert _parse_fr("1er chaabane 1447") == (1447, 8, 1)
    assert _parse_fr("22 safar 1430") == (1430, 2, 22)


def test_jourmada_typo_variant():
    """« jourmada » (typo fréquente du BO_6758, ex. formule de clôture du
    décret 2-19-40 : « Fait à Rabat, le 17 jourmada 1 1440 (24 janvier
    2019) ») est reconnu comme joumada ; « jourmada it » = ii."""
    assert _parse_fr("17 jourmada 1 1440 (24 janvier 2019)") == (1440, 5, 17)
    assert _parse_fr("21 jourmada it 1440 (27 février 2019)") == (1440, 6, 21)