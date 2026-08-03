"""
test_instrument_type_ar.py
---------------------------
Régression : le type et la référence des instruments ARABES du BO sont
extraits de leur titre/préambule (قرار / مرسوم / قانون … et « رقم X.Y.Z »).

Avant le correctif, _classify_instrument_type et _extract_reference ne
connaissaient que les mots-clés français (« Arrêté », « Décret », clauses
« Vu ») : pour les BO arabes, EVERY instrument était typé DECRET et toute
référence restait None (audit BO_7408 : 24/24 DECRET, 24/24 references null).

Usage:
    python -m pytest tests/test_instrument_type_ar.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _classify(text, has_article=True):
    from enrich_json_with_pages import _classify_instrument_type
    return _classify_instrument_type([{"text": "x"}] if has_article else [], text)


def _ref(text, itype):
    from enrich_json_with_pages import _extract_reference
    return _extract_reference(text, itype)


def test_ar_type_classified_from_preamble():
    cases = {
        "مرسوم رقم 2.23.1143 صادر في 24 من ذي القعدة 1446": "DECRET",
        "قرار لوزير الفلاحة والصيد البحري رقم 731.25 صادر في 16 من رمضان":
            "ARRETE",
        "قرار مشترك لوزير الداخلية ووزير الصحة رقم 1100.25": "ARRETE_CONJOINT",
        "قانون رقم 38.15 المتعلق بالتنظيم القضائي": "LOI",
        "ظهير شريف رقم 1.09.20 صادر في 22 صفر 1430": "DAHIR",
        "منشور رقم 4.22": "CIRCULAIRE",
        "مقرر رقم 123.22": "DECISION",
    }
    for preamble, expected in cases.items():
        assert enrich_classify(preamble) == expected, f"{preamble} -> {enrich_classify(preamble)}"


def enrich_classify(preamble):
    from enrich_json_with_pages import _classify_instrument_type
    return _classify_instrument_type([], preamble)


def test_ar_reference_from_preamble():
    cases = {
        "مرسوم رقم 2.23.1143 صادر في 24 من ذي القعدة 1446": "2.23.1143",
        "قرار لوزير الفلاحة رقم 731.25 صادر في 16 من رمضان": "731.25",
        "مرسوم رقم 2.25.269 صادر في 21 من ذي القعدة": "2.25.269",
        "قرار لوزيرة الاقتصاد والمالية رقم 1149.25 صادر في 9 ذي القعدة": "1149.25",
    }
    for preamble, expected in cases.items():
        itype = enrich_classify(preamble)
        got = _ref(preamble, itype)
        assert got == expected, f"{preamble!r}: {got!r} != {expected!r}"


def test_ar_reference_skips_cross_reference():
    """Le numéro propre précède la première référence croisée : on ne doit
    PAS capturer « 1.60.063 » (dahir cité) d'un قرار d'approbation qui n'a
    pas de numéro propre."""
    preamble = (
        "قرار لعامل اقليم جرادة باقرار مخطط تنمية الكتلة العمرانية "
        "لركز جماعة كفايت باقليم جرادة. "
        "بناء على الظهير الشريف رقم 1.60.063 الصادر في 30 من ذي الحجة 1379"
    )
    itype = enrich_classify(preamble)
    assert itype == "ARRETE"
    assert _ref(preamble, itype) is None


def test_fr_still_classified():
    """Non-régression : la logique FR (Arrêté/Décret/Dahir + clauses Vu)
    n'est pas affectée par la branche arabe."""
    preamble = (
        "Décret n° 2-23-1143 du 6 février 2023 fixant les dispositions "
        "générales. Vu la loi n° 49-19, Vu le dahir n° 1-58-198"
    )
    assert enrich_classify(preamble) == "DECRET"
    assert _ref(preamble, "DECRET") == "2-23-1143"