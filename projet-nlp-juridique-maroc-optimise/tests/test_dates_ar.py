"""
test_dates_ar.py
-----------------
Régression : les dates grégoriennes arabes (« 28 فبراير 2025 ») doivent
être étiquetées DATE_GREGORIAN et non DATE_HIJRI ; les dates hégiriennes
(« 24 من ذي القعدة 1446 », « 22 صفر 1430 ») DATE_HIJRI.

Avant le correctif, le pattern _HIJRI_DATE était générique (jour + n'importe
quel mot + année) : il capturait aussi les dates grégoriennes sous le label
DATE_HIJRI, et comme il était appliqué avant extract_dates_ar (ordre de la
liste dans entities_to_spacy_doc), le span DATE_GREGORIAN correct était
ignoré comme chevauchant.

Usage:
    python -m pytest tests/test_dates_ar.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _sorted_labels(doc):
    return [(e.label_, e.text) for e in sorted(doc.ents, key=lambda s: s.start)]


def test_gregorian_date_labeled_not_hijri():
    from extraction.entity_ruler_builder_ar import extract_legal_entities_ar

    doc = extract_legal_entities_ar(
        "صادر في 28 فبراير 2025 بخصوص المشروع الثاني للتعريف."
    )
    labels = _sorted_labels(doc)
    assert ("DATE_GREGORIAN", "28 فبراير 2025") in labels, f"labels réels : {labels}"
    assert not any(
        label == "DATE_HIJRI" and "2025" in text for label, text in labels
    ), f"date grégorienne étiquetée hégirienne : {labels}"


def test_hijri_date_with_men_separator():
    from extraction.entity_ruler_builder_ar import extract_legal_entities_ar

    doc = extract_legal_entities_ar(
        "وعلى مرسوم رقم 2.23.1143 صادر في 24 من ذي القعدة 1446 (22 ماي 2025)"
    )
    labels = _sorted_labels(doc)
    assert ("DATE_HIJRI", "24 من ذي القعدة 1446") in labels, f"labels réels : {labels}"
    assert ("DATE_GREGORIAN", "22 ماي 2025") in labels, f"labels réels : {labels}"


def test_plain_hijri_date():
    from extraction.entity_ruler_builder_ar import extract_legal_entities_ar

    doc = extract_legal_entities_ar("المنشور بالجريدة الرسمية بتاريخ 25 شوال 1447")
    labels = _sorted_labels(doc)
    assert any(
        label == "DATE_HIJRI" and "شوال" in text for label, text in labels
    ), f"labels réels : {labels}"


def test_dahir_still_detected_with_hijri_date():
    """Non-régression : élargir une ceinture de sécurité sur _HIJRI_DATE ne
    doit pas casser la détection DAHIR qui intègre la date (« الصادر في »)."""
    from extraction.entity_ruler_builder_ar import extract_legal_entities_ar

    doc = extract_legal_entities_ar(
        "وعلى الظهير الشريف رقم 1.09.20 الصادر في 22 صفر 1430"
    )
    labels = _sorted_labels(doc)
    assert ("DAHIR", "الظهير الشريف رقم 1.09.20 الصادر في 22 صفر 1430") in labels, \
        f"labels réels : {labels}"