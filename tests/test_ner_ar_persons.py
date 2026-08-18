"""
tests/test_ner_ar_persons.py
Correctifs #6 (NER AR) :
  - spans nettoyés des ponctuations collées par le tokenizer camel-tools
    ("البيضاء." -> "البيضاء", "بن صالح:" -> "بن صالح") avec offsets
    cohérents (text[start:end] == entité) ;
  - toponymes marocains étiquetés PERSON par le NER rejetés par le
    gazetteer ("الفقيه بن صالح", "بني ملال", "سيدي قاسم", "الغرب").

Tests unitaires : n'exigent PAS camel-tools (pas de chargement du modèle).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.ner_statistical_ar import _bio_tags_to_spans
from src.extraction.gazetteer_filter import gazetteer_filter_persons


def _span(full, frag):
    start = full.find(frag)
    assert start >= 0, frag
    return {"text": frag, "start": start, "end": start + len(frag), "label": "PERSON"}


def test_span_trimming_trailing_colon():
    text = "صدر القرار وعليه؛ وقع بن صالح: والوزير محمد صديقي."
    tokens = text.split()
    # Le tokenizer camel-tools colle la ponctuation au token ("صالح:").
    labels = ["O"] * len(tokens)
    for i, t in enumerate(tokens):
        if t.startswith("بن"):
            labels[i] = "B-PERS"
        elif t.startswith("صالح:"):
            labels[i] = "I-PERS"

    persons, _ = _bio_tags_to_spans(tokens, labels, text)
    assert any(x["text"] == "بن صالح" for x in persons), persons
    for x in persons:
        assert x["text"] == text[x["start"]:x["end"]]


def test_span_trimming_org_trailing_period():
    text = "وحرر هذا القرار في الدار البيضاء. الموقع محمد بن صالح."
    tokens = text.split()
    labels = ["O"] * len(tokens)
    for i, t in enumerate(tokens):
        if t == "البيضاء.":
            labels[i] = "B-ORG"
        elif t.startswith("صالح."):
            labels[i] = "B-PERS"

    persons, orgs = _bio_tags_to_spans(tokens, labels, text)
    for x in persons + orgs:
        assert x["text"] == text[x["start"]:x["end"]]
    assert any(x["text"] == "البيضاء" for x in orgs), orgs


def test_gazetteer_filters_ar_toponyms():
    full = (
        "يعين بمجلس مدينة الفقيه بن صالح الغرب بني ملال جهة الدار "
        "البيضاء سطات. وقع محمد صديقي هذا القرار."
    )
    persons = [
        _span(full, "الفقيه"),
        _span(full, "بن صالح الغرب"),
        _span(full, "الغرب"),
        _span(full, "محمد صديقي"),
    ]
    kept = gazetteer_filter_persons(persons, full)
    assert [x["text"] for x in kept] == ["محمد صديقي"], kept


def test_gazetteer_keeps_real_person():
    full = "وقع محمد صديقي الوزير هذا القرار في الرباط."
    persons = [_span(full, "محمد صديقي")]
    kept = gazetteer_filter_persons(persons, full)
    assert [x["text"] for x in kept] == ["محمد صديقي"]