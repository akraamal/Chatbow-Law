"""
test_metadata.py
----------------
Régression pour adli_v2.metadata : le post-enrichissement ajoute le bloc
metadata (doc_name, bo_number, date_parution) et les compteurs de
mots-clés au niveau document ET instrument, sans toucher aux champs v1.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from adli_v2.metadata import (
    add_keyword_counts,
    build_document_metadata,
    post_enrich,
)


def _sample_data() -> dict:
    return {
        "doc_id": "fr_BO_7480_Fr",
        "lang": "fr",
        "bo_number": "7480",
        "bo_number_source": "filename+header",
        "bo_number_confidence": "high",
        "date_publication": "2026-04-16",
        "preamble_text": "Décret n° 2-20-716 relatif aux performances énergétiques.",
        "articles": [
            {"number": "premier", "raw_header": "Article premier", "text": "Le présent décret fixe le travail salarié."},
            {"number": "2", "raw_header": "Art. 2", "text": "Toute infraction est punie d'une amende."},
        ],
        "instruments": [
            {
                "instrument_type": "DECRET",
                "reference": "2-20-716",
                "n_articles": 2,
                "article_indices": [0, 1],
                "title": "Décret n° 2-20-716",
            },
        ],
    }


def test_build_document_metadata():
    meta = build_document_metadata(_sample_data())
    assert meta["doc_name"] == "fr_BO_7480_Fr"
    assert meta["bo_number"] == "7480"
    assert meta["date_parution"] == "2026-04-16"
    assert meta["n_articles"] == 2
    assert meta["n_instruments"] == 1


def test_add_keyword_counts_document_and_instrument():
    data = _sample_data()
    add_keyword_counts(data)
    assert "per_term" in data["keyword_counts"]
    assert data["keyword_counts"]["per_category"]["Fiscal"] == 0
    inst = data["instruments"][0]
    assert "per_category" in inst["keyword_counts"]


def test_post_enrich_roundtrip(tmp_path):
    path = tmp_path / "fr_BO_7480_Fr_entities.json"
    path.write_text(json.dumps(_sample_data()), encoding="utf-8")
    data = post_enrich(path)
    assert data["metadata"]["doc_name"] == "fr_BO_7480_Fr"
    assert "keyword_counts" in data
    with open(path, encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["metadata"]["bo_number"] == "7480"
    assert reloaded["instruments"][0]["keyword_counts"]["per_category"]


def test_instrument_text_falls_back_to_article_indices():
    from adli_v2.metadata import instrument_text

    data = _sample_data()
    text = instrument_text(data["instruments"][0], data["articles"])
    assert "travail salarié" in text
    assert "amende" in text


def test_post_enrich_preserves_v1_fields(tmp_path):
    path = tmp_path / "x_entities.json"
    data = _sample_data()
    data["instruments"][0]["signatories"] = [{"name": "X", "role": "Y"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    reloaded = post_enrich(path)
    assert reloaded["instruments"][0]["signatories"] == [{"name": "X", "role": "Y"}]
    assert reloaded["articles"][0]["text"] == "Le présent décret fixe le travail salarié."