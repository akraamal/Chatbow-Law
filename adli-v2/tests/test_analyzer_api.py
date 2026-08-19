"""Tests de l'API analyseur v2 : contrat API identique à v1, backend v2."""

import json

import pytest

from adli_v2 import pipeline
from adli_v2.app import analyzer as analyzer_mod
from adli_v2.app.main import app


@pytest.fixture()
def annotated_dir(tmp_path, monkeypatch):
    d = tmp_path / "annotated"
    d.mkdir()
    monkeypatch.setattr(pipeline, "DEFAULT_ANNOTATED", d)
    monkeypatch.setattr(analyzer_mod, "DEFAULT_ANNOTATED", d)
    return d


SAMPLE_JSON = {
    "doc_id": "BO_9999_Fr",
    "lang": "fr",
    "bo_number": "9999",
    "date_publication": "2026-08-01",
    "preamble_text": "LOI n° 10-26 portant test",
    "preamble_entities": [
        {"label": "LOI", "start": 0, "end": 3, "text": "LOI"}
    ],
    "articles": [
        {
            "number": "1",
            "text": "Il est institué un DECRET n° 2-26-100 test.",
            "pdf_page": 3,
            "entities": [
                {"label": "DECRET", "start": 22, "end": 41, "text": "DECRET n° 2-26-100"},
            ],
        },
        {
            "number": "2",
            "text": "Les modalités d'application sont fixées par arrêté.",
            "pdf_page": 3,
            "entities": [],
        },
    ],
    "instruments": [
        {
            "instrument_id": "inst-0",
            "instrument_type": "LOI",
            "reference": "10-26",
            "n_articles": 2,
            "article_indices": [0, 1],
        }
    ],
    "keyword_counts": {
        "per_category": {"Fiscal": 4},
        "per_term": {"impôt": 3, "taxe": 1},
    },
}


def _write_sample(annotated_dir):
    p = annotated_dir / "BO_9999_Fr_entities.json"
    p.write_text(json.dumps(SAMPLE_JSON, ensure_ascii=False), encoding="utf-8")
    return p


def test_build_response_contract(annotated_dir):
    p = _write_sample(annotated_dir)
    data = json.loads(p.read_text(encoding="utf-8"))
    resp = analyzer_mod.build_response(data)

    assert resp["doc_id"] == "BO_9999_Fr"
    assert resp["bo_number"] == "9999"
    assert resp["n_articles"] == 2
    assert resp["n_instruments"] == 1

    inst = resp["instruments"][0]
    assert inst["instrument_type"] == "LOI"
    assert inst["reference"] == "10-26"
    assert len(inst["articles"]) == 2
    assert inst["articles"][0]["number"] == "1"
    assert inst["articles"][0]["entities"][0]["label"] == "DECRET"
    assert inst["articles"][0]["entities"][0]["color"] == analyzer_mod.ENTITY_COLORS["DECRET"]

    assert resp["entity_counts"][0] == {
        "label": "DECRET", "count": 1, "color": analyzer_mod.ENTITY_COLORS["DECRET"],
    }
    assert resp["preamble_entities"][0]["label"] == "LOI"
    assert "tables" in inst and inst["tables"] == []


def test_analyses_lists_annotated_dir(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    r = client.get("/analyses")
    assert r.status_code == 200
    entries = r.get_json()["analyses"]
    assert len(entries) == 1
    assert entries[0]["doc_id"] == "BO_9999_Fr"
    assert entries[0]["n_instruments"] == 1
    assert entries[0]["n_articles"] == 2


def test_open_analysis_returns_v1_shape(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    r = client.get("/open-analysis/BO_9999_Fr")
    assert r.status_code == 200
    body = r.get_json()
    assert body["doc_id"] == "BO_9999_Fr"
    assert body["n_instruments"] == 1
    assert body["instruments"][0]["articles"][1]["number"] == "2"


def test_open_analysis_unknown_doc():
    client = app.test_client()
    r = client.get("/open-analysis/BO_0000_Inconnu")
    assert r.status_code == 404


def test_upload_rejects_non_pdf():
    client = app.test_client()
    r = client.post("/upload", data={"file": (b"hello", "fake.pdf")})
    assert r.status_code == 400


def test_chat_answers_from_context(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    r = client.post("/chat", json={"question": "Combien d'articles ?", "doc_id": "BO_9999_Fr"})
    assert r.status_code == 200
    assert "2 articles" in r.get_json()["answer"]

    r = client.post("/chat", json={"question": "Liste des instruments ?", "doc_id": "BO_9999_Fr"})
    assert "LOI 10-26" in r.get_json()["answer"]


def test_documents_and_keywords_endpoints(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()

    r = client.get("/documents")
    docs = r.get_json()["documents"]
    assert len(docs) == 1
    assert docs[0]["n_instruments"] == 1

    r = client.get("/document/BO_9999_Fr")
    assert r.status_code == 200
    assert r.get_json()["keyword_counts"]["per_term"]["impôt"] == 3

    r = client.get("/keywords")
    body = r.get_json()
    assert body["n_documents"] == 1
    assert body["per_category"]["Fiscal"] == 4
    assert body["top_terms"][0] == ["impôt", 3]