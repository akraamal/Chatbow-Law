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
    "edition_label": "Test",
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
            "instrument_type": "DECRET",
            "reference": "2-26-100",
            "reference_label": "décret n° 2-26-100",
            "title": "décret portant institution d'un test",
            "decree_date_gregorian": "2026-06-15",
            "n_articles": 2,
            "article_indices": [0, 1],
            "keyword_counts": {"per_term": {"impôt": 1}},
            "signatories": [
                {"role": "Chef du Gouvernement", "name": "AZIZ AKHANNOUCH", "type": "issuer"},
                {"role": "La ministre de l'économie et des finances", "name": "NADIA FETTAH", "type": "contreseing"},
            ],
            "signatories_flat": ["AZIZ AKHANNOUCH", "NADIA FETTAH"],
        },
        {
            "instrument_id": "inst-1",
            "instrument_type": "ARRETE",
            "reference": "3-26-1",
            "n_articles": 1,
            "article_indices": [0],
        },
        {
            "instrument_id": "inst-2",
            "instrument_type": "DECRET",
            "reference": "2-22-1020",
            "reference_label": "décret n° 2-22-1020",
            "title": "décret fixant la liste des équipements",
            "decree_date_gregorian": "2025-11-27",
            "n_articles": 1,
            "article_indices": [1],
            "signatories": [
                {"role": "Chef du Gouvernement", "name": "AZIZ AKHANNOUCH", "type": "issuer"},
                {"role": "La ministre de l'économie et des finances", "name": "NADIA FETTAH", "type": "contreseing"},
            ],
            "signatories_flat": ["AZIZ AKHANNOUCH", "NADIA FETTAH"],
        },
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
    assert resp["n_instruments"] == 2
    assert resp["date_publication"] == "2026-08-01"
    assert resp["keyword_counts"]["per_category"]["Fiscal"] == 4

    inst = resp["instruments"][0]
    assert inst["instrument_type"] == "DECRET"
    assert inst["reference"] == "2-26-100"
    assert inst["title"] == "décret portant institution d'un test"
    assert inst["decree_date_gregorian"] == "2026-06-15"
    assert inst["keyword_counts"]["per_term"]["impôt"] == 1
    assert inst["signatories"][0]["role"] == "Chef du Gouvernement"
    assert inst["signatories"][0]["name"] == "AZIZ AKHANNOUCH"
    assert inst["signatories_flat"] == ["AZIZ AKHANNOUCH", "NADIA FETTAH"]
    assert len(inst["articles"]) == 2
    assert inst["articles"][0]["number"] == "1"
    assert inst["articles"][0]["entities"][0]["label"] == "DECRET"
    assert inst["articles"][0]["entities"][0]["color"] == analyzer_mod.ENTITY_COLORS["DECRET"]

    assert resp["entity_counts"][0] == {
        "label": "DECRET", "count": 1, "color": analyzer_mod.ENTITY_COLORS["DECRET"],
    }
    assert resp["preamble_entities"][0]["label"] == "LOI"
    assert "tables" in inst and inst["tables"] == []


def test_build_response_filters_non_decrees(annotated_dir):
    p = _write_sample(annotated_dir)
    data = json.loads(p.read_text(encoding="utf-8"))
    resp = analyzer_mod.build_response(data)
    assert all(
        i["instrument_type"] in ("DECRET", "DECRET_LOI")
        for i in resp["instruments"]
    )
    assert resp["n_instruments"] == 2


def test_analyses_lists_annotated_dir(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    r = client.get("/analyses")
    assert r.status_code == 200
    entries = r.get_json()["analyses"]
    assert len(entries) == 1
    assert entries[0]["doc_id"] == "BO_9999_Fr"
    assert entries[0]["n_instruments"] == 2
    assert entries[0]["n_articles"] == 2


def test_open_analysis_returns_v1_shape(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    r = client.get("/open-analysis/BO_9999_Fr")
    assert r.status_code == 200
    body = r.get_json()
    assert body["doc_id"] == "BO_9999_Fr"
    assert body["n_instruments"] == 2
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
    assert "DECRET 2-26-100" in r.get_json()["answer"]


def test_chat_finds_decree_any_phrasing(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    cases = [
        ("contenu de decret 2.26.100", "2-26-100"),
        ("contenu du décret 2-26-100", "2-26-100"),
        ("décret 2 26 100", "2-26-100"),
        ("que dit le décret n° 2.26.100 ?", "2-26-100"),
        ("what is the content of decree 2.26.100", "2-26-100"),
        ("رقم ٢.٢٦.١٠٠", "2-26-100"),
        ("numéro 2-26-100", "2-26-100"),
        ("combien d'articles dans le décret 2.26.100 ?", "2-26-100"),
        ("que dit le décret 2.22.1020 ?", "2-22-1020"),
        ("décret 2.22.1020", "2-22-1020"),
    ]
    for question, expected in cases:
        r = client.post("/chat", json={"question": question, "doc_id": "BO_9999_Fr"})
        assert r.status_code == 200
        answer = r.get_json()["answer"]
        assert expected in answer, f"{question!r} -> {answer!r}"


def test_chat_content_answer_has_title_and_articles(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    r = client.post("/chat", json={"question": "contenu de decret 2.26.100", "doc_id": "BO_9999_Fr"})
    answer = r.get_json()["answer"]
    assert "portant institution d'un test" in answer
    assert "2026-06-15" in answer
    assert "Article 1" in answer
    assert "Article 2" in answer


def test_chat_search_articles_digit_insensitive(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    r = client.post("/chat", json={"question": "recherche 2.26.100", "doc_id": "BO_9999_Fr"})
    assert r.status_code == 200
    assert "Article 1" in r.get_json()["answer"]


def test_chat_who_signed_a_decree(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    for question in [
        "qui a signé le décret 2.26.100 ?",
        "qui signe le décret 2.26.100",
        "signataires du décret 2-26-100",
        "signé par qui le décret 2.26.100",
        "من وقع المرسوم 2.26.100",
    ]:
        r = client.post("/chat", json={"question": question, "doc_id": "BO_9999_Fr"})
        answer = r.get_json()["answer"]
        assert "2-26-100" in answer, f"{question!r} -> {answer!r}"
        assert "Chef du Gouvernement" in answer
        assert "AZIZ AKHANNOUCH" in answer
        assert "NADIA FETTAH" in answer


def test_chat_which_decree_a_person_signed(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    for question in [
        "qu'a signé Aziz Akhannouch ?",
        "quel décret a signé nadia fettah",
        "décrets signés par le chef du gouvernement",
        "ما الذي وقع عليه عزيز أخنوش",
    ]:
        r = client.post("/chat", json={"question": question, "doc_id": "BO_9999_Fr"})
        answer = r.get_json()["answer"]
        assert "2-26-100" in answer, f"{question!r} -> {answer!r}"
        assert "AZIZ AKHANNOUCH" in answer or "NADIA FETTAH" in answer


def test_chat_count_signers(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    r = client.post("/chat", json={"question": "combien de signataires pour le décret 2.26.100 ?", "doc_id": "BO_9999_Fr"})
    answer = r.get_json()["answer"]
    assert "2 personne(s)" in answer


def test_documents_and_keywords_endpoints(annotated_dir):
    _write_sample(annotated_dir)
    client = app.test_client()

    r = client.get("/documents")
    docs = r.get_json()["documents"]
    assert len(docs) == 1
    assert docs[0]["n_instruments"] == 2

    r = client.get("/document/BO_9999_Fr")
    assert r.status_code == 200
    assert r.get_json()["keyword_counts"]["per_term"]["impôt"] == 3

    r = client.get("/keywords")
    body = r.get_json()
    assert body["n_documents"] == 1
    assert body["per_category"]["Fiscal"] == 4
    assert body["top_terms"][0] == ["impôt", 3]


# ── Analyse de contenu par LLM (repli au-delà de la cascade de règles) ──

def test_articles_as_rag_sources_mapping():
    data = {
        "doc_id": "BO_9999_Fr",
        "lang": "fr",
        "bo_number": "9999",
        "articles": [
            {"number": "1", "text": "Il est institué un DECRET n° 2-26-100 test.", "pdf_page": 3},
            {"number": "2", "text": "Les modalités d'application sont fixées par arrêté.", "pdf_page": 3},
            {"number": "3", "text": "La présente loi entre en vigueur à sa publication.", "pdf_page": 4},
        ],
        "instruments": [
            {"instrument_type": "DECRET", "reference": "2-26-100", "n_articles": 2, "article_indices": [0, 1]},
            {"instrument_type": "LOI", "reference": "1-93-153", "n_articles": 1, "article_indices": [2]},
        ],
    }
    srcs = analyzer_mod._articles_as_rag_sources(data)
    assert len(srcs) == 3
    a1, a2, a3 = srcs
    assert a1["article_number"] == "1"
    assert a1["doc_id"] == "BO_9999_Fr"
    assert a1["bo_number"] == "9999"
    assert a1["instrument_type"] == "DECRET"
    assert a1["reference"] == "2-26-100"
    assert a1["text"].startswith("Il est institué")
    assert a2["reference"] == "2-26-100"
    assert a3["instrument_type"] == "LOI"
    assert a3["reference"] == "1-93-153"


def test_llm_analysis_answer_synthesis_path(monkeypatch, annotated_dir):
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, system_instruction, user_prompt):
            assert "article 1" in user_prompt or "Article 1" in user_prompt
            return ("Ce document institue un décret portant un test.\n\n"
                    "[[GROUNDED-IN]]\nSource 1, Source 2\n[[END]]")

    monkeypatch.setattr(llm_mod, "LLMClient", FakeLLM)
    ans = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, "Décris ce document")
    assert "Ce document institue" in ans
    assert "[[GROUNDED-IN]]" not in ans
    assert "📄 Sources" in ans
    assert "art. 1" in ans


def test_llm_analysis_answer_factual_path(monkeypatch):
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def generate_with_citation_guarantee(self, system_instruction, user_prompt):
            return ("Le décret n° 2-26-100 institue un test.\n\n"
                    "[[CITATIONS]]\n«Il est institué un DECRET n° 2-26-100» [Source 1]\n[[END]]")

    monkeypatch.setattr(llm_mod, "LLMClient", FakeLLM)
    ans = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, "que prévoit le décret ?")
    assert "institue un test" in ans
    assert "[[CITATIONS]]" not in ans
    assert "📄" in ans


def test_llm_analysis_answer_refusal(monkeypatch):
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")

    class RefuseLLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, system_instruction, user_prompt):
            return ("Je ne peux répondre qu'à partir des documents chargés dans le corpus "
                    "indexé — cette question sort de leur contenu.\n\n[[GROUNDED-IN]]\n[[END]]")

        def generate_with_citation_guarantee(self, system_instruction, user_prompt):
            return ("Je ne peux répondre qu'à partir des documents chargés dans le corpus "
                    "indexé — cette question sort de leur contenu.\n\n[[CITATIONS]]\n[[END]]")

    monkeypatch.setattr(llm_mod, "LLMClient", RefuseLLM)
    ans = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, "quelle est la météo ?")
    assert "Je ne peux répondre" in ans


def test_chat_routes_unknown_question_to_llm(annotated_dir, monkeypatch):
    _write_sample(annotated_dir)
    calls = {"n": 0}

    def fake_llm_answer(data, question):
        calls["n"] += 1
        return "réponse LLM testée"

    monkeypatch.setattr(analyzer_mod, "_llm_analysis_answer", fake_llm_answer)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    r = client.post("/chat", json={"question": "comment s'applique le régime fiscal ?", "doc_id": "BO_9999_Fr"})
    assert r.status_code == 200
    assert r.get_json()["answer"] == "réponse LLM testée"
    assert calls["n"] == 1

    # Une question couverte par les règles reste sur les règles (pas de LLM)
    r = client.post("/chat", json={"question": "Combien d'articles ?", "doc_id": "BO_9999_Fr"})
    assert "2 articles" in r.get_json()["answer"]
    assert calls["n"] == 1


def test_chat_deep_analysis_action(annotated_dir, monkeypatch):
    _write_sample(annotated_dir)
    seen = {}

    def fake_llm_answer(data, question):
        seen["q"] = question
        return "résumé du document"

    monkeypatch.setattr(analyzer_mod, "_llm_analysis_answer", fake_llm_answer)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    r = client.post("/chat", json={
        "question": "Décris en détail ce que traite ce document, article par article, avec pour chaque point la page correspondante.",
        "doc_id": "BO_9999_Fr",
    })
    assert r.status_code == 200
    assert r.get_json()["answer"] == "résumé du document"
    assert "Décris en détail" in seen["q"]