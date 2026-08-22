"""Tests de l'API analyseur v2 : contrat API identique à v1, backend v2."""

import json

import pytest

from adli_v2 import pipeline
from adli_v2.app import analyzer as analyzer_mod
from adli_v2.app.main import app


@pytest.fixture(autouse=True)
def _clear_chat_state():
    """État module du repli LLM vidé avant/après chaque test : cache,
    historique de conversation et compteur de budget quotidien — pour
    qu'aucun test n'hérite d'une réponse mockée ou d'un budget entamé."""
    analyzer_mod._analysis_cache.clear()
    analyzer_mod._chat_history.clear()
    analyzer_mod._deep_analysis_count_today.update(date=None, count=0)
    yield
    analyzer_mod._analysis_cache.clear()
    analyzer_mod._chat_history.clear()
    analyzer_mod._deep_analysis_count_today.update(date=None, count=0)


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

    def fake_llm_answer(data, question, doc_id=""):
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

    def fake_llm_answer(data, question, doc_id=""):
        seen["q"] = question
        return "résumé du document", None, None

    monkeypatch.setattr(analyzer_mod, "_llm_analysis_answer_with_meta",
                        fake_llm_answer)
    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")

    r = client.post("/chat", json={
        "question": "Décris en détail ce que traite ce document, article par article, avec pour chaque point la page correspondante.",
        "doc_id": "BO_9999_Fr",
    })
    assert r.status_code == 200
    assert r.get_json()["answer"] == "résumé du document"
    assert "Décris en détail" in seen["q"]


# ── Analyse profonde fragmentée (documents volumineux, ex. BO_7488-bis) ──

def _make_big_data(n_articles=40, text_len=400) -> dict:
    """Document dépassant largement le budget d'une seule requête."""
    articles = [
        {"number": str(i + 1),
         "text": f"Article {i + 1} contenu test " + "x" * text_len,
         "pdf_page": i // 10 + 3}
        for i in range(n_articles)
    ]
    half = n_articles // 2
    instruments = [
        {"instrument_type": "DECRET", "reference": "2-26-100",
         "n_articles": half, "article_indices": list(range(half))},
        {"instrument_type": "ARRETE", "reference": "3-26-1",
         "n_articles": n_articles - half, "article_indices": list(range(half, n_articles))},
    ]
    return {
        "doc_id": "BO_9999_Fr", "lang": "fr", "bo_number": "9999",
        "date_publication": "2026-08-01",
        "articles": articles, "instruments": instruments,
    }


def test_plan_analysis_chunks_respects_budget(monkeypatch):
    monkeypatch.setattr(analyzer_mod, "ANALYSIS_CHUNK_CONTEXT_CHARS", 1000)
    srcs = [{"text": "a" * 300} for _ in range(5)]   # coût ≈ 500/article → 2/paquet

    chunks, covered = analyzer_mod._plan_analysis_chunks(srcs)

    assert [len(c) for c in chunks] == [2, 2, 1]
    assert covered == 5
    # Aucun paquet ne dépasse le budget (marge incluse)
    for chunk in chunks:
        assert sum(analyzer_mod._article_context_len(a) for a in chunk) \
            <= analyzer_mod.ANALYSIS_CHUNK_CONTEXT_CHARS


def test_plan_analysis_chunks_caps_max_chunks(monkeypatch):
    monkeypatch.setattr(analyzer_mod, "ANALYSIS_CHUNK_CONTEXT_CHARS", 1000)
    monkeypatch.setattr(analyzer_mod, "ANALYSIS_MAX_CHUNKS", 2)
    srcs = [{"text": "a" * 300} for _ in range(7)]   # 2/paquet → 4 paquets nécessaires

    chunks, covered = analyzer_mod._plan_analysis_chunks(srcs)

    assert len(chunks) == 2
    assert covered == 4                              # les 3 derniers sont exclus


def _install_chunk_llm(monkeypatch, behavior):
    """Remplace LLMClient par une fausse classe pilotée par behavior(call_idx).
    Renvoie la liste des prompts reçus."""
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")
    prompts: list[str] = []

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, system_instruction, user_prompt):
            prompts.append(user_prompt)
            return behavior(len(prompts) - 1)

    monkeypatch.setattr(llm_mod, "LLMClient", FakeLLM)
    return prompts


def test_llm_analysis_answer_chunked_path(monkeypatch):
    data = _make_big_data(n_articles=40, text_len=400)
    srcs = analyzer_mod._articles_as_rag_sources(data)
    total = sum(analyzer_mod._article_context_len(a) for a in srcs)
    assert total > analyzer_mod.ANALYSIS_MAX_CONTEXT_CHARS   # précondition mode fragmenté

    def behave(idx):
        return ("Contenu de cette tranche du document.\n\n"
                "[[GROUNDED-IN]]\nSource 1\n[[END]]")

    prompts = _install_chunk_llm(monkeypatch, behave)
    chunks, covered = analyzer_mod._plan_analysis_chunks(srcs)

    ans = analyzer_mod._llm_analysis_answer(
        data, "Décris en détail ce que traite ce document, article par article.")

    assert len(prompts) == len(chunks) >= 2           # plusieurs requêtes séquentielles
    assert covered == len(srcs)                       # tout le document tient dans les paquets
    for chunk in chunks:                              # chaque section porte son en-tête
        title = analyzer_mod._analysis_section_title(chunk)
        assert title and title in ans
    assert "[[GROUNDED-IN]]" not in ans               # bloc d'ancrage retiré
    assert "📄 Sources" in ans
    assert "Analyse partielle" not in ans             # pas de note : couverture complète


def test_llm_analysis_answer_chunked_partial_coverage(monkeypatch):
    data = _make_big_data(n_articles=40, text_len=400)
    monkeypatch.setattr(analyzer_mod, "ANALYSIS_MAX_CHUNKS", 1)   # latence bornée

    prompts = _install_chunk_llm(
        monkeypatch,
        lambda idx: ("Résumé de la première tranche.\n\n"
                     "[[GROUNDED-IN]]\nSource 1\n[[END]]"))

    ans = analyzer_mod._llm_analysis_answer(
        data, "Décris en détail ce que traite ce document, article par article.")

    srcs = analyzer_mod._articles_as_rag_sources(data)
    chunks, covered = analyzer_mod._plan_analysis_chunks(srcs)
    assert len(prompts) == 1
    assert "Analyse partielle" in ans
    assert f"{covered} premiers articles couverts sur {len(srcs)}" in ans
    assert covered < len(srcs)


def test_llm_analysis_answer_chunk_failure_skips_section(monkeypatch):
    data = _make_big_data(n_articles=40, text_len=400)

    state = {"calls": 0}

    def behave(idx):
        state["calls"] += 1
        if state["calls"] == 1:                       # premier paquet en erreur
            raise RuntimeError("boom")
        return ("Tranche suivante.\n\n[[GROUNDED-IN]]\nSource 1\n[[END]]")

    _install_chunk_llm(monkeypatch, behave)
    srcs = analyzer_mod._articles_as_rag_sources(data)
    chunks, _cov = analyzer_mod._plan_analysis_chunks(srcs)

    ans = analyzer_mod._llm_analysis_answer(
        data, "Décris en détail ce que traite ce document, article par article.")

    assert ans.count("📄 Sources") == len(chunks) - 1   # section en erreur sautée
    assert "Tranche suivante" in ans


def test_llm_analysis_answer_chunk_all_failed_messages(monkeypatch):
    data = _make_big_data(n_articles=40, text_len=400)

    def boom(idx):
        raise RuntimeError("boom")

    # Tous les appels échouent → message générique (et pas un faux refus)
    _install_chunk_llm(monkeypatch, boom)
    ans = analyzer_mod._llm_analysis_answer(
        data, "Décris en détail ce que traite ce document, article par article.")
    assert "pas pu interroger" in ans

    # Le modèle répond mais rien n'est ancré → refus hors périmètre
    _install_chunk_llm(
        monkeypatch, lambda idx: "Je ne sais pas.\n\n[[GROUNDED-IN]]\n[[END]]")
    ans = analyzer_mod._llm_analysis_answer(
        data, "Décris en détail ce que traite ce document, article par article.")
    assert "Je ne peux répondre qu'à partir des documents chargés" in ans


def _make_status_error(status: int):
    import httpx
    from groq import APIStatusError

    resp = httpx.Response(status, request=httpx.Request("POST", "http://test"))
    return APIStatusError("err", response=resp, body=None)


class _StatusErrorLLM:
    """Faux LLMClient qui lève une APIStatusError du code fourni."""

    def __init__(self, status: int):
        self.status = status

    def generate(self, system_instruction, user_prompt):
        raise _make_status_error(self.status)

    def generate_with_citation_guarantee(self, system_instruction, user_prompt):
        raise _make_status_error(self.status)


def test_llm_analysis_answer_typed_413_message(monkeypatch):
    """Le message « document trop long » est déclenché par le CODE HTTP 413
    (APIStatusError typé), plus par un appariement de chaîne sur str(e)."""
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")

    # Document assez petit pour rester en une seule requête
    monkeypatch.setattr(llm_mod, "LLMClient",
                        lambda *a, **k: _StatusErrorLLM(413))
    ans = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, "Décris ce document")
    assert "trop long pour une analyse" in ans


def test_llm_analysis_answer_other_status_generic_message(monkeypatch):
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")

    monkeypatch.setattr(llm_mod, "LLMClient",
                        lambda *a, **k: _StatusErrorLLM(400))
    ans = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, "Décris ce document")
    assert "pas pu interroger" in ans


def _make_rate_limit_error():
    import httpx
    from groq import RateLimitError

    resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
    return RateLimitError("rate limit", response=resp, body=None)


def test_llm_analysis_answer_rate_limit_message(monkeypatch):
    """RateLimitError produit le message « Quota quotidien », distinct du
    générique réseau — la cause est nommée au lieu d'être avalée."""
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")

    class RateLimitedLLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, system_instruction, user_prompt):
            raise _make_rate_limit_error()

    monkeypatch.setattr(llm_mod, "LLMClient",
                        lambda *a, **k: RateLimitedLLM())
    ans = analyzer_mod._llm_analysis_answer_uncached(
        SAMPLE_JSON, "résumé du document")
    assert "Quota quotidien" in ans
    assert "réseau" not in ans


def test_llm_analysis_answer_cache_hit(monkeypatch):
    """Même question sur le même document : un seul appel LLM, le second
    appel vient du cache."""
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")
    calls = {"n": 0}

    class CountingLLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, system_instruction, user_prompt):
            calls["n"] += 1
            return "Réponse test.\n\n[[GROUNDED-IN]]\nSource 1\n[[END]]"

    monkeypatch.setattr(llm_mod, "LLMClient", lambda *a, **k: CountingLLM())

    q = "Décris ce document"
    a1 = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, q)
    a2 = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, q)

    assert a1 == a2
    assert calls["n"] == 1


def test_llm_analysis_answer_errors_not_cached(monkeypatch):
    """Une réponse d'erreur n'est jamais mise en cache : chaque appel
    retente réellement le LLM."""
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")
    calls = {"n": 0}

    class FlakyLLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, system_instruction, user_prompt):
            calls["n"] += 1
            raise ConnectionError("panne réseau simulée")

    monkeypatch.setattr(llm_mod, "LLMClient", lambda *a, **k: FlakyLLM())

    q = "Décris ce document"
    analyzer_mod._llm_analysis_answer(SAMPLE_JSON, q)
    analyzer_mod._llm_analysis_answer(SAMPLE_JSON, q)

    assert calls["n"] == 2


def test_chunked_all_failed_rate_limit_message(monkeypatch):
    """Vue d'ensemble fragmentée, tous les paquets en 429 : le quota est
    nommé (et pas le générique) — c'était invisible avant Fix 1."""
    data = _make_big_data(n_articles=40, text_len=400)

    def rate_limited(idx):
        raise _make_rate_limit_error()

    _install_chunk_llm(monkeypatch, rate_limited)
    ans = analyzer_mod._llm_analysis_answer(
        data, "Décris en détail ce que traite ce document, article par article.")
    assert "Quota quotidien" in ans


def test_plan_chunks_align_on_instrument_boundary(monkeypatch):
    """TÂCHE 1.4 : sans la correction, un paquet mélangera la fin du décret A
    et le début du décret B ; avec, chaque paquet reste mono-référence."""
    monkeypatch.setattr(analyzer_mod, "ANALYSIS_CHUNK_CONTEXT_CHARS", 1000)
    srcs = [{"reference": ref, "instrument_type": "DECRET", "text": "a" * 300}
            for ref in ("A", "A", "A", "B", "B", "B")]  # coût 500/article

    chunks, covered = analyzer_mod._plan_analysis_chunks(srcs)

    assert covered == 6
    assert len(chunks) >= 3                       # la frontière a bien coupé
    for chunk in chunks:
        refs = {s["reference"] for s in chunk}
        assert len(refs) == 1                     # aucun mélange A/B


def test_chunked_calls_run_in_parallel(monkeypatch):
    """TÂCHE 1.2 : 6 paquets × 0,3 s sous 3 workers ≈ 0,6 s, pas 1,8 s ;
    toutes les réponses sont réassemblées dans l'ordre des paquets."""
    import time

    # Force le chemin fragmenté : contexte total > ANALYSIS_MAX_CONTEXT_CHARS
    monkeypatch.setattr(analyzer_mod, "ANALYSIS_MAX_CONTEXT_CHARS", 500)
    monkeypatch.setattr(analyzer_mod, "ANALYSIS_CHUNK_CONTEXT_CHARS", 1000)
    data = _make_big_data(n_articles=12, text_len=250)   # coût ≈ 470 → 2/paquet

    calls = {"n": 0}

    def behave(idx):
        calls["n"] += 1
        time.sleep(0.3)
        return ("Section.\n\n[[GROUNDED-IN]]\nSource 1\n[[END]]")

    _install_chunk_llm(monkeypatch, behave)

    srcs = analyzer_mod._articles_as_rag_sources(data)
    chunks, covered = analyzer_mod._plan_analysis_chunks(srcs)
    assert len(chunks) == 6 and covered == 12

    t0 = time.monotonic()
    ans = analyzer_mod._llm_analysis_answer(
        data, "Décris en détail ce que traite ce document, article par article.")
    elapsed = time.monotonic() - t0

    assert calls["n"] == 6
    assert elapsed < 1.4          # séquentiel serait ≥ 1,8 s
    # ordre du document conservé : les titres de sections apparaissent dans
    # l'ordre DECRET (paquets 1-3) puis ARRETE (paquets 4-6)
    titles = [analyzer_mod._analysis_section_title(c) for c in chunks]
    pos = [ans.find(t) for t in titles]
    assert pos == sorted(pos)


def test_deep_analysis_budget_blocks_second_launch(monkeypatch):
    """TÂCHE 1.3 : budget=1 lancement/jour — le second appel renvoie le
    message dédié SANS rappeler le LLM (le cache est purgé entre les deux
    pour prouver que c'est le garde-fou qui bloque, pas lui)."""
    monkeypatch.setattr(analyzer_mod, "_DEEP_ANALYSIS_DAILY_BUDGET", 1)
    data = _make_big_data(n_articles=40, text_len=400)

    def behave(idx):
        return ("Section.\n\n[[GROUNDED-IN]]\nSource 1\n[[END]]")

    prompts = _install_chunk_llm(monkeypatch, behave)

    q = "Décris en détail ce que traite ce document, article par article."
    ans1 = analyzer_mod._llm_analysis_answer(data, q)
    n_after_first = len(prompts)
    assert n_after_first > 0 and "Budget quotidien" not in ans1

    analyzer_mod._analysis_cache.clear()
    ans2 = analyzer_mod._llm_analysis_answer(data, q)
    assert "Budget quotidien d'analyses en profondeur atteint" in ans2
    assert len(prompts) == n_after_first           # aucun appel LLM de plus


def test_coverage_meta_in_chat_response(annotated_dir, monkeypatch):
    """Tâche 1.1 : /chat expose covered_articles/total_articles sur le chemin
    fragmenté partiel ; None ailleurs (question factuelle)."""
    _write_sample(annotated_dir)
    monkeypatch.setattr(analyzer_mod, "ANALYSIS_MAX_CHUNKS", 1)  # couverture partielle forcée
    data = _make_big_data(n_articles=40, text_len=400)
    p = annotated_dir / "BO_9999_Fr_entities.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    _install_chunk_llm(
        monkeypatch,
        lambda idx: ("Tranche.\n\n[[GROUNDED-IN]]\nSource 1\n[[END]]"))

    client = app.test_client()
    client.get("/open-analysis/BO_9999_Fr")
    r = client.post("/chat", json={
        "question": "Décris en détail ce que traite ce document, article par article.",
        "doc_id": "BO_9999_Fr",
    })
    body = r.get_json()
    total = len(analyzer_mod._articles_as_rag_sources(data))
    assert body["covered_articles"] is not None
    assert 0 < body["covered_articles"] < body["total_articles"] == total

    # Chemin non fragmenté : méta absente (None)
    r2 = client.post("/chat", json={"question": "Combien d'articles ?", "doc_id": "BO_9999_Fr"})
    assert r2.get_json()["covered_articles"] is None
    assert r2.get_json()["total_articles"] is None