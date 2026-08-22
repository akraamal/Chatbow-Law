"""Tests de la mémoire conversationnelle du chat analyseur v2 (TÂCHE 2).

Couvre : trims d'historique, injection dans le repli LLM factuel UNIQUEMENT,
bypass du cache quand une conversation est active, écriture via /chat et
lecture via /chat/history/<doc_id>.
"""

import json

import pytest

from adli_v2 import pipeline
from adli_v2.app import analyzer as analyzer_mod
from adli_v2.app.main import app

SAMPLE_JSON = {
    "doc_id": "BO_8888_Fr",
    "lang": "fr",
    "bo_number": "8888",
    "date_publication": "2026-08-01",
    "articles": [
        {"number": "1", "text": "Il est institué un DECRET n° 2-26-100 test.",
         "pdf_page": 3},
        {"number": "2", "text": "Les modalités sont fixées par arrêté.",
         "pdf_page": 3},
    ],
    "instruments": [
        {"instrument_type": "DECRET", "reference": "2-26-100",
         "n_articles": 2, "article_indices": [0, 1]},
    ],
}


@pytest.fixture(autouse=True)
def _clear_chat_state():
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


def _install_prompt_capturing_llm(monkeypatch):
    """Remplace LLMClient par un faux qui enregistre les prompts user."""
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")
    prompts: list[str] = []

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, system_instruction, user_prompt):
            prompts.append(user_prompt)
            return ("Réponse.\n\n[[GROUNDED-IN]]\nSource 1\n[[END]]")

        def generate_with_citation_guarantee(self, system_instruction, user_prompt):
            prompts.append(user_prompt)
            return "Réponse factuelle [[citation 1]]"

    monkeypatch.setattr(llm_mod, "LLMClient", lambda *a, **k: FakeLLM())
    return prompts


def test_append_history_trims_to_max_turns():
    for i in range(8):
        analyzer_mod._append_history("D1", "user", f"q{i}")
        analyzer_mod._append_history("D1", "assistant", f"r{i}")

    hist = analyzer_mod._chat_history["D1"]
    max_msgs = analyzer_mod._MAX_HISTORY_TURNS * 2
    assert len(hist) <= max_msgs
    assert hist[-1]["content"] == "r7"          # les plus récents survivent
    assert hist[0]["content"] != "q0"           # les plus anciens sont purgés


def test_history_block_injected_for_factual_llm(monkeypatch):
    """Question de suivi : le prompt LLM embarque les échanges précédents."""
    analyzer_mod._append_history("T1", "user", "Que dit le décret 2-25-1080 ?")
    analyzer_mod._append_history("T1", "assistant", "Il institue une commission.")
    prompts = _install_prompt_capturing_llm(monkeypatch)

    analyzer_mod._llm_analysis_answer_uncached(
        SAMPLE_JSON, "et son article 3 ?", doc_id="T1")

    assert prompts, "le repli LLM doit être appelé"
    assert "Échanges précédents sur ce document" in prompts[-1]
    assert "décret 2-25-1080" in prompts[-1]
    assert "et son article 3 ?" in prompts[-1]


def test_no_history_in_overview_prompt(monkeypatch):
    """L'analyse d'ensemble reste auto-suffisante : pas d'historique injecté,
    pour préserver le budget de contexte réservé aux articles."""
    analyzer_mod._append_history("T1", "user", "ancienne question")
    analyzer_mod._append_history("T1", "assistant", "ancienne réponse")
    prompts = _install_prompt_capturing_llm(monkeypatch)

    ans = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, "résumé du document")

    assert isinstance(ans, str) and ans
    for p in prompts:
        assert "Échanges précédents" not in p


def test_cache_bypassed_when_history_present():
    key = analyzer_mod._cache_key(SAMPLE_JSON, "Décris ce document")
    analyzer_mod._analysis_cache[key] = ("CACHÉ", None, None)

    # Sans historique : l'entrée en cache est servie telle quelle.
    ans = analyzer_mod._llm_analysis_answer(SAMPLE_JSON, "Décris ce document")
    assert ans == "CACHÉ"

    # Avec historique : la même question doit repasser par le LLM.
    analyzer_mod._append_history("BO_8888_Fr", "user", "autre sujet")
    prompts = None  # pas de LLM réel : on vérifie juste que ce n'est pas CACHÉ
    import importlib
    llm_mod = importlib.import_module("src.rag.llm_client")

    class FreshLLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, si, up):
            return "Frais.\n\n[[GROUNDED-IN]]\nSource 1\n[[END]]"

        def generate_with_citation_guarantee(self, si, up):
            return "Réponse fraîche [[citation 1]]"

    llm_mod.LLMClient = lambda *a, **k: FreshLLM()
    ans2 = analyzer_mod._llm_analysis_answer(
        SAMPLE_JSON, "Décris ce document", doc_id="BO_8888_Fr")
    assert ans2 != "CACHÉ"


def test_route_appends_and_serves_history(annotated_dir, monkeypatch):
    p = annotated_dir / "BO_8888_Fr_entities.json"
    p.write_text(json.dumps(SAMPLE_JSON, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        analyzer_mod, "_llm_analysis_answer",
        lambda data, question, doc_id="": f"réponse à « {question} »")
    client = app.test_client()
    client.get("/open-analysis/BO_8888_Fr")

    client.post("/chat", json={
        "question": "comment s'applique le régime fiscal ?",
        "doc_id": "BO_8888_Fr"})
    client.post("/chat", json={
        "question": "quel est l'esprit général du texte ?",
        "doc_id": "BO_8888_Fr"})

    r = client.get("/chat/history/BO_8888_Fr")
    hist = r.get_json()["history"]
    assert [h["role"] for h in hist] == ["user", "assistant", "user", "assistant"]
    assert "régime fiscal" in hist[0]["content"]
    assert "réponse à" in hist[1]["content"]
    # question vide : ni réponse ni pollution d'historique
    r2 = client.post("/chat", json={"question": "", "doc_id": "BO_8888_Fr"})
    assert "Veuillez poser une question" in r2.get_json()["answer"]
    hist2 = client.get("/chat/history/BO_8888_Fr").get_json()["history"]
    assert len(hist2) == len(hist)


def test_chat_history_endpoint_unknown_doc():
    client = app.test_client()
    r = client.get("/chat/history/document_inconnu")
    assert r.status_code == 200
    assert r.get_json() == {"history": []}
