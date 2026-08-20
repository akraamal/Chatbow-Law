"""
tests/test_analyzer_interrupt_history.py
----------------------------------------
Tests de l'interruption d'une analyse, du registre persistant des
analyses précédentes et de la réouverture d'une ancienne analyse
(app/analyzer.py : _cancel_task, _record_analysis, routes /cancel,
/analyses, /open-analysis) — sans lancer de vrai pipeline.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import app.analyzer as analyzer


# ── Interruption ──────────────────────────────────────────────────────

class _FakeProc:
    """Sous-processus simulé : poll() non-None uniquement après terminate()."""
    def __init__(self):
        self._terminated = False
        self._poll_count = 0

    def poll(self):
        return -15 if self._terminated else None

    def terminate(self):
        self._terminated = True


def test_cancel_task_sets_flag_and_terminates_proc():
    tid = analyzer._new_task(filename="BO_test.pdf")
    fake_proc = _FakeProc()
    with analyzer._tasks_lock:
        analyzer._tasks[tid]["proc"] = fake_proc

    assert analyzer._cancel_task(tid) is True
    assert analyzer._task_cancelled(tid) is True
    assert fake_proc._terminated is True

    # Interruption d'une tâche déjà terminée → refusée
    with analyzer._tasks_lock:
        analyzer._tasks[tid]["done"] = True
    assert analyzer._cancel_task(tid) is False

    with analyzer._tasks_lock:
        analyzer._tasks.pop(tid, None)


def test_cancel_unknown_task_returns_false():
    assert analyzer._cancel_task("doesnotexist") is False


def test_cancel_route_returns_json():
    app = _make_app()
    tid = analyzer._new_task(filename="BO_test.pdf")
    with app.test_client() as c:
        r = c.post(f"/cancel/{tid}")
        assert r.status_code == 200
        assert r.get_json()["ok"] is True
        # Tâche inconnue → 404
        r2 = c.post("/cancel/nope")
        assert r2.status_code == 404
    with analyzer._tasks_lock:
        analyzer._tasks.pop(tid, None)


# ── Historique persistant ─────────────────────────────────────────────

@pytest.fixture()
def _isolated_history(tmp_path, monkeypatch):
    """Redirige le registre et le dossier annoté vers des chemins temporaires."""
    monkeypatch.setattr(analyzer, "HISTORY_FILE", tmp_path / "analyses_history.json")
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    monkeypatch.setattr(analyzer, "ANNOTATED_DIR", annotated)
    yield tmp_path / "analyses_history.json"


def _make_result_file(tmp_path: Path, doc_id: str) -> Path:
    """Crée un fichier de résultat annoté factice."""
    data = {
        "doc_id": doc_id,
        "bo_number": "7510",
        "date_publication": "2026-05-21",
        "filename": f"{doc_id}.pdf",
        "articles": [{"number": "1", "text": "Article 1 test."}],
        "instruments": [{"instrument_type": "Loi", "reference": "1-93-153",
                         "n_articles": 1, "article_indices": [0]}],
        "preamble_text": "",
        "preamble_entities": [],
        "entity_counts": [{"label": "LOI", "count": 1, "color": "#e74c3c"}],
    }
    path = tmp_path / f"{doc_id}_entities.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_record_analysis_and_load(_isolated_history, tmp_path):
    result_path = _make_result_file(tmp_path, "abc123")
    analyzer._record_analysis("task1", result_path)

    history = analyzer._load_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["doc_id"] == "abc123"
    assert entry["bo_number"] == "7510"
    assert entry["n_instruments"] == 1
    assert entry["n_articles"] == 1
    assert entry["result_path"] == str(result_path)


def test_record_analysis_deduplicates_by_doc_id(_isolated_history, tmp_path):
    """Réanalyser le même document remplace l'entrée (pas de doublon)."""
    result_path = _make_result_file(tmp_path, "abc123")
    analyzer._record_analysis("task1", result_path)
    analyzer._record_analysis("task2", result_path)

    history = analyzer._load_history()
    assert len(history) == 1
    assert history[0]["task_id"] == "task2"


def test_record_analysis_missing_file_is_silent(_isolated_history, tmp_path):
    analyzer._record_analysis("task1", tmp_path / "absent.json")
    assert analyzer._load_history() == []


def test_analyses_route_lists_visible_only(_isolated_history, tmp_path):
    good = _make_result_file(tmp_path, "good123")
    analyzer._record_analysis("task_good", good)
    # Entrée dont le fichier résultat a disparu → masquée
    with analyzer._history_lock:
        analyzer._save_history(analyzer._load_history() + [{
            "doc_id": "ghost999", "task_id": "t2", "filename": "",
            "bo_number": "", "date_publication": "", "n_instruments": 0,
            "n_articles": 0, "result_path": str(tmp_path / "ghost_entities.json"),
            "created_at": time.time(),
        }])

    app = _make_app()
    with app.test_client() as c:
        r = c.get("/analyses")
        assert r.status_code == 200
        docs = [a["doc_id"] for a in r.get_json()["analyses"]]
    assert "good123" in docs
    assert "ghost999" not in docs


def test_open_analysis_reloads_context_and_chat(_isolated_history, tmp_path):
    result_path = _make_result_file(tmp_path, "abc123")
    analyzer._record_analysis("task1", result_path)

    app = _make_app()
    with app.test_client() as c:
        r = c.get("/open-analysis/abc123")
        assert r.status_code == 200
        data = r.get_json()
        assert data["bo_number"] == "7510"
        assert data["n_articles"] == 1

        # Le contexte chat est rechargé → le chat répond sur ce document
        r2 = c.post("/chat", json={"question": "combien d'articles ?", "doc_id": "abc123"})
        assert r2.status_code == 200
        assert "1 article" in r2.get_json()["answer"]

    with analyzer._chat_lock:
        analyzer._chat_contexts.pop("abc123", None)


def test_open_analysis_unknown_returns_404(_isolated_history, tmp_path):
    app = _make_app()
    with app.test_client() as c:
        assert c.get("/open-analysis/inconnu").status_code == 404


def test_history_reconciles_after_project_move(_isolated_history, tmp_path):
    """Un-nesting (déménagement du dépôt) : les chemins absolus du registre
    pointent dans le vide. L'historique doit se reconstruire à partir des
    JSON annotés présents sur disque, et l'ouverture doit fonctionner."""
    orphan = _make_result_file(analyzer.ANNOTATED_DIR, "orphan123")
    with analyzer._history_lock:
        analyzer._save_history([{
            "doc_id": "ghost999", "task_id": "t1", "filename": "",
            "bo_number": "", "date_publication": "", "n_instruments": 0,
            "n_articles": 0,
            "result_path": str(tmp_path / "absent" / "ghost_entities.json"),
            "created_at": time.time(),
        }])

    app = _make_app()
    with app.test_client() as c:
        r = c.get("/analyses")
        assert r.status_code == 200
        docs = [a["doc_id"] for a in r.get_json()["analyses"]]
        assert "orphan123" in docs
        assert "ghost999" not in docs

        r2 = c.get("/open-analysis/orphan123")
        assert r2.status_code == 200
        assert r2.get_json()["bo_number"] == "7510"
        assert r2.get_json()["n_articles"] == 1


# ── Helpers ───────────────────────────────────────────────────────────

def _make_app():
    from app.main import app
    app.config["TESTING"] = True
    return app
