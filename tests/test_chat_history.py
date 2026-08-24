"""Tests du « Chat historique » du portail : store SQLite (app/
chat_history_store.py) et routes /api/chat/history|conversations.

Le bot RAG est simulé (get_chatbot monkeypatché) : aucun embedding ni
appel Groq ici — on teste la persistance, le câblage serveur de
l'historique de reformulation et les contrats de routes.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import chat_history_store as store  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """SQLite isolée par test (le module met en cache sa connexion)."""
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "chat_history.sqlite3")
    store._conn = None
    yield
    if store._conn is not None:
        store._conn.close()
        store._conn = None


@pytest.fixture()
def client():
    from app.main import app
    return app.test_client()


def test_store_roundtrip_and_sources_json():
    conv_id = store.create_conversation(title="Première question")
    store.append_message(conv_id, "user", "Que dit le décret X ?")
    store.append_message(conv_id, "assistant", "Il institue Y.",
                         sources=[{"doc_id": "D1", "text": "extrait"}])

    hist = store.get_history(conv_id)
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[0]["content"] == "Que dit le décret X ?"
    assert hist[1]["sources"] == [{"doc_id": "D1", "text": "extrait"}]  # round-trip JSON
    assert store.list_conversations()[0]["title"] == "Première question"


def test_store_title_set_once():
    conv_id = store.create_conversation()
    assert store.list_conversations()[0]["title"] == ""
    store.set_title(conv_id, "Premier sujet")
    store.set_title(conv_id, "Second sujet ignoré")   # ne s'applique qu'une fois
    assert store.list_conversations()[0]["title"] == "Premier sujet"


def test_store_delete_cascades_messages():
    conv_id = store.create_conversation("à supprimer")
    store.append_message(conv_id, "user", "q")
    assert store.delete_conversation(conv_id) is True
    assert store.get_history(conv_id) == []
    assert store.conversation_exists(conv_id) is False
    assert store.delete_conversation(conv_id) is False   # idempotent


def test_unknown_conversation_history_returns_empty_list(client):
    r = client.get("/api/chat/history/id_inconnu")
    assert r.status_code == 200
    assert r.get_json() == {"messages": []}


@pytest.fixture()
def fake_bot(monkeypatch):
    from app import chat as chat_mod

    class FakeBot:
        def __init__(self):
            self.calls = []

        def answer(self, query, history=None, lang=None):
            self.calls.append({"query": query, "history": list(history or [])})
            return {"answer": f"Réponse à : {query}",
                    "sources": [{"doc_id": "D1", "text": "x"}]}

    bot = FakeBot()
    monkeypatch.setattr(chat_mod, "get_chatbot", lambda: (bot, None))
    return bot


def test_post_creates_conversation_and_persists_sources(client, fake_bot):
    r = client.post("/api/chat", json={"query": "Que dit le décret 2-25-1 ?"})
    body = r.get_json()
    conv_id = body["conversation_id"]
    assert conv_id

    msgs = client.get(f"/api/chat/history/{conv_id}").get_json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"][0]["doc_id"] == "D1"

    convos = client.get("/api/chat/conversations").get_json()["conversations"]
    assert convos[0]["id"] == conv_id
    assert convos[0]["title"].startswith("Que dit le décret")


def test_second_turn_uses_server_side_history_for_reformulation(
        client, fake_bot):
    r1 = client.post("/api/chat", json={"query": "première question"}).get_json()

    # Le client ne renvoie PLUS d'historique : seul conversation_id voyage.
    client.post("/api/chat",
                json={"query": "question de suivi",
                      "conversation_id": r1["conversation_id"]})

    second_call = fake_bot.calls[1]
    assert second_call["history"] == [
        {"question": "première question", "answer": "Réponse à : première question"}
    ]


def test_delete_route_removes_conversation(client, fake_bot):
    r = client.post("/api/chat", json={"query": "à supprimer"}).get_json()
    conv_id = r["conversation_id"]

    dr = client.delete(f"/api/chat/conversations/{conv_id}")
    assert dr.get_json() == {"ok": True, "deleted": True}
    assert client.get(f"/api/chat/history/{conv_id}").get_json()["messages"] == []
