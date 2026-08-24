"""
app/chat_history_store.py
-------------------------
Persistance SQLite des conversations du chatbot principal (portail « / »).

Contrairement à l'historique de l'analyseur v2 (en mémoire, scopé par
document, TTL 24 h — un cache de questions de suivi), cet historique est
généraliste et doit survivre aux redémarrages : c'est une vraie liste de
conversations réouvrables, d'où SQLite plutôt qu'un dict en mémoire.

Fichier : data/chat_history.sqlite3 (même racine data/ que le reste).
Accès sérialisé par un lock process-wide ; connexion unique réutilisée.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.sqlite3"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )""")
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                ts              REAL NOT NULL,
                sources_json    TEXT
            )""")
        _conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conv
            ON messages(conversation_id, id)""")
        _conn.commit()
    return _conn


def create_conversation(title: str = "") -> str:
    conv_id = uuid.uuid4().hex[:16]
    now = time.time()
    with _lock:
        _db().execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)", (conv_id, title, now, now))
        _db().commit()
    return conv_id


def conversation_exists(conv_id: str) -> bool:
    with _lock:
        row = _db().execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return row is not None


def set_title(conv_id: str, title: str) -> None:
    """Titre = première question utilisateur, tronquée — ne s'applique
    qu'une fois (les appels suivants sont sans effet sur un titre non vide)."""
    with _lock:
        _db().execute(
            "UPDATE conversations SET title = ? "
            "WHERE id = ? AND (title IS NULL OR title = '')",
            (title, conv_id))
        _db().commit()


def touch_conversation(conv_id: str) -> None:
    with _lock:
        _db().execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                      (time.time(), conv_id))
        _db().commit()


def append_message(conv_id: str, role: str, content: str,
                   sources: list | None = None) -> None:
    now = time.time()
    # default=str : tolère les scalaires numpy/objets des sources RAG.
    try:
        sources_json = json.dumps(sources or [], ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        sources_json = "[]"
    with _lock:
        _db().execute(
            "INSERT INTO messages (conversation_id, role, content, ts, sources_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, role, content, now, sources_json))
        _db().execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                      (now, conv_id))
        _db().commit()


def get_history(conv_id: str, limit: int = 200) -> list[dict]:
    """Messages du plus ancien au plus récent, plafonnés aux `limit`
    derniers. `sources` est désérialisé (liste vide si absent)."""
    with _lock:
        rows = _db().execute(
            "SELECT role, content, ts, sources_json FROM messages "
            "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conv_id, limit)).fetchall()
    out = []
    for r in reversed(rows):
        try:
            sources = json.loads(r["sources_json"]) if r["sources_json"] else []
        except (json.JSONDecodeError, TypeError):
            sources = []
        out.append({"role": r["role"], "content": r["content"],
                    "ts": r["ts"], "sources": sources})
    return out


def list_conversations(limit: int = 50) -> list[dict]:
    with _lock:
        rows = _db().execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r["id"], "title": r["title"],
             "created_at": r["created_at"], "updated_at": r["updated_at"]}
            for r in rows]


def delete_conversation(conv_id: str) -> bool:
    with _lock:
        cur = _db().execute("DELETE FROM conversations WHERE id = ?",
                            (conv_id,))
        _db().execute("DELETE FROM messages WHERE conversation_id = ?",
                      (conv_id,))
        _db().commit()
    return cur.rowcount > 0
