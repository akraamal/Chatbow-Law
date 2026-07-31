"""
src/storage/db_connector.py
Connecteur SQLite : écrit un document consolidé (sortie de
document_consolidator.consolidate_document()) et fournit des requêtes de
base pour la suite (classification, search_engine, chatbot).
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .db_models import SCHEMA_SQL

from ..extraction.entity_span_utils import get_start, get_end


class DBConnector:
    def __init__(self, db_path: str = "data/processed/juridique.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def save_document(self, consolidated_document: dict, overwrite: bool = True):
        """
        consolidated_document : sortie de
            document_consolidator.consolidate_document()
        overwrite : si True, supprime d'abord toute version existante du
                    même doc_id (utile en cas de ré-exécution du pipeline)
        """
        doc_id = consolidated_document["doc_id"]

        with self._connect() as conn:
            if overwrite:
                conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                # articles/entities/citations/entity_index suivent via ON DELETE CASCADE

            conn.execute(
                """INSERT INTO documents
                   (doc_id, lang, bo_number, date_publication, edition_label, num_articles)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    consolidated_document.get("lang"),
                    consolidated_document.get("bo_number"),
                    consolidated_document.get("date_publication"),
                    consolidated_document.get("edition_label"),
                    consolidated_document.get("num_articles", 0),
                ),
            )

            for article in consolidated_document.get("articles", []):
                cursor = conn.execute(
                    "INSERT INTO articles (doc_id, number, raw_text) VALUES (?, ?, ?)",
                    (doc_id, article.get("number", "?"), article.get("text")),
                )
                article_id = cursor.lastrowid

                for e in article.get("entities", []):
                    conn.execute(
                        """INSERT INTO entities (article_id, text, label, start_pos, end_pos)
                           VALUES (?, ?, ?, ?, ?)""",
                        (article_id, e.get("text", ""), e.get("label", ""), get_start(e), get_end(e)),
                    )
                # persons/organizations sont aussi des entités : même table,
                # ça simplifie les requêtes transverses par label plus tard
                for p in article.get("persons", []):
                    conn.execute(
                        """INSERT INTO entities (article_id, text, label, start_pos, end_pos)
                           VALUES (?, ?, ?, ?, ?)""",
                        (article_id, p.get("text", ""), p.get("label", ""), get_start(p), get_end(p)),
                    )
                for o in article.get("organizations", []):
                    conn.execute(
                        """INSERT INTO entities (article_id, text, label, start_pos, end_pos)
                           VALUES (?, ?, ?, ?, ?)""",
                        (article_id, o.get("text", ""), o.get("label", ""), get_start(o), get_end(o)),
                    )

                for c in article.get("citations", []):
                    conn.execute(
                        """INSERT INTO citations (article_id, text, target_label, target_text, resolved)
                           VALUES (?, ?, ?, ?, ?)""",
                        (article_id, c.get("text", ""), c.get("target_label"),
                         c.get("target_text"), int(bool(c.get("resolved", False)))),
                    )

            entities_index = consolidated_document.get("entities_index", {})
            for category, entries in entities_index.items():
                for entry in entries:
                    conn.execute(
                        """INSERT INTO entity_index
                           (doc_id, category, canonical, variants, articles, merged_from)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            doc_id, category, entry.get("canonical", ""),
                            json.dumps(entry.get("variants", []), ensure_ascii=False),
                            json.dumps(entry.get("articles", []), ensure_ascii=False),
                            json.dumps(entry.get("merged_from", []), ensure_ascii=False),
                        ),
                    )

    # ------------------------------------------------------------------
    # Lecture / requêtes
    # ------------------------------------------------------------------

    def get_document(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM documents ORDER BY date_publication").fetchall()
            return [dict(r) for r in rows]

    def find_entities_by_label(self, label: str) -> list[dict]:
        """Ex: find_entities_by_label('LOI') -> toutes les lois citées, tous documents confondus."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT e.text, e.label, a.number AS article_number, a.doc_id
                   FROM entities e
                   JOIN articles a ON a.id = e.article_id
                   WHERE e.label = ?""",
                (label,),
            ).fetchall()
            return [dict(r) for r in rows]

    def find_citations_to(self, target_text: str) -> list[dict]:
        """Toutes les citations qui pointent vers un texte donné (ex: 'loi n° 03-25')."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT c.text AS citation_text, c.target_label, a.number AS article_number, a.doc_id
                   FROM citations c
                   JOIN articles a ON a.id = c.article_id
                   WHERE c.target_text = ? AND c.resolved = 1""",
                (target_text,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_entity_index(self, doc_id: str, category: str | None = None) -> list[dict]:
        """Entités canoniques dédupliquées pour un document (persons/organizations/legal_texts)."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if category:
                rows = conn.execute(
                    "SELECT * FROM entity_index WHERE doc_id = ? AND category = ?",
                    (doc_id, category),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entity_index WHERE doc_id = ?", (doc_id,)
                ).fetchall()

            results = []
            for r in rows:
                entry = dict(r)
                entry["variants"] = json.loads(entry["variants"])
                entry["articles"] = json.loads(entry["articles"])
                entry["merged_from"] = json.loads(entry["merged_from"])
                results.append(entry)
            return results

    def list_articles(self, lang: str | None = None) -> list[dict]:
        """
        Renvoie tous les articles avec le texte et les métadonnées de leur
        document parent (langue, numéro de BO, date de publication) — utilisé
        par le moteur de recherche sémantique (étape 6) pour construire
        l'index FAISS.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT a.id AS article_id, a.doc_id, a.number AS article_number,
                       a.raw_text, d.lang, d.bo_number, d.date_publication
                FROM articles a
                JOIN documents d ON d.doc_id = a.doc_id
            """
            params = ()
            if lang:
                query += " WHERE d.lang = ?"
                params = (lang,)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]