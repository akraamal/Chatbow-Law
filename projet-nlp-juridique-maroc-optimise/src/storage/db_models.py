"""
src/storage/db_models.py
Schéma SQLite. Choix : SQLite plutôt que Postgres/Elasticsearch pour cette
taille de corpus — zéro serveur à gérer, requêtes SQL structurées possibles,
fichier unique versionnable.

Tables :
  documents        1 ligne par Bulletin Officiel
  articles         1 ligne par article, rattaché à un document
  entities         toutes les entités (LOI/DECRET/MINISTERE/PERSON/ORGANIZATION...)
                   rattachées à un article, avec position caractère
  citations        citations d'articles résolues (étape 4a-bis), rattachées
                   à un article
  entity_index     entités canoniques dédupliquées au niveau document
                   (sortie de document_consolidator.py), avec leurs variantes
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id            TEXT PRIMARY KEY,
    lang              TEXT NOT NULL,
    bo_number         TEXT,
    date_publication  TEXT,
    edition_label     TEXT,
    num_articles      INTEGER,
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    number      TEXT NOT NULL,
    raw_text    TEXT
    -- Pas de UNIQUE(doc_id, number) : un même Bulletin Officiel regroupe
    -- souvent plusieurs textes juridiques distincts (dahir, décret,
    -- arrêté...) publiés à la suite, chacun recommançant sa propre
    -- numérotation d'articles à 1 — "article 2" peut donc légitimement
    -- apparaître plusieurs fois dans le même doc_id. `id` (autoincrement)
    -- reste l'identifiant unique de la ligne.
);

CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id  INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    label       TEXT NOT NULL,     -- LOI, DECRET, MINISTERE, INSTITUTION, PERSON, ORGANIZATION, DATE...
    start_pos   INTEGER,
    end_pos     INTEGER
);

CREATE TABLE IF NOT EXISTS citations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id    INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    target_label  TEXT,
    target_text   TEXT,
    resolved      INTEGER NOT NULL DEFAULT 0   -- 0/1 (SQLite n'a pas de type booléen natif)
);

CREATE TABLE IF NOT EXISTS entity_index (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    category    TEXT NOT NULL,       -- "persons" | "organizations" | "legal_texts"
    canonical   TEXT NOT NULL,
    variants    TEXT NOT NULL,       -- JSON-encodé (liste de strings)
    articles    TEXT NOT NULL,       -- JSON-encodé (liste de numéros d'article)
    merged_from TEXT NOT NULL        -- JSON-encodé (liste de strings)
);

CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label);
CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(text);
CREATE INDEX IF NOT EXISTS idx_citations_target ON citations(target_text);
CREATE INDEX IF NOT EXISTS idx_entity_index_canonical ON entity_index(canonical);
"""