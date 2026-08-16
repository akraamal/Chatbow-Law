"""
src/search_engine/index_builder.py
--------------------------------------
Étape 6 : construit un index FAISS à partir des articles de
data/processed/juridique.db ou d'une liste dict d'articles enrichis
(issue de run_rag_pipeline.py) et le persiste sur disque avec les
métadonnées associées.

Deux fichiers produits (répertoire configurable, défaut data/index/) :
  faiss.index       l'index FAISS lui-même (vecteurs)
  metadata.json     un objet JSON par vecteur, dans le même ordre que
                    l'index — doc_id, article_number, texte, langue,
                    date de publication — pour retrouver l'article
                    correspondant à un résultat de recherche.

Choix d'index : IndexFlatIP (produit scalaire exact, pas d'approximation)
sur des embeddings normalisés (= similarité cosinus). Corpus actuel de
quelques centaines à quelques milliers d'articles : la recherche exacte
reste largement assez rapide, pas besoin d'un index approximatif
(IVF/HNSW) qui ajouterait de la complexité pour un gain de vitesse
inutile à cette échelle. À revisiter si le corpus dépasse ~100 000
articles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.search_engine.embedder import Embedder

DEFAULT_INDEX_DIR = "data/index"
DEFAULT_MIN_CHARS = 20  # articles plus courts que ça (boilerplate pur) n'apportent rien à la recherche


def _load_articles_from_db(db_path: str, min_chars: int) -> list[dict]:
    from src.storage.db_connector import DBConnector
    db = DBConnector(db_path)
    articles = db.list_articles()
    return [a for a in articles if len(a["raw_text"] or "") >= min_chars]


def _normalize_article(art: dict) -> dict:
    """Normalize either a DB article or an enriched JSON article to a
    consistent metadata dict for FAISS indexing."""
    # Determine the text field: DB uses "raw_text", enriched JSON uses "text"
    text = art.get("raw_text") or art.get("text", "")
    return {
        "article_id": art.get("article_id", art.get("number", "")),
        "doc_id": art.get("doc_id", ""),
        "article_number": art.get("article_number", art.get("number", "")),
        "lang": art.get("lang", "fr"),
        "bo_number": art.get("bo_number", ""),
        "date_publication": art.get("date_publication", ""),
        "text": text,
        # text_clean = texte + tableaux linéarisés (produit par
        # enrich_json_with_pages --tables). Conservé dans l'index : le LLM
        # voit et cite le contenu des tableaux à partir de lui.
        "text_clean": art.get("text_clean", ""),
        # Enriched fields (may be empty; instrument_type can be None for
        # content with no legal-instrument keyword, e.g. CESE annexes)
        "instrument_type": art.get("instrument_type") or "",
        "reference": art.get("reference", ""),
        "pdf_page": art.get("pdf_page"),
        "printed_page": art.get("printed_page"),
        "extracted_tables": art.get("extracted_tables", []),
    }


def build_index(
    articles: list[dict] | str = "data/processed/juridique.db",
    index_dir: str = DEFAULT_INDEX_DIR,
    model_name: str | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    batch_size: int = 32,
) -> dict:
    """
    Construit l'index FAISS et le sauvegarde dans index_dir.

    *articles* peut être :
      - une chaîne (chemin vers la base SQLite) — charge depuis DB
        (comportement historique)
      - une liste de dicts — indexe directement les articles enrichis
        (utilisé par run_rag_pipeline.py --build-index)

    Renvoie un résumé (nombre d'articles indexés, dimension, chemins).
    """
    if isinstance(articles, str):
        raw = _load_articles_from_db(articles, min_chars)
    else:
        raw = [a for a in articles if len((a.get("text") or a.get("raw_text") or "")) >= min_chars]

    if not raw:
        raise ValueError(
            f"Aucun article à indexer (ou tous plus courts que "
            f"{min_chars} caractères)."
        )

    embedder = Embedder(model_name) if model_name else Embedder()
    texts = []
    for a in raw:
        # Embedder text_clean (texte + tableaux linéarisés) quand il existe :
        # sinon les valeurs des tableaux (553,00 DH/TM, charges > 5 kg, ...)
        # sont invisibles pour le retrieval sémantique.
        t = a.get("text_clean") or a.get("raw_text") or a.get("text", "")
        texts.append(t)
    vectors = embedder.embed_passages(texts, batch_size=batch_size)

    import faiss
    index = faiss.IndexFlatIP(embedder.dimension)
    index.add(np.asarray(vectors, dtype="float32"))

    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "faiss.index"))

    metadata = [_normalize_article(a) for a in raw]
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "model_name.txt").write_text(embedder.model_name, encoding="utf-8")

    return {
        "n_articles": len(raw),
        "dimension": embedder.dimension,
        "model_name": embedder.model_name,
        "index_path": str(out_dir / "faiss.index"),
        "metadata_path": str(out_dir / "metadata.json"),
    }
