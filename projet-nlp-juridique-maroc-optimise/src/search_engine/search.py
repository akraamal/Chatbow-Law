"""
src/search_engine/search.py
-------------------------------
Étape 6 : recherche sémantique — charge l'index FAISS et les métadonnées
produits par index_builder.build_index(), encode une requête et renvoie
les articles les plus proches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.search_engine.embedder import Embedder
from src.search_engine.index_builder import DEFAULT_INDEX_DIR


class SemanticSearchEngine:
    """
    Usage :
        engine = SemanticSearchEngine()  # charge data/index/ par défaut
        results = engine.search("licence de télécommunications", top_k=5)
    """

    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR, model_name: str | None = None):
        import faiss

        index_path = Path(index_dir)
        faiss_file = index_path / "faiss.index"
        metadata_file = index_path / "metadata.json"

        if not faiss_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"Index introuvable dans {index_dir}/ — lance d'abord "
                f"scripts/build_search_index.py pour le construire."
            )

        self.index = faiss.read_index(str(faiss_file))
        self.metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

        # Réutilise le modèle avec lequel l'index a été construit, sauf
        # override explicite — encoder une requête avec un modèle différent
        # de celui utilisé pour les passages produirait des vecteurs dans
        # un espace différent, rendant la similarité dénuée de sens.
        saved_model_name_file = index_path / "model_name.txt"
        if model_name is None and saved_model_name_file.exists():
            model_name = saved_model_name_file.read_text(encoding="utf-8").strip()

        self.embedder = Embedder(model_name) if model_name else Embedder()

    def search(self, query: str, top_k: int = 5, lang: str | None = None) -> list[dict]:
        """
        Renvoie les top_k articles les plus proches sémantiquement de
        `query`, chacun avec son score de similarité cosinus (1.0 = identique).

        lang : si fourni ("fr" ou "ar"), ne renvoie que les articles de
        cette langue — utile car une requête en français peut aussi
        remonter des articles arabes sémantiquement proches (le modèle
        est multilingue par conception), ce qui n'est pas toujours voulu.
        """
        query_vector = self.embedder.embed_query(query).reshape(1, -1).astype("float32")

        # Sur-échantillonne quand un filtre de langue est actif : on ne sait
        # pas à l'avance combien de résultats bruts passeront le filtre.
        raw_k = top_k * 5 if lang else top_k
        raw_k = min(raw_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, raw_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if lang and meta["lang"] != lang:
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= top_k:
                break

        return results
