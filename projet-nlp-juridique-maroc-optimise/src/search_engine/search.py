"""
src/search_engine/search.py
-------------------------------
Étape 6 : recherche — charge l'index FAISS et les métadonnées produits par
index_builder.build_index(), encode une requête et renvoie les articles
les plus proches.

Depuis l'évaluation Phase 1, la recherche est hybride par défaut : elle
fusionne (RRF, Reciprocal Rank Fusion) le top-k FAISS (dense, multilingue)
avec un top-k BM25 lexical (rank_bm25) sur le texte brut. Le BM25 attrape
les identifiants précis — numéros de loi, références de décret, numéros
d'article ("loi n° 18-97", "2.23.919", "787-14") — que l'embedding dense
dilue souvent ; le dense, lui, couvre la paraphrase sémantique que le
lexical rate. Passer hybrid=False restaure le comportement dense seul.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
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

        self._bm25 = None  # BM25Okapi construit paresseusement au 1er appel

    # ------------------------------------------------------------------
    # Recherche hybride
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Découpage en mots simples (unicode : couvre français accentué
        et arabe). Pas de stemmer : l'OCR du corpus est bruité, un
        matching lexical exact reste le plus fiable pour les
        identifiants de références."""
        return re.findall(r"\w+", (text or "").lower())

    def _ensure_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            corpus = [
                self._tokenize(m.get("text_clean") or m.get("text"))
                for m in self.metadata
            ]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    def _faiss_ranked(
        self, query_vector, k: int, lang: str | None
    ) -> list[tuple[int, float]]:
        """Top-k positions de l'index FAISS (filtré par langue si demandé),
        avec leur score cosinus."""
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)
        ranked = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if lang and self.metadata[idx]["lang"] != lang:
                continue
            ranked.append((int(idx), float(score)))
        return ranked

    def _bm25_ranked(self, query: str, k: int, lang: str | None) -> list[int]:
        """Top-k positions du classement BM25 (filtré par langue si demandé)."""
        bm25 = self._ensure_bm25()
        scores = bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for idx in ranked:
            if lang and self.metadata[idx]["lang"] != lang:
                continue
            out.append(idx)
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
        """Fusionne des listes de positions par Reciprocal Rank Fusion.
        score(pos) = sum over lists of 1/(k + rank). k=60 est le réglage
        classique de la littérature RRF."""
        fused: dict[int, float] = defaultdict(float)
        for lst in ranked_lists:
            for rank, idx in enumerate(lst, start=1):
                fused[idx] += 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    def search(
        self,
        query: str,
        top_k: int = 5,
        lang: str | None = None,
        hybrid: bool = True,
        faiss_k: int = 30,
        bm25_k: int = 30,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Renvoie les top_k articles les plus proches de `query`, chacun
        avec son score (similarité cosinus si hybrid=False, score RRF
        sinon).

        hybrid : fusionne FAISS + BM25 par RRF (défaut). hybrid=False
        restaure la recherche dense pure d'origine.

        lang : si fourni ("fr" ou "ar"), ne renvoie que les articles de
        cette langue — utile car une requête en français peut aussi
        remonter des articles arabes sémantiquement proches (le modèle
        est multilingue par conception), ce qui n'est pas toujours voulu.
        """
        query_vector = self.embedder.embed_query(query).reshape(1, -1).astype("float32")

        if not hybrid:
            # Comportement historique : dense seul. Sur-échantillonne
            # quand un filtre de langue est actif, comme à l'origine.
            raw_k = top_k * 5 if lang else top_k
            ranked = self._faiss_ranked(query_vector, raw_k, lang)
            return [
                {**self.metadata[idx], "score": score}
                for idx, score in ranked[:top_k]
            ]

        fused = self._rrf_fuse(
            [
                [idx for idx, _ in self._faiss_ranked(query_vector, faiss_k, lang)],
                self._bm25_ranked(query, bm25_k, lang),
            ],
            rrf_k,
        )
        return [
            {**self.metadata[idx], "score": round(score, 6)}
            for idx, score in fused[:top_k]
        ]
