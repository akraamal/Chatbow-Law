"""
src/search_engine/embedder.py
--------------------------------
Étape 6 : encodage des textes en vecteurs, pour l'indexation FAISS et la
recherche sémantique.

Modèle par défaut : intfloat/multilingual-e5-base — choisi plutôt qu'un
modèle "paraphrase-multilingual-*" plus généraliste car E5 est entraîné
spécifiquement pour la RECHERCHE (retrieval), avec des embeddings
asymétriques requête/passage, et couvre le français et l'arabe. Contexte
jusqu'à 512 tokens (contre 128 par défaut pour beaucoup de modèles
"paraphrase-*"), important ici vu la longueur de certains articles.

Convention E5 (à respecter, sinon la qualité de recherche se dégrade
nettement) : préfixer chaque passage à indexer par "passage: " et chaque
requête de recherche par "query: " avant encodage — c'est le modèle lui
gère différemment les deux rôles.

Note environnement : le modèle par défaut (~1.1 Go) est volontairement
téléchargé à la demande (pas au chargement du module) pour ne pas
pénaliser les imports quand seule l'API est utilisée sans modèle chargé
(ex. tests unitaires sur la logique d'indexation).
"""
from __future__ import annotations

from typing import Iterable

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"

_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "


class Embedder:
    """
    Wrapper autour de sentence-transformers, avec les préfixes E5
    query/passage appliqués automatiquement — le reste du pipeline
    (index_builder, search) n'a jamais à s'en soucier directement.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        # Import différé : évite de charger torch/sentence-transformers
        # (lourd) pour du code qui ne fait qu'importer ce module sans
        # jamais instancier Embedder (ex. tests sur index_builder avec un
        # embedder factice).
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: Iterable[str], batch_size: int = 32, show_progress: bool = True):
        """Encode des textes à INDEXER (articles). Renvoie un array numpy (n, dim)."""
        texts = [_PASSAGE_PREFIX + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # produit similarité cosinus = produit scalaire -> compatible IndexFlatIP
            convert_to_numpy=True,
        )

    def embed_query(self, text: str):
        """Encode UNE requête de recherche. Renvoie un array numpy (dim,)."""
        return self.model.encode(
            _QUERY_PREFIX + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
