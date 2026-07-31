"""
build_search_index.py
-------------------------
Étape 6 : construit l'index de recherche sémantique (FAISS) à partir des
articles déjà en base (data/processed/juridique.db, voir
scripts/run_consolidation.py).

Usage :
    python -m scripts.build_search_index
    python -m scripts.build_search_index --model intfloat/multilingual-e5-base
    python -m scripts.build_search_index --db data/processed/juridique.db --out data/index
"""
import argparse

from src.search_engine.index_builder import build_index, DEFAULT_INDEX_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construit l'index FAISS de recherche sémantique (étape 6)."
    )
    parser.add_argument("--db", type=str, default="data/processed/juridique.db",
                         help="Base SQLite source (défaut: data/processed/juridique.db)")
    parser.add_argument("--out", type=str, default=DEFAULT_INDEX_DIR,
                         help=f"Répertoire de sortie de l'index (défaut: {DEFAULT_INDEX_DIR})")
    parser.add_argument("--model", type=str, default=None,
                         help="Modèle sentence-transformers à utiliser (défaut: intfloat/multilingual-e5-base)")
    parser.add_argument("--min-chars", type=int, default=20,
                         help="Longueur minimale (caractères) pour indexer un article (défaut: 20)")
    args = parser.parse_args()

    print("Construction de l'index sémantique — le téléchargement du modèle "
          "(première exécution) peut prendre quelques minutes selon la connexion.")

    summary = build_index(
        articles=args.db,
        index_dir=args.out,
        model_name=args.model,
        min_chars=args.min_chars,
    )

    print(f"\nIndex construit : {summary['n_articles']} article(s) indexé(s)")
    print(f"  Modèle       : {summary['model_name']} (dimension {summary['dimension']})")
    print(f"  Index FAISS  : {summary['index_path']}")
    print(f"  Métadonnées  : {summary['metadata_path']}")
