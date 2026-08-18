"""
run_consolidation.py
-----------------------
Étape 4-bis : Consolidation + persistance.

Parcourt les fichiers JSON produits par scripts/run_extraction.py dans
data/annotated/ (un fichier par document, motif "*_entities.json"),
déduplique les entités au niveau document (document_consolidator.py) et
sauvegarde le résultat dans la base SQLite (db_connector.py).

Usage :
    python -m scripts.run_consolidation
    # ou pour un seul fichier :
    python -m scripts.run_consolidation --file data/annotated/BO_7500_Fr_entities.json
    # pour choisir l'emplacement de la base :
    python -m scripts.run_consolidation --db data/processed/juridique.db
"""

import argparse
import json
from pathlib import Path

from src.storage.document_consolidator import consolidate_document
from src.storage.db_connector import DBConnector

ANNOTATED_DIR = Path("data/annotated")
DEFAULT_DB_PATH = "data/processed/juridique.db"


def _load_doc_metadata(doc: dict, fallback_doc_id: str) -> dict:
    """
    Reconstruit le dict de métadonnées attendu par consolidate_document() à
    partir du JSON produit par run_extraction.py.

    fallback_doc_id : utilisé si le JSON provient d'une exécution antérieure
    à l'ajout de document_metadata_extractor (pas de champ "doc_id").
    """
    return {
        "doc_id": doc.get("doc_id", fallback_doc_id),
        "lang": doc.get("lang", "fr"),
        "bo_number": doc.get("bo_number"),
        "date_publication": doc.get("date_publication"),
        "edition_label": doc.get("edition_label"),
    }


def consolidate_and_save(json_path: Path, db: DBConnector) -> dict:
    """
    Charge un JSON annoté (étape 3+4), le consolide et l'enregistre en base.

    Returns:
        Résumé (doc_id, nombre d'articles, entités dédupliquées, citations).
    """
    doc = json.loads(json_path.read_text(encoding="utf-8"))

    fallback_doc_id = json_path.stem.removesuffix("_entities")
    doc_metadata = _load_doc_metadata(doc, fallback_doc_id)

    articles = doc.get("articles", [])
    if not articles:
        raise ValueError(
            f"Aucun article dans {json_path.name} — relance scripts/run_extraction.py "
            f"avant la consolidation."
        )

    consolidated = consolidate_document(doc_metadata, articles)
    db.save_document(consolidated)

    entities_index = consolidated["entities_index"]
    return {
        "doc_id": consolidated["doc_id"],
        "num_articles": consolidated["num_articles"],
        "persons": len(entities_index["persons"]),
        "organizations": len(entities_index["organizations"]),
        "legal_texts": len(entities_index["legal_texts"]),
        "citations": len(consolidated["citations_graph"]),
        "citations_resolved": sum(1 for c in consolidated["citations_graph"] if c["resolved"]),
    }


def process_single_file(file_path: str, db_path: str) -> dict:
    path = Path(file_path)
    db = DBConnector(db_path)
    summary = consolidate_and_save(path, db)
    _print_summary(path.name, summary)
    return summary


def process_all_files(db_path: str) -> list:
    """
    Parcourt data/annotated/*_entities.json (sortie de run_extraction.py) et
    consolide/sauvegarde chaque document dans la même base SQLite.
    """
    json_files = sorted(ANNOTATED_DIR.glob("*_entities.json"))

    if not json_files:
        print(f"Aucun fichier '*_entities.json' trouvé dans {ANNOTATED_DIR}/. "
              f"Lance d'abord scripts/run_extraction.py.")
        return []

    db = DBConnector(db_path)
    summaries = []

    for json_path in json_files:
        try:
            summary = consolidate_and_save(json_path, db)
            _print_summary(json_path.name, summary)
            summaries.append(summary)
        except Exception as e:
            print(f"  ✗ Erreur sur {json_path.name} : {e}")
            summaries.append({"source": str(json_path), "error": str(e)})

    n_ok = sum(1 for s in summaries if "error" not in s)
    print(f"\n{n_ok}/{len(json_files)} document(s) consolidé(s) et sauvegardé(s) dans {db_path}")
    return summaries


def _print_summary(filename: str, summary: dict) -> None:
    print(f"Consolidation : {filename}")
    print(
        f"  → {summary['num_articles']} article(s) | "
        f"{summary['persons']} personne(s) | "
        f"{summary['organizations']} organisation(s) | "
        f"{summary['legal_texts']} texte(s) légal/légaux | "
        f"{summary['citations_resolved']}/{summary['citations']} citation(s) résolue(s)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolide les entités par document et les sauvegarde en base SQLite (étape 4-bis)."
    )
    parser.add_argument("--file", type=str, help="Traiter un seul fichier JSON au lieu de tout data/annotated/")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help=f"Chemin de la base SQLite (défaut: {DEFAULT_DB_PATH})")
    args = parser.parse_args()

    if args.file:
        process_single_file(args.file, args.db)
    else:
        process_all_files(args.db)
