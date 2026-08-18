"""
build_training_dataset.py
----------------------------
Étape 5 (préparation) : construit un jeu de données étiqueté à partir des
articles déjà en base (data/processed/juridique.db, voir
scripts/run_consolidation.py), pour le fine-tuning d'un transformer de
classification par domaine juridique.

Le classifieur par mots-clés (keyword_classifier.py) sert de PRÉ-étiquetage
automatique — un point de départ à corriger, pas une vérité terrain. Ce
script ne remplace pas la relecture humaine : il l'accélère.

Colonnes du CSV produit :
    id              identifiant unique de la ligne (= id de l'article en base)
    doc_id          document source (traçabilité, ex: "BO_7500_Fr")
    article_number  numéro d'article dans le document
    lang            "fr" ou "ar"
    text            texte de l'article (à lire pour vérifier le label)
    keyword_label   label proposé automatiquement par keyword_classifier
    top_scores      les 3 meilleurs scores mots-clés, pour juger de la
                    confiance du pré-étiquetage (ex: score du 1er domaine
                    très supérieur aux autres = pré-étiquetage fiable ;
                    scores proches ou tous à 0 = à vérifier en priorité)
    label           colonne à corriger à la main — initialisée à
                    keyword_label, c'est CETTE colonne que le script
                    d'entraînement (étape B) lira comme vérité terrain
    truncated       True si le texte a été coupé (annexe/tableau probable
                    avalé en fin d'article, faute de marqueur suivant — voir
                    segmenter.py) — à vérifier en priorité, le label peut
                    être moins fiable sur un texte tronqué

Usage :
    python -m scripts.build_training_dataset
    python -m scripts.build_training_dataset --db data/processed/juridique.db --out data/training/domain_dataset.csv
    python -m scripts.build_training_dataset --min-chars 40   # ignore les articles trop courts pour être classables
"""

import argparse
import csv
import sqlite3
from pathlib import Path

from src.classification.keyword_classifier import classify_text_with_scores

DEFAULT_DB_PATH = "data/processed/juridique.db"
DEFAULT_OUT_PATH = "data/training/domain_dataset.csv"
DEFAULT_MIN_CHARS = 30  # articles plus courts que ça (ex: "Le présent arrêté sera publié...") ne portent pas assez de signal pour la classification par domaine
DEFAULT_MAX_CHARS = 6000  # au-delà, il s'agit presque toujours d'une annexe/tableau avalé par le dernier article faute de marqueur suivant (voir segmenter.py) — on tronque plutôt que de polluer le dataset avec un outlier énorme


def fetch_articles(db_path: str):
    """Lit tous les articles (avec leur texte et la langue du document parent)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT a.id, a.doc_id, a.number AS article_number, a.raw_text, d.lang
            FROM articles a
            JOIN documents d ON d.doc_id = a.doc_id
            ORDER BY a.doc_id, a.id
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def build_dataset(db_path: str, out_path: str, min_chars: int, max_chars: int) -> dict:
    articles = fetch_articles(db_path)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_written, n_skipped_short, n_indetermine, n_truncated = 0, 0, 0, 0

    # encoding="utf-8-sig" : Excel (Windows) affiche mal l'UTF-8 sans BOM,
    # notamment pour l'arabe — le BOM lui fait détecter le bon encodage.
    with open(out_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "doc_id", "article_number", "lang", "text",
            "keyword_label", "top_scores", "label", "truncated",
        ])

        for art in articles:
            text = (art["raw_text"] or "").strip()
            if len(text) < min_chars:
                n_skipped_short += 1
                continue

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
                n_truncated += 1

            scores = classify_text_with_scores(text, lang=art["lang"])
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_domain, best_score = ranked[0]
            keyword_label = best_domain if best_score > 0 else "Indéterminé"
            if keyword_label == "Indéterminé":
                n_indetermine += 1

            top_scores = ", ".join(f"{d}={s}" for d, s in ranked[:3])

            writer.writerow([
                art["id"], art["doc_id"], art["article_number"], art["lang"],
                text, keyword_label, top_scores, keyword_label, truncated,
            ])
            n_written += 1

    return {
        "out_path": str(out_file),
        "n_written": n_written,
        "n_skipped_short": n_skipped_short,
        "n_indetermine": n_indetermine,
        "n_truncated": n_truncated,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exporte un CSV pré-étiqueté (mots-clés) à corriger à la main avant fine-tuning."
    )
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help=f"Base SQLite (défaut: {DEFAULT_DB_PATH})")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_PATH, help=f"Fichier CSV de sortie (défaut: {DEFAULT_OUT_PATH})")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                         help=f"Longueur minimale (caractères) pour garder un article (défaut: {DEFAULT_MIN_CHARS})")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                         help=f"Longueur au-delà de laquelle un article est tronqué (défaut: {DEFAULT_MAX_CHARS})")
    args = parser.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(
            f"Base introuvable : {args.db}\n"
            f"Lance d'abord scripts/run_extraction.py puis scripts/run_consolidation.py."
        )

    summary = build_dataset(args.db, args.out, args.min_chars, args.max_chars)

    print(f"Dataset écrit : {summary['out_path']}")
    print(f"  {summary['n_written']} article(s) exporté(s)")
    print(f"  {summary['n_skipped_short']} article(s) ignoré(s) (texte trop court)")
    print(f"  {summary['n_truncated']} article(s) tronqué(s) (annexe/tableau probable en fin d'article — colonne 'truncated')")
    print(f"  {summary['n_indetermine']} article(s) pré-étiqueté(s) 'Indéterminé' (aucun mot-clé trouvé — à vérifier en priorité)")
    print()
    print("Prochaine étape : ouvre le CSV (Excel/Google Sheets) et corrige la colonne "
          "'label' pour chaque ligne (garde la valeur proposée si elle est correcte).")
