"""
run_ingestion_batch.py
------------------------
Traite tous les PDF présents dans data/raw/ (fr/ et ar/), applique le
pipeline d'ingestion (src/ingestion/pipeline.py) sur chacun, et
sauvegarde le texte extrait sous forme de fichiers .txt dans data/interim/.

Structure de sortie :
    data/interim/fr/<nom_du_pdf>.txt   → texte français extrait
    data/interim/ar/<nom_du_pdf>.txt   → texte arabe extrait (si présent)

Un fichier .txt n'est créé que si le texte correspondant n'est pas vide,
pour éviter de polluer data/interim/ar/ avec des fichiers vides quand un
document est mono-langue français (et inversement).

Usage :
    python -m scripts.run_ingestion_batch
    # ou pour un seul fichier :
    python -m scripts.run_ingestion_batch --file chemin/vers/document.pdf
"""

import argparse
import json
from pathlib import Path

from src.ingestion.pipeline import run_ingestion_pipeline, IngestionResult

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")


def _save_result(result: IngestionResult, pdf_path: Path) -> dict:
    """
    Sauvegarde le texte extrait dans data/interim/fr/ ou data/interim/ar/.

    Si le document est en vraies colonnes bilingues (detected_layout ==
    "colonnes"), les deux langues sont sauvegardées séparément (cas normal
    du Bulletin Officiel bilingue).

    Sinon (document mono-langue), on ne garde que la langue dominante et on
    n'écrit qu'un seul fichier, dans le bon dossier.
    """
    saved_files = []
    stem = pdf_path.stem

    is_bilingual_columns = result.detected_layout == "colonnes"

    if is_bilingual_columns:
        if result.text_fr.strip():
            out_path = INTERIM_DIR / "fr" / f"{stem}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.text_fr, encoding="utf-8")
            saved_files.append(str(out_path))

        if result.text_ar.strip():
            out_path = INTERIM_DIR / "ar" / f"{stem}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.text_ar, encoding="utf-8")
            saved_files.append(str(out_path))

    else:
        len_fr = len(result.text_fr.strip())
        len_ar = len(result.text_ar.strip())

        if len_fr == 0 and len_ar == 0:
            dominant_lang, dominant_text = None, ""
        elif len_fr >= len_ar:
            dominant_lang, dominant_text = "fr", result.text_fr
        else:
            dominant_lang, dominant_text = "ar", result.text_ar

        if dominant_lang:
            out_path = INTERIM_DIR / dominant_lang / f"{stem}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(dominant_text, encoding="utf-8")
            saved_files.append(str(out_path))

    if result.text_unknown.strip():
        # Texte que le pipeline n'a pas pu attribuer avec certitude à une langue :
        # sauvegardé à part pour ne pas le perdre, mais signalé pour vérification manuelle.
        out_path = INTERIM_DIR / f"{stem}_unknown.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.text_unknown, encoding="utf-8")
        saved_files.append(str(out_path))

    if result.tables:
        # Tableaux extraits séparément, structure ligne/colonne préservée
        out_path = INTERIM_DIR / "tables" / f"{stem}_tables.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.tables, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        saved_files.append(str(out_path))

    return {
        "source": str(pdf_path),
        "used_ocr": result.used_ocr,
        "detected_layout": result.detected_layout,
        "saved_files": saved_files,
        "warnings": result.warnings,
    }


def process_single_pdf(pdf_path: str) -> dict:
    """Traite un seul PDF et sauvegarde le résultat."""
    path = Path(pdf_path)
    print(f"Traitement : {path.name}")
    result = run_ingestion_pipeline(str(path))
    summary = _save_result(result, path)

    for f in summary["saved_files"]:
        print(f"  → écrit : {f}")
    for w in summary["warnings"]:
        print(f"  ⚠ {w}")

    return summary


def process_all_pdfs() -> list:
    """
    Parcourt data/raw/fr/ et data/raw/ar/ à la recherche de tous les PDF,
    et les traite un par un. Un même document peut être placé dans l'un ou
    l'autre dossier selon sa langue dominante à l'origine — le pipeline
    d'ingestion se charge de toute façon de retrier le contenu par langue
    réelle une fois le texte extrait.
    """
    pdf_files = list(RAW_DIR.glob("**/*.pdf"))

    if not pdf_files:
        print(f"Aucun PDF trouvé dans {RAW_DIR}/. Dépose tes fichiers dans data/raw/fr/ ou data/raw/ar/.")
        return []

    print(f"{len(pdf_files)} fichier(s) PDF trouvé(s).\n")

    all_summaries = []
    for pdf_path in pdf_files:
        try:
            summary = process_single_pdf(str(pdf_path))
            all_summaries.append(summary)
        except Exception as e:
            import traceback
            print(f"  ✗ Erreur sur {pdf_path.name} : {e}")
            traceback.print_exc()
            all_summaries.append({"source": str(pdf_path), "error": str(e)})
        print()

    # Sauvegarde d'un rapport de lot (utile pour vérifier rapidement quels
    # documents ont posé problème, sans avoir à tout relire dans le terminal)
    report_path = INTERIM_DIR / "_rapport_ingestion.json"
    report_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rapport de lot sauvegardé : {report_path}")

    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traite les PDF juridiques et sauvegarde le texte extrait.")
    parser.add_argument("--file", type=str, help="Traiter un seul fichier PDF au lieu de tout data/raw/")
    args = parser.parse_args()

    if args.file:
        process_single_pdf(args.file)
    else:
        process_all_pdfs()
