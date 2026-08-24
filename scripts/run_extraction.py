"""
run_extraction.py
--------------------
Étape 3 : Extraction NLP.

Parcourt data/processed/fr/ et data/processed/ar/ (texte nettoyé, voir
cleaner_fr.py / cleaner_ar.py — étape 2), découpe chaque fichier en
articles (segmenter.py), extrait les entités légales de chaque article
(entity_ruler_builder_fr.py / entity_ruler_builder_ar.py), et sauvegarde le
résultat en JSON structuré dans data/annotated/.

Usage :
    python -m scripts.run_extraction
    # ou pour un seul fichier :
    python -m scripts.run_extraction --file data/processed/fr/BO_7500_Fr.txt --lang fr
"""

import argparse
import json
from pathlib import Path

from src.extraction.etape4_pipeline import enrich_article_json
from src.extraction.document_metadata_extractor import (
    extract_document_metadata,
    resolve_raw_pdf_path,
)
from src.extraction.ner_filter import filter_entities
from src.preprocessing.arabic_utils import arabic_char_ratio
from src.preprocessing.segmenter import segment_into_articles, get_preamble
from src.extraction.entity_ruler_builder_fr import build_fr_nlp, extract_legal_entities_fr
from src.extraction.entity_ruler_builder_ar import build_ar_nlp, extract_legal_entities_ar
from src.ingestion.pipeline import ensure_interim_fresh

RAW_DIR = Path("data/raw")   # PDFs bruts : source du repli OCR en-tête

PROCESSED_DIR = Path("data/processed")
ANNOTATED_DIR = Path("data/annotated")


def _ensure_processed_fresh(processed_file: Path) -> None:
    """Refuse d'extraire un JSON depuis un texte processed/ dont l'interim
    correspondant est stale (version d'extracteur antérieure ou PDF modifié)
    — même garde que run_pipeline_complet.run_extraction()."""
    parts = list(processed_file.parts)
    interim = Path(*["interim" if p == "processed" else p for p in parts])
    if interim.exists():
        ensure_interim_fresh(interim)


def _entities_to_dicts(doc) -> list:
    """Convertit doc.ents (spans spaCy) en liste de dicts sérialisables JSON."""
    return [
        {
            "label": ent.label_,
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]


def extract_entities_from_file(input_path: Path, lang: str, nlp_fr=None, nlp_ar=None) -> dict:
    """
    Segmente un fichier texte nettoyé en articles et extrait les entités
    légales de chaque article (+ du préambule, souvent riche en
    références : "vu le dahir...", "vu la loi...").

    Returns:
        dict prêt à être sérialisé en JSON, avec une entrée par article
        (numéro, texte, entités) et une entrée "preamble" séparée.
    """
    text = input_path.read_text(encoding="utf-8")
    print(text[:500])

    extract_fn = extract_legal_entities_fr if lang == "fr" else extract_legal_entities_ar
    nlp = nlp_fr if lang == "fr" else nlp_ar

    preamble = get_preamble(text, lang=lang)
    preamble_doc = extract_fn(preamble, nlp=nlp)

    articles = segment_into_articles(text, lang=lang)
    print("Nombre d'articles :", len(articles))
    articles_out = [
    process_article(art, extract_fn, nlp, input_path, lang)
    for art in articles
    ]


    return {
        "source": str(input_path),
        "lang": lang,
        **extract_document_metadata(
            text,
            doc_id=input_path.stem,
            lang=lang,
            pdf_path=resolve_raw_pdf_path(input_path.stem, RAW_DIR),  # repli OCR en-tête
        ),
        "preamble_entities": filter_entities(_entities_to_dicts(preamble_doc)),
        "n_articles": len(articles_out),
        "articles": articles_out,
    }


def detect_language_from_text(text: str) -> str:
    """Détecte la langue à partir du ratio de caractères arabes."""
    return "ar" if arabic_char_ratio(text) > 0.30 else "fr"


def process_single_file(input_path: str, lang: str | None = None) -> Path:
    """Traite un seul fichier et sauvegarde le résultat en JSON."""
    path = Path(input_path)
    _ensure_processed_fresh(path)
    text_sample = path.read_text(encoding="utf-8")[:1000]
    if lang is None:
        lang = detect_language_from_text(text_sample)
    print(f"Extraction : {path.name} ({lang})")

    nlp_fr = build_fr_nlp() if lang == "fr" else None
    nlp_ar = build_ar_nlp() if lang == "ar" else None

    result = extract_entities_from_file(path, lang, nlp_fr=nlp_fr, nlp_ar=nlp_ar)

    out_path = ANNOTATED_DIR / f"{path.stem}_entities.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    n_entities = sum(len(a["entities"]) for a in result["articles"]) + len(result["preamble_entities"])
    print(f"  {result['n_articles']} article(s), {n_entities} entité(s) trouvée(s)")
    print(f"  → écrit : {out_path}")

    return out_path

def process_article(art, extract_fn, nlp, input_path: Path, lang: str):
    """
    Traite un article :
        - étape 3 : EntityRuler
        - étape 4 : enrichissement
    """

    doc = extract_fn(art.text, nlp=nlp)

    article = {
        "number": art.number,
        "raw_header": art.raw_header,
        "entities": filter_entities(_entities_to_dicts(doc)),
        "dates": [],
    }

    article = enrich_article_json(
        article=article,
        full_text=art.text,
        doc_id=input_path.stem,
        lang=lang,
    )

    return article


def process_all_files() -> list:
    """
    Parcourt data/processed/fr/ et data/processed/ar/, traite chaque
    fichier .txt trouvé. Les pipelines spaCy (nlp_fr / nlp_ar) sont
    construits une seule fois et réutilisés pour tous les fichiers d'une
    même langue, pour éviter de recharger l'EntityRuler à chaque fichier.
    """
    saved = []

    for lang in ("fr", "ar"):
        lang_dir = PROCESSED_DIR / lang
        txt_files = sorted(lang_dir.glob("*.txt")) if lang_dir.exists() else []

        if not txt_files:
            continue

        nlp = build_fr_nlp() if lang == "fr" else build_ar_nlp()

        for txt_path in txt_files:
            try:
                if lang == "fr":
                    result = extract_entities_from_file(txt_path, lang, nlp_fr=nlp)
                else:
                    result = extract_entities_from_file(txt_path, lang, nlp_ar=nlp)

                out_path = ANNOTATED_DIR / f"{txt_path.stem}_entities.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"Extraction : {txt_path.name} ({lang}) → {out_path}")
                saved.append(str(out_path))

            except Exception as e:
                print(f"  ✗ Erreur sur {txt_path.name} : {e}")

    if not saved:
        print(f"Aucun fichier trouvé dans {PROCESSED_DIR}/fr ou {PROCESSED_DIR}/ar. "
              f"Lance d'abord l'étape 2 (scripts/run_preprocess.py).")

    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrait les entités légales des textes nettoyés (étape 3).")
    parser.add_argument("--file", type=str, help="Traiter un seul fichier au lieu de tout data/processed/")
    parser.add_argument("--lang", choices=["fr", "ar"], default=None,
                        help="Langue du fichier (optionnel : détectée automatiquement depuis le contenu)")
    args = parser.parse_args()

    if args.file:
        process_single_file(args.file, args.lang)
    else:
        process_all_files()
