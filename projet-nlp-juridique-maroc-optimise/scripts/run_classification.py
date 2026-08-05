#!/usr/bin/env python
"""
scripts/run_classification.py
Tester la classification d'un document juridique.

Usage:
    python -m scripts.run_classification <fichier> [--lang {fr,ar}] [--model]

Le flag --model utilise le classifieur transformer fine-tuné (s'il est présent
dans models/domain_classifier) au lieu des mots-clés ; sinon repli automatique
sur les mots-clés.

Exemples:
    python -m scripts.run_classification data/processed/ar/BO_7506_Ar.txt --lang ar
    python -m scripts.run_classification data/processed/ar/BO_7506_Ar.json --lang ar
"""

import argparse
import json
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH pour pouvoir importer src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classification.keyword_classifier import classify_text, classify_document
from src.classification.transformer_classifier import TransformerDomainClassifier
from src.preprocessing.arabic_utils import arabic_char_ratio


def detect_language_from_text(text: str) -> str:
    return "ar" if arabic_char_ratio(text) > 0.30 else "fr"


def load_text_from_file(file_path: Path, lang: str | None = None) -> str:
    """Charge le texte depuis un fichier .txt ou extrait le texte depuis un JSON."""
    if file_path.suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        # 1. Si le JSON contient le texte brut (champ 'full_text')
        if "full_text" in doc:
            return doc["full_text"]

        # 2. Sinon, reconstruire à partir des articles (si chaque article a un champ 'text')
        if "articles" in doc:
            parts = []
            for article in doc["articles"]:
                if "text" in article:
                    parts.append(article["text"])
            if parts:
                return "\n".join(parts)

        # 3. Fallback : utiliser le fichier source original (.txt) si le champ 'source' existe
        if "source" in doc:
            source_path = Path(doc["source"])
            if source_path.exists():
                return load_text_from_file(source_path, lang)

        raise ValueError("Impossible d'extraire le texte du fichier JSON.")

    else:
        raise ValueError("Format de fichier non supporté. Utilisez .txt ou .json.")


def main():
    parser = argparse.ArgumentParser(description="Classifier un document juridique par domaine.")
    parser.add_argument("file", type=Path, help="Chemin vers le fichier .txt ou .json")
    parser.add_argument("--lang", choices=["fr", "ar"], default=None,
                        help="Langue du texte (optionnel : détectée automatiquement depuis le contenu)")
    parser.add_argument("--model", action="store_true",
                        help="Utilise le classifieur transformer fine-tuné s'il est disponible")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Erreur: fichier introuvable: {args.file}")
        sys.exit(1)

    try:
        text = load_text_from_file(args.file)
    except Exception as e:
        print(f"Erreur lors du chargement du texte: {e}")
        sys.exit(1)

    if args.lang is None:
        args.lang = detect_language_from_text(text)
        print(f"Langue détectée : {args.lang}")

    # Classification
    if args.model:
        clf = TransformerDomainClassifier()
        print(f"Classifieur : {clf.status}")
        if args.file.suffix == ".json":
            with open(args.file, "r", encoding="utf-8") as f:
                doc = json.load(f)
            json_lang = doc.get("lang")
            if json_lang in ("fr", "ar"):
                args.lang = json_lang
            domain = clf.classify_document(doc, lang=args.lang)
        else:
            domain = clf.classify_text(text, lang=args.lang)
    elif args.file.suffix == ".json":
        # Utiliser la fonction qui traite le document entier (reconstruit le texte)
        with open(args.file, "r", encoding="utf-8") as f:
            doc = json.load(f)
        # Si le JSON a un champ lang, on l'utilise en priorité
        json_lang = doc.get("lang")
        if json_lang in ("fr", "ar"):
            args.lang = json_lang
        domain = classify_document(doc, lang=args.lang)
    else:
        # Texte brut
        domain = classify_text(text, lang=args.lang)

    print(f"Domaine détecté : {domain}")


if __name__ == "__main__":
    main()