"""
preprocess.py
-------------
Pipeline de prétraitement des fichiers texte.

Usage :

Détection automatique :
python -m src.preprocessing.preprocess --file data/interim/fr/BO_7500_Fr.txt

Langue imposée :
python -m src.preprocessing.preprocess --file data/interim/ar/BO_7511_Ar.txt --lang ar
"""

from pathlib import Path
import argparse

from src.preprocessing.cleaner_fr import clean_french_text
from src.preprocessing.cleaner_ar import clean_arabic_text
from src.preprocessing.arabic_utils import arabic_char_ratio




def detect_language(text: str) -> str:
    """
    Détecte la langue à partir de la proportion de caractères arabes.
    """
    return "ar" if arabic_char_ratio(text) > 0.30 else "fr"


def preprocess_file(input_path: Path, lang: str | None = None) -> Path:
    """
    Nettoie un fichier texte et sauvegarde le résultat dans
    data/processed/<lang>.
    """

    raw_text = input_path.read_text(encoding="utf-8")

    if lang is None:
        lang = detect_language(raw_text)

    if lang == "ar":
        cleaned = clean_arabic_text(raw_text)
    elif lang == "fr":
        cleaned = clean_french_text(raw_text)
    else:
        raise ValueError("Langue inconnue : choisir 'fr' ou 'ar'.")

    # data/interim/ar/fichier.txt -> data/processed/ar/fichier.txt
    # (remplacement sur le nom du dossier, indépendant de l'OS — l'ancienne
    # version utilisait "data\\interim"/"data\\processed" avec des
    # antislashs, qui ne correspondaient à rien sous Linux/macOS : le
    # fichier était alors réécrit silencieusement dans data/interim/ au
    # lieu de data/processed/)
    parts = list(input_path.parts)
    parts = ["processed" if p == "interim" else p for p in parts]
    output_path = Path(*parts)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(cleaned, encoding="utf-8")
    return output_path


def main():

    parser = argparse.ArgumentParser(
        description="Prétraitement d'un fichier texte."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Chemin vers le fichier txt"
    )

    parser.add_argument(
        "--lang",
        choices=["fr", "ar"],
        default=None,
        help="Langue (optionnelle). Si absente, elle est détectée automatiquement."
    )

    args = parser.parse_args()

    output = preprocess_file(Path(args.file), args.lang)

    print(f"Fichier traité avec succès.")
    print(f"Résultat enregistré dans : {output}")


if __name__ == "__main__":
    main()