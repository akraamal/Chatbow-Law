import argparse
import json
from pathlib import Path

from src.preprocessing.arabic_utils import arabic_char_ratio

from scripts.run_extraction import (
    extract_entities_from_file,
    build_fr_nlp,
    build_ar_nlp,
)

ANNOTATED_DIR = Path("data/annotated")


def detect_language_from_text(text: str) -> str:
    return "ar" if arabic_char_ratio(text) > 0.30 else "fr"


def detect_language(path: Path) -> str:
    """Détecte la langue : d'abord par le chemin (dossier fr/ ou ar/),
    puis par le contenu du fichier si le chemin ne permet pas de décider."""
    for part in path.parts:
        if part == "ar":
            return "ar"
        if part == "fr":
            return "fr"
    text_sample = path.read_text(encoding="utf-8")[:1000]
    return detect_language_from_text(text_sample)


def process_file(path: Path, lang: str):

    if lang == "fr":
        nlp_fr = build_fr_nlp()
        result = extract_entities_from_file(
            path,
            lang,
            nlp_fr=nlp_fr,
        )

    else:
        nlp_ar = build_ar_nlp()
        result = extract_entities_from_file(
            path,
            lang,
            nlp_ar=nlp_ar,
        )
        print(result["lang"])

    output_dir = ANNOTATED_DIR / lang
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"{path.stem}.json"

    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✔ Sauvegardé : {output}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        help="Fichier texte"
    )

    args = parser.parse_args()

    path = Path(args.input_file)
    lang = detect_language(path)
    print(f"Langue détectée : {lang}")

    process_file(path, lang)


if __name__ == "__main__":
    main()