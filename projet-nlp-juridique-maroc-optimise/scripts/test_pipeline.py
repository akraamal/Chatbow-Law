"""
test_pipeline.py

Traite un PDF puis sauvegarde automatiquement le texte
dans data/interim/fr, data/interim/ar et data/interim/unknown
selon la langue détectée.

Usage:
    python test_pipeline.py data/raw/mon_document.pdf
"""

from pathlib import Path
import sys

from src.ingestion.pipeline import process_pdf


OUTPUT_ROOT = Path("data/interim")


def save_text(output_dir: Path, filename: str, text: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / filename

    output_file.write_text(
        text,
        encoding="utf-8"
    )

    return output_file


def main(pdf_path: str):

    pdf = Path(pdf_path)

    if not pdf.exists():
        raise FileNotFoundError(pdf)

    document = process_pdf(str(pdf))

    # Regrouper le texte par langue
    texts = {
        "fr": [],
        "ar": [],
        "unknown": []
    }

    for page in document.pages:

        lang = getattr(page, "language", "unknown")

        if lang not in texts:
            lang = "unknown"

        texts[lang].append(page.text)

    # Sauvegarde
    saved_files = []

    for lang, pages in texts.items():

        if not pages:
            continue

        output = save_text(
            OUTPUT_ROOT / lang,
            f"{pdf.stem}.txt",
            "\n\n".join(pages)
        )

        saved_files.append(output)

    # Résumé
    print("=" * 60)
    print(f"Document : {pdf.name}")
    print(f"Pages    : {document.n_pages}")
    print("=" * 60)

    print()

    for page in document.pages:

        print(
            f"Page {page.page_number:3d} | "
            f"{page.language:8s} | "
            f"{page.extraction_method}"
        )

    print("\nFichiers générés :\n")

    for f in saved_files:
        print(f" - {f}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage : python test_pipeline.py <pdf>")
        sys.exit(1)

    main(sys.argv[1])