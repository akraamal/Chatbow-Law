"""
adli_v2.scripts.run_extraction
------------------------------
CLI de traitement v2 : un PDF (ou un dossier) → JSON enrichi v2 dans
adli-v2/data/annotated/ (métadonnées + compteurs de mots-clés inclus).

Usage (depuis la racine du dépôt) :
    python -m adli_v2.scripts.run_extraction --file chemin/vers/document.pdf
    python -m adli_v2.scripts.run_extraction --dir chemin/vers/dossier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adli_v2.pipeline import (  # noqa: E402
    DEFAULT_ANNOTATED,
    DEFAULT_UPLOADS,
    process_pdf,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline v2 : PDF → JSON enrichi (métadonnées + compteurs).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Traiter un seul PDF")
    group.add_argument("--dir", type=Path, help="Traiter tous les PDF d'un dossier")
    parser.add_argument("--output", type=Path, default=DEFAULT_ANNOTATED,
                        help="Dossier de sortie des JSON annotés")
    parser.add_argument("--uploads", type=Path, default=DEFAULT_UPLOADS,
                        help="Dossier où chercher les PDF source (backfill des pages)")
    args = parser.parse_args()

    pdfs = [args.file] if args.file else sorted(args.dir.glob("**/*.pdf"))
    if not pdfs:
        print("Aucun PDF à traiter.")
        return 1

    for pdf in pdfs:
        print(f"\n=== {pdf.name} ===")
        try:
            out = process_pdf(pdf, annotated_dir=args.output, uploads_dir=args.uploads)
            for p in out:
                print(f"  OK -> {p}")
        except Exception as exc:  # noqa: BLE001 — CLI: on continue sur les suivants
            import traceback
            print(f"  ECHEC : {exc}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())