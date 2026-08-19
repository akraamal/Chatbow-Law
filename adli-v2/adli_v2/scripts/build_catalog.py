"""
adli_v2.scripts.build_catalog
-----------------------------
Construit le catalogue décret-first depuis adli-v2/data/annotated/ et
l'écrit dans adli-v2/data/catalog.json.

Usage (depuis la racine du dépôt) :
    python -m adli_v2.scripts.build_catalog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adli_v2.catalog import build_catalog, save_catalog  # noqa: E402
from adli_v2.pipeline import DEFAULT_ANNOTATED, DEFAULT_DATA  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalogue décret-first v2.")
    parser.add_argument("--annotated", type=Path, default=DEFAULT_ANNOTATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA / "catalog.json")
    args = parser.parse_args()

    entries = build_catalog(args.annotated)
    save_catalog(entries, args.output)
    print(f"{len(entries)} instrument(s) catalogué(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())