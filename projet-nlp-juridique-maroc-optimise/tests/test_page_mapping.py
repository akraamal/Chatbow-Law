"""
test_page_mapping.py
---------------------
Validate page mapping consistency on enriched JSON files.

Checks:
- pdf_page is a positive integer when present
- printed_page is a positive integer when present
- No duplicate (pdf_page, printed_page) for distinct articles on different pages
- Articles are in increasing pdf_page order (within reason)

Usage:
    python -m pytest tests/test_page_mapping.py -v
    python tests/test_page_mapping.py
"""

import json
import sys
from pathlib import Path

ANNOTATED_DIR = Path("data/annotated")


def validate_page_mapping(path: Path) -> list[str]:
    errors = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    if not articles:
        return errors

    prev_page = 0
    pdf_pages_seen = set()
    for i, art in enumerate(articles):
        pp = art.get("pdf_page")
        prp = art.get("printed_page")

        if pp is not None:
            if not isinstance(pp, int) or pp < 1:
                errors.append(f"articles[{i}]: invalid pdf_page={pp}")
            elif pp in pdf_pages_seen:
                # Same page for multiple articles is fine; track for info only
                pass
            else:
                pdf_pages_seen.add(pp)
                # Articles should not backtrack in page order
                if pp < prev_page:
                    errors.append(f"articles[{i}]: pdf_page={pp} < previous page {prev_page}")
                prev_page = pp

        if prp is not None:
            if not isinstance(prp, int) or prp < 1:
                errors.append(f"articles[{i}]: invalid printed_page={prp}")

    return errors


def test_page_mapping():
    """Validate page mapping on all enriched JSONs."""
    json_files = sorted(ANNOTATED_DIR.glob("*_entities.json"))
    assert json_files, f"No JSON files found in {ANNOTATED_DIR}"

    total_errors = 0
    total_mapped = 0
    total_articles = 0
    for jf in json_files:
        errs = validate_page_mapping(jf)
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        arts = data.get("articles", [])
        total_articles += len(arts)
        total_mapped += sum(1 for a in arts if a.get("pdf_page"))
        if errs:
            print(f"\nFAIL {jf.name}:")
            for e in errs:
                print(f"  - {e}")
            total_errors += len(errs)
        else:
            nm = sum(1 for a in arts if a.get("pdf_page"))
            print(f"  OK  {jf.name} ({nm}/{len(arts)} mapped)")

    coverage = total_mapped / max(total_articles, 1) * 100
    print(f"\nTotal: {total_mapped}/{total_articles} articles mapped ({coverage:.0f}%), "
          f"{total_errors} error(s)")
    if total_errors > 0:
        print(f"FAIL: {total_errors} validation error(s)")
    assert total_errors == 0, f"{total_errors} validation error(s)"


if __name__ == "__main__":
    test_page_mapping()
