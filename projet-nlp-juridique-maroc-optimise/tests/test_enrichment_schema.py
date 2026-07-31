"""
test_enrichment_schema.py
--------------------------
Validate the schema of enriched JSON files: required fields, data types,
and consistency constraints.

Usage:
    python -m pytest tests/test_enrichment_schema.py -v
    python tests/test_enrichment_schema.py
"""

import json
import sys
from pathlib import Path

ANNOTATED_DIR = Path("data/annotated")

REQUIRED_ARTICLE_FIELDS = {
    "number": str,
    "raw_header": str,
    "text": str,
}

OPTIONAL_ARTICLE_FIELDS = {
    "pdf_page": (int, type(None)),
    "printed_page": (int, type(None)),
    "entities": list,
    "dates": list,
    "citations": list,
    "persons": list,
    "organizations": list,
    "extracted_tables": list,
}

REQUIRED_INSTRUMENT_FIELDS = {
    "instrument_type": str,
    "reference": (str, type(None)),
    "articles": list,
    "instrument_id": str,
}

REQUIRED_DOC_FIELDS = {
    "source": str,
    "doc_id": str,
    "articles": list,
    "n_articles": int,
}


def _check_type(value, expected) -> list[str]:
    errors = []
    if isinstance(expected, tuple):
        if not isinstance(value, expected):
            errors.append(f"expected one of {expected}, got {type(value).__name__}")
    elif not isinstance(value, expected):
        errors.append(f"expected {expected.__name__}, got {type(value).__name__}")
    return errors


def validate_enriched_json(path: Path) -> list[str]:
    errors = []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [f"JSON parse error: {e}"]

    # Document-level fields
    for field, ftype in REQUIRED_DOC_FIELDS.items():
        if field not in data:
            errors.append(f"missing doc field: {field}")
        else:
            errors += [f"{field}: {e}" for e in _check_type(data[field], ftype)]

    # Articles
    for i, art in enumerate(data.get("articles", [])):
        for field, ftype in REQUIRED_ARTICLE_FIELDS.items():
            if field not in art:
                errors.append(f"articles[{i}] missing field: {field}")
            else:
                errs = _check_type(art[field], ftype)
                errors += [f"articles[{i}].{field}: {e}" for e in errs]

    # Instrument structure if present
    instruments = data.get("instruments", [])
    if instruments:
        for i, instr in enumerate(instruments):
            for field, ftype in REQUIRED_INSTRUMENT_FIELDS.items():
                if field not in instr:
                    errors.append(f"instruments[{i}] missing field: {field}")
                else:
                    errs = _check_type(instr[field], ftype)
                    errors += [f"instruments[{i}].{field}: {e}" for e in errs]

            # Check that instrument articles have page info (optional)
            nm = sum(1 for a in instr.get("articles", []) if a.get("pdf_page"))
            if nm == 0 and len(instr.get("articles", [])) > 0:
                pass  # acceptable — page backfill may not have run

        # Check instrument_id uniqueness
        ids = [i.get("instrument_id") for i in instruments]
        if len(ids) != len(set(ids)):
            errors.append("duplicate instrument_id values")

    # Page mapping consistency: total_pdf_pages if backfill ran
    tpp = data.get("total_pdf_pages")
    if tpp is not None:
        max_page = max((a.get("pdf_page") or 0 for a in data.get("articles", [])), default=0)
        if max_page > tpp:
            errors.append(f"max article pdf_page ({max_page}) > total_pdf_pages ({tpp})")

    return errors


def test_all_enriched_jsons():
    """Run schema validation on every enriched JSON in data/annotated/."""
    json_files = sorted(ANNOTATED_DIR.glob("*_entities.json"))
    assert json_files, f"No JSON files found in {ANNOTATED_DIR}"

    total_errors = 0
    for jf in json_files:
        errs = validate_enriched_json(jf)
        if errs:
            print(f"\nFAIL {jf.name}:")
            for e in errs:
                print(f"  - {e}")
            total_errors += len(errs)
        else:
            print(f"  OK  {jf.name}")

    assert total_errors == 0, f"{total_errors} validation error(s) across {len(json_files)} files"


if __name__ == "__main__":
    test_all_enriched_jsons()
