"""
test_instrument_detection.py
-----------------------------
Validate instrument boundary detection on known BO files.

Checks:
- All instruments have valid type
- instrument_id values are unique per document
- Article numbering resets correctly at instrument boundaries
- No instrument has zero articles

Usage:
    python -m pytest tests/test_instrument_detection.py -v
    python tests/test_instrument_detection.py
"""

import json
import sys
from pathlib import Path

ANNOTATED_DIR = Path("data/annotated")

VALID_TYPES = {"DECRET", "ARRETE", "ARRETE_CONJOINT", "DAHIR", "CIRCULAIRE", "DECISION"}


def validate_instruments(path: Path) -> list[str]:
    errors = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    instruments = data.get("instruments", [])
    if not instruments:
        # It's OK for a file to have no instruments (e.g., not yet enriched)
        return errors

    # Check instrument_id uniqueness
    ids = [i.get("instrument_id") for i in instruments]
    if len(ids) != len(set(ids)):
        errors.append("duplicate instrument_id values")

    # Check each instrument
    for i, instr in enumerate(instruments):
        instr_id = instr.get("instrument_id", f"instr_{i+1}")

        # Type validation
        itype = instr.get("instrument_type", "")
        if itype not in VALID_TYPES:
            errors.append(f"{instr_id}: invalid type '{itype}' (expected one of {VALID_TYPES})")

        # Reference may be absent (Arabic BOs often lack n° patterns)
        ref = instr.get("reference")
        if not ref:
            pass  # acceptable — Arabic BOs rarely have extractable n°

        # Articles
        arts = instr.get("articles", [])
        if not arts:
            errors.append(f"{instr_id}: zero articles")
            continue

        # Check article numbers start with a reset keyword
        first_num = arts[0].get("number", "").strip().upper()
        reset_keywords = {"1", "PREMIER", "1ER", "UNIQUE", "PREMIÈRE"}
        if first_num not in reset_keywords and not first_num.startswith("PREMIER"):
            errors.append(f"{instr_id}: first article number is '{first_num}', "
                          f"expected a reset keyword {reset_keywords}")

    return errors


def test_instrument_detection():
    """Validate instrument boundaries on all enriched JSONs."""
    json_files = sorted(ANNOTATED_DIR.glob("*_entities.json"))
    assert json_files, f"No JSON files found in {ANNOTATED_DIR}"

    total_errors = 0
    total_instruments = 0
    for jf in json_files:
        errs = validate_instruments(jf)
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        n_inst = len(data.get("instruments", []))
        total_instruments += n_inst
        if errs:
            safe_name = jf.name.encode("ascii", errors="replace").decode()
            print(f"\nFAIL {safe_name} ({n_inst} instruments):")
            for e in errs:
                safe_err = e.encode("ascii", errors="replace").decode()
                print(f"  - {safe_err}")
            total_errors += len(errs)
        else:
            print(f"  OK  {jf.name} ({n_inst} instruments)")

    print(f"\nTotal: {total_instruments} instruments across {len(json_files)} files, "
          f"{total_errors} error(s)")
    assert total_errors == 0, f"{total_errors} validation error(s)"


if __name__ == "__main__":
    test_instrument_detection()
