"""
test_instrument_schema_enrichment.py
------------------------------------
Verify that instruments in enriched JSONs carry the schema_optimal_v2
fields (type, reference_label, title, date_hijri, date_gregorian,
signatory/signatories, domain) and that the two reference instruments of
BO_7510 match the hand-curated reference file (docs/reference/
schema_optimal_v2_reel.json) on the automatically-derivable fields.

Usage:
    python -m pytest tests/test_instrument_schema_enrichment.py -v
    python tests/test_instrument_schema_enrichment.py
"""

import json
from pathlib import Path

REFERENCE = Path("docs/reference/schema_optimal_v2_reel.json")
ANNOTATED = Path("data/annotated/fr_BO_7510_Fr_entities.json")

# Champs alignables automatiquement avec la référence.
_COMPARABLE = (
    ("type", "type"),
    ("date_hijri", "date_hijri"),
    ("date_gregorian", "date_gregorian"),
)


def _enriched_instruments() -> dict:
    with ANNOTATED.open(encoding="utf-8") as f:
        data = json.load(f)
    return {i["instrument_id"]: i for i in data.get("instruments", [])}


def _reference_instruments() -> dict:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    return {i["instrument_id"]: i for i in ref.get("instruments", [])}


def test_all_instruments_have_schema_fields():
    """Tout instrument enrichi porte les champs du schéma optimal v2."""
    instrs = _enriched_instruments()
    assert instrs, "aucun instrument dans BO_7510"
    for iid, instr in instrs.items():
        for field in ("type", "reference_label", "title", "domain",
                      "date_hijri", "date_gregorian"):
            assert field in instr, f"{iid} missing {field}"
        for key in ("label", "confidence", "method", "model_id", "fallback_used"):
            assert key in instr["domain"], f"{iid}.domain missing {key}"


def test_bo7510_matches_reference_derivable_fields():
    """Les champs dérivables de BO_7510 correspondent à la référence."""
    ref_instr = _reference_instruments()
    assert ref_instr, "référence sans instruments"
    instrs = _enriched_instruments()
    for iid, expected in ref_instr.items():
        cur = instrs.get(iid)
        assert cur, f"{iid} absent du JSON annoté"
        for ref_key, pipe_key in _COMPARABLE:
            assert cur.get(pipe_key) == expected.get(ref_key), (
                f"{iid}.{pipe_key} = {cur.get(pipe_key)!r} "
                f"!= référence {expected.get(ref_key)!r}"
            )
        # Signatory (singulier) ou signatories (pluriel) cohérents avec la
        # référence.
        if expected.get("signatory"):
            assert cur.get("signatory") == expected["signatory"], (
                f"{iid}.signatory = {cur.get('signatory')!r} "
                f"!= référence {expected['signatory']!r}"
            )
        if expected.get("signatories"):
            assert cur.get("signatories") == expected["signatories"], (
                f"{iid}.signatories = {cur.get('signatories')!r} "
                f"!= référence {expected['signatories']!r}"
            )
        assert cur.get("title"), f"{iid} title vide"


if __name__ == "__main__":
    test_all_instruments_have_schema_fields()
    test_bo7510_matches_reference_derivable_fields()
    print("OK: schema enrichment")
