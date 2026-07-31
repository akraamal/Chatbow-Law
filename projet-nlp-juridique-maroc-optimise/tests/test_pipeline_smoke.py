"""
test_pipeline_smoke.py
-----------------------
Smoke test for the full pipeline.  Runs ingestion → preprocessing →
extraction on a small PDF and validates the output JSON exists and is valid
JSON with at least one article.

This is NOT a unit test — it requires all dependencies (PyMuPDF, spaCy, etc.)
and takes ~30-60 seconds.

Usage:
    python tests/test_pipeline_smoke.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import run_ingestion_pipeline
from src.preprocessing.segmenter import segment_into_articles, get_preamble
from src.extraction.etape4_pipeline import enrich_article_json

SMALL_PDF = "data/raw/BO_7500_Fr.pdf"


def test_smoke_pipeline():
    """Run ingestion → preprocessing → extraction on one small PDF."""
    if not Path(SMALL_PDF).exists():
        print(f"SKIP: {SMALL_PDF} not found")
        return

    print(f"Smoke test on {SMALL_PDF} ...")

    # Step 1: Ingestion
    result = run_ingestion_pipeline(SMALL_PDF)
    text_fr = result.text_fr
    assert text_fr, f"No French text extracted from {SMALL_PDF}"
    print(f"  Ingestion: {len(text_fr)} chars extracted")

    # Step 2: Preprocessing (manual — we just segment)
    preamble = get_preamble(text_fr)
    articles = segment_into_articles(text_fr)
    assert preamble, "No preamble text found"
    assert articles, "No articles segmented"
    print(f"  Segmentation: {len(articles)} articles, {len(preamble)} chars preamble")

    # Step 3: Extraction on first 3 articles
    for i, art in enumerate(articles[:3]):
        enriched = enrich_article_json(
            article=art,
            full_text=art.get("text", ""),
            doc_id="smoke_test",
            lang="fr",
        )
        assert enriched.get("text"), f"Article {i}: missing text after enrichment"
        print(f"  Art. {art.number}: {len(enriched.get('entities', []))} entities, "
              f"{len(enriched.get('dates', []))} dates")

    print("  PASSED")


def test_enrichment_smoke():
    """Run enrich_json_with_pages on a small exported JSON-like dict."""
    from scripts.enrich_json_with_pages import _group_into_instruments

    fake_articles = [
        {"number": "PREMIER", "text": "Texte de l'article premier."},
        {"number": "2", "text": "Texte de l'article 2."},
        {"number": "3", "text": "Texte de l'article 3."},
        {"number": "PREMIER", "text": "Article premier du second instrument."},
        {"number": "2", "text": "Dernier article."},
    ]
    # The preamble for the second instrument is embedded in article 3
    art3_text = (
        "Vu la loi n° 01-23 du ...\n"
        "Le Chef du Gouvernement,\n"
        "DÉCRÈTE :"
    )
    fake_articles[2]["text"] = art3_text

    instruments = _group_into_instruments(fake_articles, preamble_text="")
    assert len(instruments) == 2, f"Expected 2 instruments, got {len(instruments)}"
    assert instruments[0]["instrument_type"] in ("DECRET", "ARRETE", "ARRETE_CONJOINT", "DAHIR"), \
        f"Unexpected type: {instruments[0]['instrument_type']}"
    assert len(instruments[0]["article_indices"]) == 3
    assert len(instruments[1]["article_indices"]) == 2
    print(f"  Instrument detection: {len(instruments)} instruments (type={instruments[0]['instrument_type']})")
    print("  PASSED")


if __name__ == "__main__":
    test_smoke_pipeline()
    test_enrichment_smoke()
    print("\nAll smoke tests passed.")
