"""
test_citation_verifier.py
-------------------------------
Tests unitaires de la vérification mécanique des citations
(src/rag/citation_verifier.py) : correspondance exacte, normalisée,
OCR-aware, suppression des citations invérifiables, parsing du bloc
[[CITATIONS]].

Usage :
    python -m pytest tests/test_citation_verifier.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.citation_verifier import (
    normalize_and_find,
    parse_citations,
    verify_citations,
)

FR_CHUNK = (
    "Le présent décret fixe les règles de licences des télécommunications.\n"
    "« L'article 3 de la loi n° 24-96 est abrogé et remplacé par les "
    "dispositions suivantes ».\nFait à Rabat, le 14 chaoual 1447 "
    "(2 avril 2026)."
)

AR_CHUNK = (
    "المادة الأولى: يُحدث بواسطة هذا الدفتر الجديد نظام للإعفاء من أداء وعائدات "
    "الجمارك في إطار الصناعات التقليدية ٥٠ بالمئة"
)


def test_exact_match_returns_raw_offset():
    quote = "Fait à Rabat, le 14 chaoual 1447"
    hit = normalize_and_find(FR_CHUNK, quote, lang="fr")
    assert hit["char_start"] is not None and hit["exact"] is True
    assert FR_CHUNK[hit["char_start"]:hit["char_end"]] == quote


def test_line_break_and_case_difference_normalized():
    quote = "LES RÈGLES DE LICENCES des\n\n télécommunications"
    hit = normalize_and_find(FR_CHUNK, quote, lang="fr")
    assert hit["char_start"] is not None
    assert hit["exact"] is False and hit["normalized"] is True


def test_interleaved_guillemet_is_tolerated():
    # Artefact de colonnes PDF : « en « gros » — guillemet intercalé
    chunk = "Les frais et marges fixés en « gros et au détail comme suit"
    quote = "fixés en gros et au détail"
    hit = normalize_and_find(chunk, quote, lang="fr")
    assert hit["char_start"] is not None


def test_paraphrase_is_dropped():
    quote = "le décret précise que tout est modifié"
    hit = normalize_and_find(FR_CHUNK, quote, lang="fr")
    assert hit["char_start"] is None


def test_absent_quote_is_dropped():
    quote = "Ce passage n'existe nulle part dans le document"
    hit = normalize_and_find(FR_CHUNK, quote, lang="fr")
    assert hit["char_start"] is None


def test_short_quote_rejected():
    hit = normalize_and_find(FR_CHUNK, "Article 3.", lang="fr")
    assert hit["char_start"] is None


def test_arabic_tashkeel_alef_digit_normalized():
    # tachkeel (ُ ّ), alef avec hamza (أ), chiffre arabe ٥ -> normalisés
    chunk = "يُحدَّث النظامِ للإعفاء من الجِمارك ٥٥ بالمئة"
    quote = "يحدث النظام للإعفاء من الجمارك 55 بالمئة"
    hit = normalize_and_find(chunk, quote, lang="ar")
    assert hit["char_start"] is not None
    assert hit["normalized"] is True


def test_arabic_quote_present():
    quote = "وعائدات الجمارك في إطار الصناعات التقليدية ٥٠"
    hit = normalize_and_find(AR_CHUNK, quote, lang="ar")
    assert hit["char_start"] is not None


def test_arabic_wrong_word_dropped():
    hit = normalize_and_find(AR_CHUNK, "وزارة الداخلية بالمغرب العاصمة", lang="ar")
    assert hit["char_start"] is None


def test_arabic_diacritics_vs_plain():
    chunk = "المَواد الجَديدة تُحدد الحقوق"
    quote = "المواد الجديدة تحدد الحقوق"
    hit = normalize_and_find(chunk, quote, lang="ar")
    assert hit["char_start"] is not None


def test_parse_citations_block():
    answer = (
        "La réponse utile.\n\n"
        "[[CITATIONS]]\n"
        "«Fait à Rabat, le 14 chaoual 1447» [Source 1]\n"
        "«L'entreprise est mise en demeure» [Source 2]\n"
        "[[END]]"
    )
    clean, spans = parse_citations(answer)
    assert "[[" not in clean and "[[END]]" not in clean
    assert "La réponse utile" in clean
    assert spans == [
        {"quote": "Fait à Rabat, le 14 chaoual 1447", "source": 1},
        {"quote": "L'entreprise est mise en demeure", "source": 2},
    ]


def test_verify_all_span_and_quote():
    quoted = "Fait à Rabat, le 14 chaoual 1447"
    bad = "Ce passage n'existe pas dans la source"
    spans = [
        {"quote": quoted, "source": 1},
        {"quote": bad, "source": 1},
        {"quote": quoted, "source": 99},
    ]
    chunks = [{
        "text": FR_CHUNK, "article_id": "art_42", "doc_id": "BO_1", "lang": "fr",
    }]
    verified, stats = verify_citations(spans, chunks)
    assert len(verified) == 1 and verified[0]["quote"] == quoted
    assert verified[0]["chunk_id"] == "art_42"
    assert verified[0]["source"] == 1
    assert stats == {"claimed": 3, "verified": 1, "failed": 2}


def test_verify_normalized_quote_uses_raw_offsets():
    span = {"quote": "LE PRÉSENT décret\n fixe", "source": 1}
    chunks = [{"text": FR_CHUNK, "lang": "fr"}]
    verified, _ = verify_citations([span], chunks)
    assert len(verified) == 1
    assert verified[0]["char_start"] is not None


def test_verify_out_of_range_source_dropped():
    chunks = [{"text": FR_CHUNK, "lang": "fr"}]
    verified, stats = verify_citations([{"quote": "Fait à Rabat", "source": 9}], chunks)
    assert verified == [] and stats["failed"] == 1


def test_parse_citations_missing_block():
    clean, spans = parse_citations("Réponse sans citation.")
    assert spans == [] and clean == "Réponse sans citation."


def test_parse_nested_guillemets_keeps_full_quote():
    # Le LLM a cité un passage contenant lui-même des guillemets « ... » :
    # le span capturé doit être le passage EXTERNE complet, pas le fragment
    # interne.
    answer = (
        "Réponse.\n\n"
        "[[CITATIONS]]\n"
        "« La société « TAIBA SEAFOOD Sarl » est autorisée à créer "
        "et exploiter une ferme dénommée « Taiba Seafood » » [Source 1]\n"
        "[[END]]"
    )
    clean, spans = parse_citations(answer)
    assert len(spans) == 1
    assert spans[0]["source"] == 1
    assert spans[0]["quote"].startswith("La société « TAIBA SEAFOOD Sarl »")
    assert spans[0]["quote"].endswith("dénommée « Taiba Seafood »")
    assert "[[" not in clean


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))