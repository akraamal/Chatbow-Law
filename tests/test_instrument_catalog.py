"""
tests/test_instrument_catalog.py
-----------------------------------
Tests du catalogue d'instruments (src/search_engine/catalog.py) et de
l'aiguillage des questions agrégées (src/rag/query_routing.py).
"""
from __future__ import annotations

import pytest

from src.search_engine.catalog import build_catalog, compute_importance, search_catalog
from src.rag.query_routing import route_query


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


# --- Construction ----------------------------------------------------------

def test_catalog_is_large(catalog):
    assert len(catalog) > 300


def test_catalog_has_no_duplicate_docs(catalog):
    docs = {(e["lang"], e["bo_number"]) for e in catalog}
    assert len(docs) >= 20


def test_type_coverage(catalog):
    types = {e["type"] for e in catalog}
    assert {"Décret", "Arrêté"} <= types


def test_dahirs_present(catalog):
    dahirs = [e for e in catalog if e["type"] == "Dahir"]
    assert len(dahirs) >= 5


def test_ar_decrees_present(catalog):
    ar = [e for e in catalog if e["type"] == "Décret" and e["lang"] == "ar"]
    assert len(ar) >= 30


def test_references_extracted(catalog):
    refs = [e["reference"] for e in catalog if e["reference"]]
    assert len(refs) > len(catalog) // 2
    assert any("-" in r or "." in r for r in refs)


def test_n_articles_positive(catalog):
    assert all(e["n_articles"] >= 1 for e in catalog)


def test_importance_range(catalog):
    assert all(0 <= e["importance"] <= 100 for e in catalog)


def test_importance_grows_with_size():
    small = compute_importance({"n_articles": 1, "title": "x", "preamble": "y"}, 2000, 2026)
    big = compute_importance({"n_articles": 30, "title": "x", "preamble": "y"}, 2000, 2026)
    assert big > small


def test_importance_modifies_bonus():
    bare = compute_importance({"n_articles": 3, "title": "x", "preamble": "relatif au transport"}, 2000, 2026)
    mod = compute_importance(
        {"n_articles": 3, "title": "x", "preamble": "modifiant le décret n° 2-09-481"}, 2000, 2026
    )
    assert mod > bare


# --- Recherche --------------------------------------------------------------

def test_search_dahirs_important(catalog):
    # Même chemin que le chatbot : route → recherche filtrée par type.
    route = route_query("les dahirs les plus importants")
    hits = search_catalog(
        catalog, "les dahirs les plus importants",
        type_filter=route["type"], top_n=10,
    )
    assert hits
    assert all(h["type"] == "Dahir" for h in hits)
    imp = [h["importance"] for h in hits]
    assert imp == sorted(imp, reverse=True)


def test_search_by_reference(catalog):
    hits = search_catalog(catalog, "combien d'articles comporte le décret n° 2-25-1080 ?", top_n=3)
    assert hits
    assert hits[0]["reference"] == "2-25-1080"


def test_search_year_filter(catalog):
    route = route_query("les décrets de 2024")
    hits = search_catalog(
        catalog, "les décrets de 2024",
        type_filter=route["type"], year=route["year"], top_n=20,
    )
    if hits:
        assert all(h["year"] == 2024 for h in hits)


def test_search_returns_metadata_safe_for_citation_verifier(catalog):
    route = route_query("les dahirs les plus importants")
    hits = search_catalog(
        catalog, "les dahirs les plus importants",
        type_filter=route["type"], top_n=3,
    )
    for h in hits:
        assert h["text"] and h["lang"] and h["bo_number"] and h["doc_id"]


# --- Aiguillage --------------------------------------------------------------

ROUTING_CASES = [
    # (question, attendu_catalog, attendu_type)
    ("les dahirs les plus importants", True, "Dahir"),
    ("les dahirs", True, "Dahir"),
    ("quels sont les décrets de 2024 ?", True, "Décret"),
    ("combien d'articles comporte le décret n° 2-25-1080 ?", True, "Décret"),
    ("le décret 2.24.874 porte sur quoi ?", True, "Décret"),
    ("Quels sont les arrêtés du BO 7510 ?", True, "Arrêté"),
    ("les textes les plus importants en matière de transport", True, None),
    ("Qui délivre le permis de construire ?", False, None),
    ("numero d'articles", False, None),
    ("quel est le taux de TVA sur les produits agricoles ?", False, None),
    ("ما هي المراسيم المهمة", True, "Décret"),
    ("أهم الظهائر", True, "Dahir"),
    ("كم عدد مواد المرسوم رقم 2.24.874 ؟", True, "Décret"),
]


@pytest.mark.parametrize("query,expected_catalog,expected_type", ROUTING_CASES)
def test_route_query(query, expected_catalog, expected_type):
    route = route_query(query)
    assert route["catalog"] == expected_catalog, f"{query!r} → {route}"
    assert route["type"] == expected_type, f"{query!r} → {route}"


def test_route_query_year():
    route = route_query("quels sont les décrets de 2024 ?")
    assert route["year"] == 2024