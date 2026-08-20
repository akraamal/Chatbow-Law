"""
tests/test_query_routing_fallback.py
---------------------------------------
Tests du repli bas-coût de l'aiguillage (src/rag/query_routing.py) :
heuristiques de phrases (agrégation protégée par le garde-fou des noms de
corpus, synthèse) et similarité d'embedding contre des questions canoniques.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.rag.query_routing import route_query


# --- Heuristique d'agrégation (garde-fou : nom de corpus requis) -----------

AGG_FALLBACK_CASES = [
    # (question, phrase d'agrégation qui doit primer)
    ("quels sont les derniers documents analysés ?", "quels sont"),
    ("combien d'actes juridiques contient ce bulletin ?", "combien"),
    ("peux-tu recenser les documents publiés cette année ?", "recenser"),
    ("ما هي أهم الوثائق؟", "ما هي"),
    ("كم عدد الوثائق الموجودة في هذا الجريدة؟", "كم عدد"),
    ("عرض لي جميع الوثائق الجديدة", "جميع"),
]


@pytest.mark.parametrize("query,phrase", AGG_FALLBACK_CASES)
def test_fallback_agg_phrase(query, phrase):
    route = route_query(query)
    assert route["catalog"] is True, f"{query!r} → {route}"
    assert route["type"] is None, f"{query!r} → {route}"
    assert route["signal"] == f"fallback:agg:phrase:{phrase}", f"{query!r} → {route}"


# Questions factuelles : la phrase d'agrégation est présente mais le
# garde-fou des noms de corpus la neutralise → reste en sémantique.
GUARD_CASES = [
    "quels sont les délais de recours ?",
    "combien de jours pour déposer un recours ?",
    "qui délivre le permis de construire ?",
    "quel est le taux de TVA sur les produits agricoles ?",
    "كم من الوقت يستغرق تسليم الترخيص؟",
]


@pytest.mark.parametrize("query", GUARD_CASES)
def test_fallback_guard_keeps_semantic(query):
    route = route_query(query)
    assert route["catalog"] is False, f"{query!r} → {route}"
    assert route["scope"] is None, f"{query!r} → {route}"
    assert route["signal"] == "none", f"{query!r} → {route}"


# --- Heuristique de synthèse ------------------------------------------------

SYNTH_FALLBACK_CASES = [
    # (question, phrase de synthèse qui doit primer)
    ("quel est le propos de ce texte ?", "le propos de"),
    ("quel est le contenu de cette norme ?", "quel est le contenu"),
    ("ما هو المضمون العام لهذه الوثيقة؟", "المضمون العام"),
    ("تحدث عن مضمون هذا النص", "تحدث عن مضمون"),
]


@pytest.mark.parametrize("query,phrase", SYNTH_FALLBACK_CASES)
def test_fallback_synth_phrase(query, phrase):
    route = route_query(query)
    assert route["scope"] == "synthesis", f"{query!r} → {route}"
    assert route["catalog"] is False, f"{query!r} → {route}"
    assert route["signal"] == f"fallback:synthesis:phrase:{phrase}", f"{query!r} → {route}"


# --- Repli par similarité d'embedding ---------------------------------------

class _FakeEmbedder:
    """Déterministe : axe agrégation (x), axe synthèse (y), sinon z
    (orthogonal aux deux — ne déclenche aucun repli). Aucun des exemples
    canoniques ne doit tomber sur z (une requête « z » y scorerait 1.0) ni
    toucher les deux axes (la similarité « max » les compterait tous deux)."""

    AGG_TOKENS = ("decrets", "combien", "liste", "documents", "textes", "dernieres")
    SYNTH_TOKENS = ("resume", "parle", "difference", "compare", "propos", "contenu", "theme")

    def __init__(self):
        self.calls: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(text)
        v = np.zeros(3)
        if any(t in text.lower() for t in self.AGG_TOKENS):
            v[0] = 1.0
        if any(t in text.lower() for t in self.SYNTH_TOKENS):
            v[1] = 1.0
        if not v.any():
            v[2] = 1.0
        return v / np.linalg.norm(v)


@pytest.fixture
def fake_embedder():
    return _FakeEmbedder()


def test_fallback_agg_embedding(fake_embedder):
    # NB : « parus » serait déjà un signal d'agrégation lexical — on utilise
    # « publiés » (accents retirés par _norm → aucun signal lexical).
    route = route_query(
        "pouvez-vous m'indiquer les derniers textes publiés ?",
        embed_fn=fake_embedder.embed_query,
    )
    assert route["catalog"] is True, f"→ {route}"
    assert route["signal"] == "fallback:agg:embed:1.00", f"→ {route}"


def test_fallback_synth_embedding(fake_embedder):
    route = route_query(
        "pouvez-vous m'indiquer la difference de traitement ?",
        embed_fn=fake_embedder.embed_query,
    )
    assert route["scope"] == "synthesis", f"→ {route}"
    assert route["signal"] == "fallback:synthesis:embed:1.00", f"→ {route}"


def test_embedding_no_match_stays_semantic(fake_embedder):
    route = route_query(
        "qui délivre le permis de construire ?",
        embed_fn=fake_embedder.embed_query,
    )
    assert route["catalog"] is False and route["scope"] is None
    assert route["signal"] == "none"


def test_embedding_examples_cached(fake_embedder):
    route_query("qui délivre le permis de construire ?", embed_fn=fake_embedder.embed_query)
    first_total = len(fake_embedder.calls)
    route_query("quel est le salaire minimum ?", embed_fn=fake_embedder.embed_query)
    second_total = len(fake_embedder.calls)
    assert first_total >= 13  # 1 requête + exemples d'agrégation + de synthèse
    assert second_total == first_total + 1  # exemples mis en cache, seule la requête est ré-embeddée


def test_embedding_failure_degrades_gracefully():
    class _Boom:
        def embed_query(self, text):
            raise RuntimeError("modèle indisponible")

    route = route_query(
        "pouvez-vous m'indiquer les derniers textes publiés ?",
        embed_fn=_Boom().embed_query,
    )
    assert route["catalog"] is False
    assert route["scope"] is None
    assert route["signal"] == "none"


# --- Prééminence et intégration ---------------------------------------------

def test_phrase_takes_precedence_over_embedding(fake_embedder):
    # « combien » + « bulletin » : la phrase gagne même avec un embedder
    # qui classerait la question en agrégation.
    route = route_query(
        "combien d'actes juridiques contient ce bulletin ?",
        embed_fn=fake_embedder.embed_query,
    )
    assert route["catalog"] is True
    assert route["signal"].startswith("fallback:agg:phrase:")


def test_lexical_routes_keep_signal_none():
    route = route_query("les dahirs les plus importants")
    assert route["catalog"] is True
    assert route["signal"] == "none"
    route = route_query("quel est le taux de TVA sur les produits agricoles ?")
    assert route["catalog"] is False
    assert route["signal"] == "none"