"""
tests/test_analyzer_chat.py
-------------------------------
Tests du chat documentaire (app/analyzer.py: _chat_answer) : règles
étendues — instruments triés par importance, recherche par référence,
domaine principal du document — sans charger le LLM ni l'index.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzer import _chat_answer


def _make_data() -> dict:
    articles = [
        {"number": "1", "text": "Article 1 relatif aux autorisations de transport."},
        {"number": "2", "text": "Article 2 relatif au transport de marchandises."},
        {"number": "3", "text": "Article 3 relatif aux normes de sécurité."},
        {"number": "4", "text": "Article 4 relatif au permis de construire."},
    ]
    instruments = [
        {"instrument_type": "Décret", "reference": "2-25-1080", "n_articles": 40, "article_indices": [0, 1]},
        {"instrument_type": "Décret", "reference": "2-24-830", "n_articles": 12, "article_indices": [2]},
        {"instrument_type": "Loi", "reference": "1-93-153", "n_articles": 30, "article_indices": [3]},
        {"instrument_type": "Loi", "reference": "1-96-124", "n_articles": 5, "article_indices": [0]},
        {"instrument_type": "Arrêté", "reference": "3-22-05", "n_articles": 2, "article_indices": [1]},
    ]
    return {
        "articles": articles,
        "instruments": instruments,
        "bo_number": "7510",
        "date_publication": "2026-05-21",
        "lang": "fr",
    }


def test_lois_les_plus_importantes():
    answer = _chat_answer(_make_data(), "lois plus importants ?")
    assert "Loi" in answer
    assert "1-93-153" in answer          # 30 articles > 5 → en premier
    assert "1-96-124" in answer
    assert "Arrêté" not in answer        # filtré au type Loi
    assert answer.index("1-93-153") < answer.index("1-96-124")


def test_liste_des_decrets():
    answer = _chat_answer(_make_data(), "liste des décrets")
    assert "Décret" in answer
    assert "2-25-1080" in answer and "2-24-830" in answer
    assert "Loi" not in answer


def test_reference_precise():
    answer = _chat_answer(_make_data(), "combien d'articles comporte le décret n° 2-25-1080 ?")
    assert "2-25-1080" in answer and "40" in answer


def test_instrument_details_par_reference():
    answer = _chat_answer(_make_data(), "détaille le décret n° 2-24-830")
    assert "2-24-830" in answer
    assert "12 articles" in answer


def test_db_reference_arabe():
    data = _make_data()
    answer = _chat_answer(data, "كم عدد مواد المرسوم رقم 2.24.830 ؟")
    assert "2-24-830" in answer and "12" in answer


def test_domaine_principal():
    answer = _chat_answer(_make_data(), "quelle domaine est primairement discuté dans ce pdf ?")
    assert "Domaine(s) principal(aux)" in answer


def test_anciennes_regles_conservees():
    data = _make_data()
    assert "4 articles" in _chat_answer(data, "combien d'articles ?")
    assert "25-1080" not in _chat_answer(data, "Article 3")           # règle article
    assert "Article 3" in _chat_answer(data, "Article 3")
    assert "n° 7510" in _chat_answer(data, "numero bo")
    assert "Recherche [mot-clé]" in _chat_answer(data, "xyzzy inconnu ??") or \
           "Je n'ai pas trouvé" in _chat_answer(data, "xyzzy inconnu ??")


def _make_entity_data() -> dict:
    """Document avec un dahir de promulgation dont le numéro ne figure QUE
    dans le préambule du décret (jamais dans le corps d'un article)."""
    return {
        "articles": [
            {"number": "unique", "text": "Est promulguée la loi n° 20-19.",
             "entities": [{"label": "DAHIR", "text": "dahir n° 1-16-115 du 6 kaada 1437 (10 août 2016)"}]},
        ],
        "preamble_entities": [
            {"label": "DAHIR", "text": "dahir n° 1-18-109 du 2 joumada I 1440 (9 janvier 2019)"},
        ],
        "decrees": [
            {"preamble": "Dahir n° 1-19-19 du 21 joumada II 1440 (27 février 2019) portant promulgation de la loi n° 20-19",
             "entities": [{"label": "DAHIR", "text": "Dahir n° 1-19-19"},
                          {"label": "LOI", "text": "loi n° 20-19"}]},
            {"preamble": "Dahir n° 1-19-20 du 21 joumada II 1440 (27 février 2019) portant promulgation de la loi n° 20-19",
             "entities": [{"label": "DAHIR", "text": "Dahir n° 1-19-20"}]},
        ],
        "bo_number": "6758",
        "date_publication": "2019-03-07",
        "lang": "fr",
    }


def test_count_entities_includes_decree_preambles():
    """Les titres de dahirs de promulgation (préambules par décret) doivent
    compter dans la répartition des entités : articles (1) + préambule du
    document (1) + préambules par décret (2) = 4 DAHIR."""
    from app.analyzer import _count_entities
    counts = _count_entities(_make_entity_data())
    assert counts.get("DAHIR") == 4, f"DAHIR compté : {counts.get('DAHIR')}"


def test_chat_entite_rule_counts_decree_preambles():
    """La règle « combien d'entités » du chat documentaire inclut les
    préambules par décret (régression : BO_6758 annonçait DAHIR : 9 alors
    que le document contient 21 dahirs)."""
    answer = _chat_answer(_make_entity_data(), "combien d'entités ?")
    assert "DAHIR : 4" in answer, answer