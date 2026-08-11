"""
tests/test_trailing_citation_ghost.py
--------------------------------------
Régression (BO_6758) : une citation bibliographique en minuscule dans la
section finale du document (Avis du CESE, note de bas de page « dahir
n° 1-16-115 du 6 kaada 1437 … portant promulgation de la loi n° 01-16 »)
ne doit PAS devenir un instrument fantôme (n_articles = 0) dans
get_per_decree_preamble_map.

Le scan de fin de document exige désormais une majuscule initiale pour les
titres français ; les vrais instruments de queue (Décision, Avis, Annexe)
restent détectés.

Au niveau de l'étape 4 (enrich_json_with_pages), la même citation ne doit
pas faire classifier l'Avis de queue en « DAHIR n° 1-16-115 » :
_classify_instrument_type lit l'en-tête en tête de préambule (AVIS) et
_extract_reference renvoie None pour un Avis sans numéro propre.

Usage:
    python -m pytest tests/test_trailing_citation_ghost.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _sample_document() -> str:
    return (
        "TEXTES GENERAUX\n"
        "Dahir n° 1-19-20 du 21 joumada II 1440 (27 février 2019) portant\n"
        "promulgation de la loi n° 20-19 approuvée par la Chambre des\n"
        "représentants et la Chambre des conseillers le 5 février 2019.\n"
        "Article premier\n"
        "Est promulguée la loi n° 20-19.\n"
        "Article unique\n"
        "La présente loi sera publiée au Bulletin officiel.\n"
        "AVIS DU CONSEIL ECONOMIQUE, SOCIAL ET ENVIRONNEMENTAL\n"
        "Le conseil a examiné la convention 143 de l'OIT.\n"
        "6 Bulletin Officiel n° 6493 du 18 kaada 1437 (22 août 2016),\n"
        "dahir n° 1-16-115 du 6 kaada 1437 (10 août 2016) portant promulgation\n"
        "de la loi n° 01-16 portant approbation de la convention n° 143 sur\n"
        "les travailleurs migrants (dispositions complémentaires) 1975.\n"
        "la couleur, le sexe, la religion, l'opinion politique.\n"
    )


def _decrees(text: str):
    from src.preprocessing.segmenter import get_per_decree_preamble_map
    return get_per_decree_preamble_map(text, lang="fr")


def test_footnote_citation_does_not_create_ghost_instrument():
    """« dahir n° 1-16-115 … » (citation en minuscule dans l'Avis de queue)
    ne doit produire aucun décret/instrument fantôme : la citation reste du
    simple texte (ici dans le préambule de l'Avis), pas un titre."""
    decrees = _decrees(_sample_document())
    assert decrees, "aucun instrument détecté sur l'échantillon"

    ghost_titles = [d for d in decrees
                    if (d.get("title") or "").strip().lower() == "dahir"
                    and (d.get("title") or "").strip() != "Dahir"]
    assert not ghost_titles, f"citation de note de bas de page devenue instrument : {ghost_titles!r}"
    assert len(decrees) == 2, f"instruments attendus : Dahir + Avis, obtenus : {[d['title'] for d in decrees]}"


def test_trailing_uppercase_title_still_detected():
    """Un vrai titre de queue en majuscule (Avis, Décision, Annexe) reste un
    instrument virtuel (first_article_idx = nombre d'articles)."""
    decrees = _decrees(_sample_document())
    # 2 articles dans l'échantillon → instruments de queue à l'index 2
    avis = [d for d in decrees if d.get("first_article_idx") == 2]
    assert avis, "Avis de queue non détecté comme instrument virtuel"
    assert "AVIS" in avis[0]["title"].upper()
    assert decrees[0]["first_article_idx"] == 0


def test_real_body_titles_unaffected():
    """Le dahir de promulgation en tête de document reste la première
    frontière (non-régression du scan entre articles)."""
    decrees = _decrees(_sample_document())
    assert "1-19-20" in (decrees[0].get("preamble", "") or "")


# ── Étape 4 (enrich_json_with_pages) : classification + référence ──

_AVIS_PREAMBLE = (
    "Avis\n"
    "du Conseil économique, Social et Environnemental\n"
    "Migration et marché du travail\n"
    "conformément à l'article 6 de la loi organique n°128-12\n"
    "relative à son organisation et à son fonctionnement, le Conseil\n"
    "économique, Social et Environnemental (CESE) s'est autosaisi\n"
    "afin de préparer un rapport sur la migration.\n"
    "Le conseil a examiné la convention 143 de l'OIT.\n"
    "6 Bulletin Officiel n° 6493 du 18 kaada 1437 (22 août 2016),\n"
    "dahir n° 1-16-115 du 6 kaada 1437 (10 août 2016) portant promulgation\n"
    "de la loi n° 01-16 portant approbation de la convention n° 143 sur\n"
    "les travailleurs migrants (dispositions complémentaires) 1975.\n"
    "la couleur, le sexe, la religion, l'opinion politique.\n"
)


def test_avis_classified_avis_not_dahir_from_footnote():
    """La note de bas de page « dahir n° 1-16-115 … » (minuscules, ligne
    précédente finissant par une virgule) ne doit pas reclassifier l'Avis
    de queue en DAHIR."""
    from enrich_json_with_pages import _classify_instrument_type
    assert _classify_instrument_type([], _AVIS_PREAMBLE) == "AVIS"


def test_avis_reference_is_none():
    """Un Avis sans numéro propre n'absorbe pas la référence croisée de sa
    note de bas de page (« 1-16-115 ») ni la « loi organique n°128-12 »."""
    from enrich_json_with_pages import _extract_reference
    assert _extract_reference(_AVIS_PREAMBLE, "AVIS") is None


def test_decision_heading_wins_over_footnote_citation():
    """Non-régression : avec une vraie en-tête de Décision, la citation
    « dahir n° 1-16-115 » (position de continuation) est ignorée et le
    numéro propre « 79 » est extrait."""
    from enrich_json_with_pages import _classify_instrument_type, _extract_reference
    heading = (
        "Décision du Wali de Bank Al-Maghrib n° 79 du 22 chaoual 1440\n"
        "(27 février 2019) portant octroi d'un agrément à la société\n"
        "« Centre monétique interbancaire ».\n"
        "6 Bulletin Officiel n° 6493 du 18 kaada 1437 (22 août 2016),\n"
        "dahir n° 1-16-115 du 6 kaada 1437 (10 août 2016) portant promulgation\n"
        "de la loi n° 01-16.\n"
    )
    assert _classify_instrument_type([], heading) == "DECISION"
    assert _extract_reference(heading, "DECISION") == "79"


def test_dahir_heading_still_extracted():
    """Non-régression : une en-tête de dahir reste DAHIR avec sa référence."""
    from enrich_json_with_pages import _classify_instrument_type, _extract_reference
    heading = (
        "Dahir n° 1-19-19 du 21 joumada II 1440 (27 février 2019) portant\n"
        "promulgation de la loi n° 20-19 approuvée par la Chambre des\n"
        "représentants et la Chambre des conseillers.\n"
    )
    assert _classify_instrument_type([], heading) == "DAHIR"
    assert _extract_reference(heading, "DAHIR") == "1-19-19"
