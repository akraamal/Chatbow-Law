"""
test_cese_annexes_not_decret.py
--------------------------------
Régression : les annexes de consultation du CESE (listes de membres,
acteurs auditionnés, résultats de sondages « Ouchariko »…) publiées en
queue des BO ne sont PAS des décrets.

Avant le correctif, la ligne « return 'DECRET'  # fallback » de
_classify_instrument_type (scripts/enrich_json_with_pages.py) étiquetait
DECRET tout contenu sans mot-clé d'instrument légal — audit 2026-08 :
annexes CESE taguées « décret » dans BO_7492_Fr (instr_32/33/34),
BO_7496_Fr, BO_7470_Fr, BO_7522_Fr, BO_7488-bis (règlements internes),
BO_6718_Fr (annexes de médicaments) et BO_7522_Fr (bouquets TV).
Depuis le correctif, le fallback renvoie None (type inconnu).

Usage:
    python -m pytest tests/test_cese_annexes_not_decret.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Les préambules réels des annexes CESE de BO_7492_Fr (décrets virtuels
# sans article, première_article_idx = len(articles)) — gelés tels quels.
_CESE_ANNEX_PREAMBLES = [
    "Annexe n°1 : Liste des membres de la Commission permanente des\n"
    "Affaires Sociales et de la Solidarité\nExperts ayant accompagné la Commission",
    "Annexe n° 2 :  Liste des acteurs auditionnés",
    "Annexe 3 : Résultats des consultations lancées sur la plateforme\n"
    "« Ouchariko » et sur les réseaux sociaux",
]

# Autres annexes réelles du corpus dont le CORPS contient des mots-clés
# juridiques en apparence (« économie circulaire » → CIRCULAIRE,
# « coordonner » → DECRET via l'ordonnancement, « décision » dans un
# texte de consultation) : le préambule commençant par « Annexe », le
# scan des mots-clés ne doit PAS avoir lieu.
_CESE_ANNEX_PREAMBLES_WITH_KEYWORDS = [
    # BO_7496_Fr instr_14 — « Annexe 5 - Benchmark international », texte
    # sur l'économie circulaire (faux positif CIRCULAIRE avant le fix).
    "Annexe 5 - Benchmark international : bonnes pratiques et modèles\n"
    "inspirants\nPlusieurs pays ont mis en place des stratégies innovantes\n"
    "Loi anti-gaspillage pour une économie circulaire (2020)",
    # BO_7470_Fr instr_25 — « Annexe 3 : Résultats de la consultation
    # citoyenne » (faux positif DECISION avant le fix).
    "Annexe 3 : Résultats de la consultation citoyenne\n"
    "Au total, 1 501 réponses au sondage en question ont été recueillies.",
    # BO_7470_Fr instr_26 — « Annexe 4 : Encadrés et tableaux », contient
    # « coordonner » (faux positif DECRET avant le fix).
    "Annexe 4 : Encadrés et tableaux :\n"
    "Son rôle est de coordonner les efforts de recherche",
    # BO_7522_Fr instr_37 — « Annexe 4 – Enseignements de la mission du
    # CESE dans la région Souss-Massa » (faux positif DECISION avant le fix).
    "Annexe 4 – Enseignements de la mission du CESE dans la région\n"
    "Souss-Massa (Agadir, 14-15 novembre 2025)",
]


def _classify(preamble: str):
    from enrich_json_with_pages import _classify_instrument_type
    return _classify_instrument_type([], preamble)


def test_cese_annex_preambles_not_decret():
    """Aucune annexe CESE ne doit être classée DECRET (ni aucun type
    d'instrument connu : ce sont des pièces de consultation, pas des
    textes juridiques)."""
    for preamble in _CESE_ANNEX_PREAMBLES:
        itype = _classify(preamble)
        assert itype is None, (
            f"préambule d'annexe CESE classé {itype!r} : {preamble[:60]!r}"
        )


def test_annex_preambles_with_legal_keywords_not_typed():
    """Les annexes dont le CORPS contient des mots-clés juridiques en
    apparence (économie circulaire, coordonner, décision…) ne doivent
    pas être typées : le préambule commençant par « Annexe » court-circuite
    le scan des mots-clés (audit 2026-08 : BO_7496/BO_7470/BO_7522)."""
    for preamble in _CESE_ANNEX_PREAMBLES_WITH_KEYWORDS:
        itype = _classify(preamble)
        assert itype is None, (
            f"annexe à mots-clés classée {itype!r} : {preamble[:60]!r}"
        )


def test_real_instrument_types_still_detected():
    """Non-régression : les vrais instruments (décret/arrêté/avis) gardent
    leur type — le correctif ne touche que le fallback."""
    from enrich_json_with_pages import _classify_instrument_type

    cases = {
        "Décret n° 2-25-632 du 20 kaada 1447 (8 mai 2026) fixant les "
        "rémunérations pour services rendus par le ministère du tourisme": "DECRET",
        "Arrêté du ministre de l'agriculture, de la pêche maritime, du "
        "développement rural et des eaux et forêts n° 198-26 du "
        "6 chaabane 1447 (26 janvier 2026) portant agrément de la "
        "société « AGROSSAR »": "ARRETE",
        "Avis\nDu Conseil économique, Social et Environnemental\n"
        "RECONNAITRE ET ORGANISER L'économie DU SOIN A AUTRUI AU MAROC": "AVIS",
    }
    for preamble, expected in cases.items():
        assert _classify_instrument_type([], preamble) == expected, preamble[:60]


ANNOTATED_7492 = Path("data/annotated/fr_BO_7492_Fr_entities.json")


@pytest.mark.skipif(
    not ANNOTATED_7492.exists(),
    reason="fr_BO_7492_Fr_entities.json absent (données gitignorées)",
)
def test_bo7492_annotated_annexes_not_decret():
    """Sur le JSON annoté réel de BO_7492_Fr : aucun instrument dont le
    préambule commence par « Annexe » ne doit porter instrument_type
    DECRET après regroupement (_group_into_instruments)."""
    from enrich_json_with_pages import _group_into_instruments

    data = json.loads(ANNOTATED_7492.read_text(encoding="utf-8"))
    instruments = _group_into_instruments(
        data["articles"],
        preamble_text=data.get("preamble_text", ""),
        decrees=data.get("decrees"),
    )

    annexes = [i for i in instruments
               if (i.get("_preamble") or "").lstrip().lower().startswith("annexe")]
    assert annexes, "aucune annexe CESE trouvée dans le fixture BO_7492_Fr"
    for instr in annexes:
        assert instr.get("instrument_type") != "DECRET", (
            f"{instr.get('instrument_id')} taggé DECRET : "
            f"{(instr.get('_preamble') or '')[:60]!r}"
        )


def test_all_annotated_fr_annexes_untyped():
    """Corpus entier : aucun instrument FR dont le préambule commence par
    « Annexe » ne doit porter un type d'instrument (ni DECRET ni aucun
    autre — audit 2026-08 : 44 annexes sur les fichiers annotés FR, dont
    4 encore typées CIRCULAIRE/DECISION/DECRET avant le fix du scan des
    mots-clés : BO_7496 instr_14, BO_7470 instr_25/26, BO_7522 instr_37)."""
    import glob

    from enrich_json_with_pages import _group_into_instruments

    files = sorted(glob.glob("data/annotated/fr_*_entities.json"))
    if not files:
        pytest.skip("aucun fichier annoté FR (données gitignorées)")

    checked = 0
    for path in files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        instruments = _group_into_instruments(
            data["articles"],
            preamble_text=data.get("preamble_text", ""),
            decrees=data.get("decrees"),
        )
        for instr in instruments:
            pre = instr.get("_preamble") or ""
            if not pre.lstrip().lower().startswith("annexe"):
                continue
            checked += 1
            itype = instr.get("instrument_type")
            assert itype is None, (
                f"{Path(path).name} {instr.get('instrument_id')} type {itype!r} : "
                f"{pre[:60]!r}"
            )
    assert checked > 0, "aucune annexe trouvée dans le corpus annoté FR"