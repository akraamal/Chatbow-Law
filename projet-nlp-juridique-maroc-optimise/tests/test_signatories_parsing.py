"""
test_signatories_parsing.py
---------------------------
Régression du parseur de signatures (priorité 4 du guide v5,
scripts/enrich_json_with_pages.py) :

  - _parse_signature_blocks : zones de signature reconstituées
    (émetteur + contreseings, colonnes parallèles, délégation, intérim,
    nom illisible, rejet des fragments d'annexe, édition arabe) ;
  - _instrument_signatories : reconstruction du bloc coupé par un saut
    de page/colonne (formule de clôture dans un article, signatures dans
    le suivant) ;
  - jeu doré (tests/signatories_golden_set.py) : les 18 cas gelés
    vérifiés à la main sur le texte source, re-diffés contre
    data/annotated à chaque exécution.

Usage:
    python -m pytest tests/test_signatories_parsing.py -v
    python tests/test_signatories_parsing.py
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.enrich_json_with_pages import (  # noqa: E402
    _instrument_signatories,
    _parse_signature_blocks,
)
from tests.signatories_golden_set import GOLDEN_SIGNATORIES  # noqa: E402

ANNOTATED_DIR = Path("data/annotated")


def _blocks(zone: str, is_ar: bool, issuer_role: str | None) -> list[dict]:
    """Le parseur ignore la formule de clôture (ancre de découpage) :
    on teste la zone de signature nue, sans « Fait à … »."""
    return _parse_signature_blocks(zone, is_ar, issuer_role)


def test_issuer_name_first_layout():
    """Layout FR standard : le nom de l'émetteur précède le bloc de
    contreseing (ex. BO_6804 décret 2-19-615) — le premier nom SANS rôle
    ni marqueur est l'émetteur (type issuer), rôle déduit du préambule."""
    zone = (
        "SAAD DINE EL OTMANI.\n"
        "Pour contreseing :\n"
        "Le ministre de l'économie et des finances,\n"
        "MOHAMED BENCHAABOUN.\n"
        "Le ministre de l'agriculture, de la pêche maritime, du\n"
        "développement rural et des eaux et forêts,\n"
        "AZIZ AKHANNOUCH.\n"
    )
    got = _blocks(zone, False, "Chef du Gouvernement")
    assert got == [
        {"role": "Chef du Gouvernement", "name": "SAAD DINE EL OTMANI",
         "type": "issuer"},
        {"role": "Le ministre de l'économie et des finances",
         "name": "MOHAMED BENCHAABOUN", "type": "contreseing"},
        {"role": "Le ministre de l'agriculture, de la pêche maritime, du "
                 "développement rural et des eaux et forêts",
         "name": "AZIZ AKHANNOUCH", "type": "contreseing"},
    ]


def test_title_case_legacy_editions():
    """Éditions 2004-2011 : noms imprimés en Title-case (ex. BO_7488-bis
    décret 2-04-554 « Driss Jettou. ») — reconnus par la branche
    Title-case de _SIGNATORY_LINE_RE."""
    zone = (
        "Driss Jettou.\n"
        "Pour contreseing :\n"
        "Le ministre des Habous et des affaires islamiques,\n"
        "AHMED TOUFIQ.\n"
    )
    got = _blocks(zone, False, "Premier Ministre")
    assert got == [
        {"role": "Premier Ministre", "name": "Driss Jettou", "type": "issuer"},
        {"role": "Le ministre des Habous et des affaires islamiques",
         "name": "AHMED TOUFIQ", "type": "contreseing"},
    ]


def test_single_signatory_no_contreseing():
    """Décret à signature unique (aucun « Pour contreseing : ») — ex.
    BO_7421 éd. arabe, mais ici variante FR : le seul nom est l'émetteur."""
    zone = "AZIZ AKHANNOUCH.\n"
    got = _blocks(zone, False, "Chef du Gouvernement")
    assert got == [
        {"role": "Chef du Gouvernement", "name": "AZIZ AKHANNOUCH",
         "type": "issuer"},
    ]


def test_delegation_block():
    """Signature par délégation : « Pour le Chef du Gouvernement et par
    délégation, » — le rôle déléguant est porté avec type 'delegated' et
    le signataire réel suit."""
    zone = (
        "Pour le Chef du Gouvernement et par délégation,\n"
        "Le ministre délégué auprès du ministre de l'intérieur,\n"
        "ABDELOUAHAD LAATI.\n"
    )
    got = _blocks(zone, False, "Chef du Gouvernement")
    assert got == [
        {"role": "Pour le Chef du Gouvernement et par délégation",
         "name": "ABDELOUAHAD LAATI", "type": "delegated"},
        {"role": "Le ministre délégué auprès du ministre de l'intérieur",
         "name": None, "type": "delegated"},
    ]


def test_interim_block():
    """Intérim : « L'intérimaire, » — type dédié, nom absent conservé en
    None s'il n'y a pas de ligne de nom."""
    zone = "L'intérimaire,\nABDELOUAHAD LAATI.\n"
    got = _blocks(zone, False, "Chef du Gouvernement")
    assert got == [
        {"role": "L'intérimaire", "name": "ABDELOUAHAD LAATI",
         "type": "interim"},
    ]


def test_missing_name_role_kept():
    """Nom illisible/absent : le rôle est conservé avec name=None (le
    fragment reste un rôle crédible, court et bien amorcé)."""
    zone = (
        "SAAD DINE EL OTMANI.\n"
        "Pour contreseing :\n"
        "Le ministre de la santé et de la protection sociale\n"
    )
    got = _blocks(zone, False, "Chef du Gouvernement")
    assert got == [
        {"role": "Chef du Gouvernement", "name": "SAAD DINE EL OTMANI",
         "type": "issuer"},
        {"role": "Le ministre de la santé et de la protection sociale",
         "name": None, "type": "contreseing"},
    ]


def test_annex_fragment_rejected():
    """Un fragment long d'annexe (« La liste des diplômes et des
    certificats… ») ne ressemble pas à un rôle : rien n'est émis."""
    zone = (
        "La liste des diplômes et des certificats délivrés par la faculté\n"
        "de médecine est arrêtée conformément aux tableaux suivants\n"
    )
    assert _blocks(zone, False, "Chef du Gouvernement") == []


def test_parallel_columns_fifo():
    """Colonnes parallèles aplaties par l'OCR (arrêté conjoint 181-26) :
    tous les rôles précèdent tous les noms — appariement FIFO."""
    zone = (
        "Le ministre de l'économie et des finances\n"
        "Le ministre de l'agriculture\n"
        "NADIA FETTAH.\n"
        "MOHAMED SADIKI.\n"
    )
    got = _blocks(zone, False, "Chef du Gouvernement")
    assert got == [
        {"role": "Le ministre de l'économie et des finances",
         "name": "NADIA FETTAH", "type": "contreseing"},
        {"role": "Le ministre de l'agriculture",
         "name": "MOHAMED SADIKI", "type": "contreseing"},
    ]


def test_arabic_edition():
    """Édition arabe : « المضاء : nom » (signature) et « وقعه بالعطف : »
    (contreseing) — le nom est porté par la ligne المضاء, pas la ligne
    de rôle qui suit."""
    zone = (
        "المضاء : عزيز اخنوش\n"
        "وقعه بالعطف :\n"
        "وزير الفلاحة والصيد البحري\n"
        "المضاء : احمد البواري\n"
    )
    got = _blocks(zone, True, "رئيس الحكومة")
    assert got == [
        {"role": "رئيس الحكومة", "name": "عزيز اخنوش", "type": "issuer"},
        {"role": "وزير الفلاحة والصيد البحري", "name": "احمد البواري",
         "type": "contreseing"},
    ]


def test_page_break_reconstruction():
    """Bloc coupé par un saut de page : la formule de clôture est dans
    l'avant-dernier article, les signatures continuent dans le dernier —
    _instrument_signatories reconstitue la zone sur les articles du
    finaux (cas gelé BO_6804 décret 2-19-615)."""
    articles = [
        {"text": "ART. 3. - Le présent décret sera publié au Bulletin "
                 "officiel. Fait à Rabat, le 16 juin 2021."},
        {"text": "SAAD DINE EL OTMANI.\n"
                 "Pour contreseing :\n"
                 "Le ministre de l'économie et des finances,\n"
                 "MOHAMED BENCHAABOUN.\n"},
    ]
    got = _instrument_signatories(articles, "LE CHEF DU GOUVERNEMENT,")
    assert got == [
        {"role": "Chef du Gouvernement", "name": "SAAD DINE EL OTMANI",
         "type": "issuer"},
        {"role": "Le ministre de l'économie et des finances",
         "name": "MOHAMED BENCHAABOUN", "type": "contreseing"},
    ]


def test_interleaved_columns_reversed_names():
    """BO_7510 arrêté conjoint 181-26 : colonnes de hauteurs différentes —
    la continuation du rôle A (« de la pêche maritime, ») est reprise
    après le bloc COMPLET du rôle B, et les noms (alignés en bas de page)
    arrivent dans l'ordre INVERSE des blocs de rôle. Vérifié sur la
    géométrie du PDF : FOUZI LEKJAA = ministre délégué chargé du budget,
    ZAKIA DRIOUICH = secrétaire d'Etat chargée de la pêche maritime."""
    zone = (
        "La secrétaire d'Etat auprès\n"
        "du ministre de l'agriculture,\n"
        "de la pêche maritime,\n"
        "du développement rural\n"
        "et des eaux et forêts, chargée\n"
        "Le ministre délégué auprès\n"
        "de la ministre de l'économie et\n"
        "des finances, chargé du budget,\n"
        "de la pêche maritime,\n"
        "FOUZI LEKJAA.\n"
        "ZAKIA DRIOUICH.\n"
    )
    got = _blocks(zone, False, "Chef du Gouvernement")
    assert got == [
        {"role": "Le ministre délégué auprès de la ministre de l'économie "
                 "et des finances, chargé du budget",
         "name": "FOUZI LEKJAA", "type": "contreseing"},
        {"role": "La secrétaire d'Etat auprès du ministre de l'agriculture, "
                 "de la pêche maritime, du développement rural et des eaux "
                 "et forêts, chargée de la pêche maritime",
         "name": "ZAKIA DRIOUICH", "type": "contreseing"},
    ]


def test_no_formula_no_signatories():
    """Pas de formule de clôture dans le texte des articles : aucune zone
    de signature (fragment d'annexe ou instrument tronqué) — pas de
    signataires fantômes."""
    articles = [
        {"text": "ART. 1. - Les dispositions du présent arrêté s'appliquent "
                 "aux établissements publics."},
        {"text": "ART. 2. - Les annexes du présent arrêté sont prises en "
                 "compte."},
    ]
    assert _instrument_signatories(articles, "LE CHEF DU GOUVERNEMENT,") == []


def _golden_diff():
    """Même diff que validate_content_integrity.py : chaque entrée du jeu
    doré doit correspondre EXACTEMENT à data/annotated/<file> ->
    instruments[<instrument_id>].signatories. Retourne la liste des
    entrées en échec (vide = conforme)."""
    failures = []
    for entry in GOLDEN_SIGNATORIES:
        path = ANNOTATED_DIR / entry["file"]
        if not path.exists():
            failures.append(
                (entry, f"fichier absent du dépôt (données gitignorées) : "
                        f"{path.name}"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        found = None
        for instr in data.get("instruments", []):
            if instr.get("instrument_id") == entry["instrument_id"]:
                found = instr.get("signatories")
                break
        if found != entry["expected"]:
            failures.append((entry, f"instr {entry['instrument_id']}: "
                                    f"{found!r} != {entry['expected']!r}"))
    return failures


@pytest.mark.parametrize("entry", GOLDEN_SIGNATORIES,
                         ids=[f"{e['file'][:20]}::{e['instrument_id']}"
                              for e in GOLDEN_SIGNATORIES])
def test_golden_entry(entry):
    """Chaque cas gelé du jeu doré doit correspondre à l'annotation."""
    for e, msg in _golden_diff():
        if e is entry:
            pytest.fail(msg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
