"""
test_keyword_counter.py
-----------------------
Régression pour adli_v2.keyword_counter : comptage déterministe, aucun
modèle.  FR : borne de mot + pluriel et casse insensibles ; AR : tokens
normalisés (préfixe d'article et diacritiques ignorés, multi-mots en
suite consécutive) ; comptages bruts + normalisés par 1 000 mots.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adli_v2.keyword_counter import (
    count_by_category,
    count_keywords,
    count_terms,
    load_keywords,
)


def test_load_keywords_static_config():
    fr = load_keywords("fr")
    ar = load_keywords("ar")
    assert set(fr) == set(ar) == {
        "Fiscal", "Social", "Administratif", "Civil",
        "Pénal", "Commercial", "Environnement", "Urbain",
    }
    assert "impôt" in fr["Fiscal"]
    assert "ضريبة" in ar["Fiscal"]


def test_fr_word_boundary_and_plural():
    text = "L'impôt sur le revenu et les impôts locaux, sans lien avec une assiette."
    counts = count_terms(text, "fr")
    assert counts["impôt"] == 2          # singulier + pluriel, pas « assiette »


def test_fr_case_insensitive():
    assert count_terms("TVA et tva et Tva.", "fr")["TVA"] == 3


def test_ar_token_counting():
    text = "ضريبة على الشركات. رسم التسجيل. ضريبة القيمة المضافة."
    counts = count_terms(text, "ar")
    assert counts["ضريبة"] == 2
    assert counts["رسم"] == 1
    assert counts["الضريبة على القيمة المضافة"] == 0  # « على » manque ici


def test_ar_no_substring_match():
    # « رسم » ne doit PAS matcher « رسمي » / « الرسمية » (ancien défaut
    # du comptage par sous-chaîne).
    counts = count_terms("رسمي ورسمية. المالية الرسمية.", "ar")
    assert counts["رسم"] == 0
    counts = count_terms("رسم التسجيل.", "ar")
    assert counts["رسم"] == 1


def test_ar_article_prefix_and_diacritics():
    # Préfixe d'article défini ignoré (بال/ال) + diacritiques ignorées.
    counts = count_terms("بالضريبة والضريبة. الضَّريبة.", "ar")
    assert counts["ضريبة"] == 3


def test_ar_multiword_terms():
    text = "الضريبة على القيمة المضافة. نص آخر. الضريبة على القيمة المضافة."
    counts = count_terms(text, "ar")
    assert counts["الضريبة على القيمة المضافة"] == 2
    # « ضريبة » seule est comptée dans le terme multi-mots (premier token).
    assert counts["ضريبة"] == 2


def test_ar_plural_forms():
    counts = count_terms("عقود الشركات والمحاكم.", "ar")
    assert counts["عقود"] == 1
    assert counts["شركات"] == 1
    assert counts["محاكم"] == 1


def test_fr_irregular_plurals():
    # Pluriels irréguliers déclarés dans keywords_fr.json (pas de règle).
    text = "des travaux dangereux, deux tribunaux, des eaux usées."
    counts = count_terms(text, "fr")
    assert counts["travaux"] == 1
    assert counts["tribunaux"] == 1
    assert counts["eaux"] == 1


def test_category_totals():
    text = "impôt sur le revenu. travail et salariés. une amende."
    per_term = count_terms(text, "fr")
    by_cat = count_by_category(per_term, "fr")
    assert by_cat["Fiscal"] >= 1
    assert by_cat["Social"] >= 1
    assert by_cat["Pénal"] >= 1


def test_count_keywords_shape():
    result = count_keywords("impôt et travail.", "fr")
    assert set(result) == {
        "per_term", "per_category",
        "per_term_normalized", "per_category_normalized", "n_words",
    }
    assert result["per_term"]["impôt"] == 1
    assert result["per_category"]["Fiscal"] >= 1
    assert result["n_words"] == 3
    assert result["per_term_normalized"]["impôt"] == round(1000.0 / 3, 2)


def test_normalized_density_comparison():
    # Même densité de « impôt » (100 % des mots) → même valeur normalisée,
    # indépendamment de la longueur du texte.
    long_ = count_keywords("impôt. impôt. impôt. impôt. impôt.", "fr")
    short = count_keywords("impôt.", "fr")
    half = count_keywords("impôt. travail.", "fr")
    assert long_["n_words"] == 5
    assert short["n_words"] == 1
    assert long_["per_term_normalized"]["impôt"] == short["per_term_normalized"]["impôt"] == 1000.0
    assert half["per_term_normalized"]["impôt"] == 500.0


def test_normalized_zero_on_empty():
    result = count_keywords("", "fr")
    assert result["n_words"] == 0
    assert result["per_term_normalized"]["impôt"] == 0.0
    assert result["per_category_normalized"]["Fiscal"] == 0.0


def test_empty_text_returns_zeros():
    result = count_keywords("", "fr")
    assert result["per_category"]["Fiscal"] == 0
    assert all(v == 0 for v in result["per_term"].values())