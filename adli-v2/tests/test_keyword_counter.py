"""
test_keyword_counter.py
-----------------------
Régression pour adli_v2.keyword_counter : comptage déterministe, aucun
modèle.  FR : borne de mot + pluriel et casse insensibles ; AR : sous-chaîne.
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


def test_ar_substring_counting():
    text = "ضريبة على الشركات. رسم التسجيل. ضريبة القيمة المضافة."
    counts = count_terms(text, "ar")
    assert counts["ضريبة"] == 2
    assert counts["رسم"] == 1


def test_category_totals():
    text = "impôt sur le revenu. travail et salariés. une amende."
    per_term = count_terms(text, "fr")
    by_cat = count_by_category(per_term, "fr")
    assert by_cat["Fiscal"] >= 1
    assert by_cat["Social"] >= 1
    assert by_cat["Pénal"] >= 1


def test_count_keywords_shape():
    result = count_keywords("impôt et travail.", "fr")
    assert set(result) == {"per_term", "per_category"}
    assert result["per_term"]["impôt"] == 1
    assert result["per_category"]["Fiscal"] >= 1


def test_empty_text_returns_zeros():
    result = count_keywords("", "fr")
    assert result["per_category"]["Fiscal"] == 0
    assert all(v == 0 for v in result["per_term"].values())