"""
test_transformer_classifier.py
-------------------------------
Vérifie le repli automatique du classifieur transformer sur les mots-clés
lorsque torch/transformers ne sont pas installés ou que le modèle fine-tuné
est absent (cas de cet environnement CPU sans Hugging Face).

En CI/Colab avec le modèle présent (models/domain_classifier), `available`
devient True et l'inférence transformer est appelée normalement.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.transformer_classifier import TransformerDomainClassifier


def test_fallback_keyword_when_model_missing(tmp_path):
    """Modèle absent -> repli mots-clés, classification fonctionnelle FR/AR."""
    clf = TransformerDomainClassifier(model_dir=str(tmp_path / "inexistant"))
    assert not clf.available
    assert "repli" in clf.status

    fr = clf.classify_text(
        "Le montant de l'impôt sur les sociétés est fixé dans la présente loi.",
        lang="fr",
    )
    ar = clf.classify_text("تحدد الضريبة على القيمة المضافة بمقتضى هذا القانون.", lang="ar")
    assert fr in {"Fiscal", "Indéterminé"}
    assert ar in {"Fiscal", "Indéterminé"}


def test_classify_document_fallback(tmp_path):
    clf = TransformerDomainClassifier(model_dir=str(tmp_path / "inexistant"))
    doc = {"articles": [{"text": "La taxe sur les transactions est recouvrée par la douane."}]}
    assert clf.classify_document(doc, lang="fr") in {"Fiscal", "Indéterminé"}


def test_bad_lang_raises(tmp_path):
    clf = TransformerDomainClassifier(model_dir=str(tmp_path / "inexistant"))
    with pytest.raises(ValueError):
        clf.classify_text("texte", lang="xx")