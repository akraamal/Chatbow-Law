"""
language_detector.py
----------------------
Détection de la langue d'un texte ou d'un bloc de texte (français / arabe).

Compatible avec le nouveau pipeline d'extraction page par page.
"""

from dataclasses import dataclass
from pathlib import Path
import re

try:
    import fasttext
    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False


ARABIC_CHAR_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")

import os
_FASTTEXT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models/lid.176.bin")
_FASTTEXT_MODEL = None

import logging
logger = logging.getLogger(__name__)

@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    method: str


def _load_fasttext_model(model_path: str = _FASTTEXT_MODEL_PATH):
    global _FASTTEXT_MODEL

    if _FASTTEXT_MODEL is None:

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Modèle fasttext introuvable : {model_path}"
            )

        _FASTTEXT_MODEL = fasttext.load_model(model_path)

    return _FASTTEXT_MODEL


def _detect_with_unicode_heuristic(text: str):

    if not text.strip():
        return LanguageDetectionResult(
            language="unknown",
            confidence=0.0,
            method="heuristique_unicode"
        )

    arabic_chars = len(ARABIC_CHAR_RANGE.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return LanguageDetectionResult(
            language="unknown",
            confidence=0.0,
            method="heuristique_unicode"
        )

    ratio = arabic_chars / total_alpha

    if ratio > 0.5:
        return LanguageDetectionResult(
            language="ar",
            confidence=ratio,
            method="heuristique_unicode"
        )
    
    if 0.4 <= ratio <= 0.5:
        return LanguageDetectionResult(
            language="mixed",
            confidence=0.5,
            method="heuristique_unicode"
        )

    return LanguageDetectionResult(
        language="fr",
        confidence=1 - ratio,
        method="heuristique_unicode"
    )


def detect_language(text: str, use_fasttext: bool = True):

    text = text.strip()

    if not text:
        return LanguageDetectionResult(
            language="unknown",
            confidence=0,
            method="none"
        )

    if use_fasttext and _FASTTEXT_AVAILABLE:

        try:

            model = _load_fasttext_model()

            labels, probs = model.predict(
                text.replace("\n", " "),
                k=1
            )
            logger.debug(f"Langue détectée par fasttext: {labels[0]} (conf={probs[0]:.2f})")

            return LanguageDetectionResult(
                language=labels[0].replace("__label__", ""),
                confidence=float(probs[0]),
                method="fasttext"
            )

        except (FileNotFoundError, PermissionError, OSError):
            logger.warning("Impossible de charger le modèle fasttext, fallback sur heuristique")

    return _detect_with_unicode_heuristic(text)


def split_text_by_language_per_line(text: str):

    result = {
        "fr": [],
        "ar": [],
        "unknown": []
    }

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lang = detect_language(line)

        if lang.language == "fr":
            result["fr"].append(line)

        elif lang.language == "ar":
            result["ar"].append(line)

        else:
            result["unknown"].append(line)

    return {
        k: "\n".join(v)
        for k, v in result.items()
    }


# --------------------------------------------------------------------
# NOUVELLES FONCTIONS POUR LE PIPELINE
# --------------------------------------------------------------------

def detect_page_language(page):
    """
    Détecte la langue d'une page.

    Ajoute automatiquement :

        page.language
        page.language_confidence
        page.language_method
    """

    result = detect_language(page.text)

    page.language = result.language
    page.language_confidence = result.confidence
    page.language_method = result.method

    return result


def detect_document_languages(document):
    """
    Détecte la langue de toutes les pages du document.
    """

    for page in document.pages:
        detect_page_language(page)

    return document


def split_document_by_language(document):
    """
    Regroupe le texte de toutes les pages selon leur langue.

    Returns
    -------
    {
        "fr": "...",
        "ar": "...",
        "unknown": "..."
    }
    """

    result = {
        "fr": [],
        "ar": [],
        "unknown": []
    }

    for page in document.pages:

        if not hasattr(page, "language"):
            detect_page_language(page)

        if page.language in result:
            result[page.language].append(page.text)

        else:
            result["unknown"].append(page.text)

    return {
        k: "\n\n".join(v)
        for k, v in result.items()
    }


if __name__ == "__main__":

    sample_fr = "Le décret n°2-21-441 relatif au ministère de l'Intérieur."

    sample_ar = "المرسوم رقم 2.21.441 المتعلق بوزارة الداخلية"

    print(detect_language(sample_fr))

    print(detect_language(sample_ar))