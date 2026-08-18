"""
entity_ruler_builder_ar.py
----------------------------
Équivalent arabe de entity_ruler_builder_fr.py.

Note importante sur le choix de nlp = spacy.blank("ar") :
    spaCy n'a PAS de modèle arabe entraîné officiel (pas de
    ar_core_news_sm), et camel-tools (mentionné dans requirements.txt) est
    la référence pour du NLP arabe avancé (lemmatisation, désambiguïsation
    morphologique). Mais pour CETTE étape — repérer des références légales
    au format très régulier une fois le mot-déclencheur identifié (ظهير,
    قانون, مرسوم...) — on n'a besoin que d'un tokenizer correct pour poser
    des spans propres via doc.char_span(). spacy.blank("ar") suffit très
    largement et évite d'ajouter camel-tools comme dépendance dure dès
    cette étape ; à réévaluer si une étape ultérieure a besoin de
    lemmatisation/désambiguïsation arabe plus poussée.
"""

from pathlib import Path

import spacy
import json
from src.extraction.entities import regex_matches_to_entities, entities_to_spacy_doc
from src.extraction.loi_decrets_patterns_ar import LEGAL_REFERENCE_PATTERNS_AR
from src.extraction.dates_patterns_ar import extract_dates_ar
from src.extraction.institution_patterns_ar import extract_institutions_ar

PATTERNS_DIR = Path(__file__).parent / "patterns" / "ar"
WIZARAT_JSONL = PATTERNS_DIR / "wizarat.jsonl"

def build_ar_nlp():
    nlp = spacy.blank("ar")
    if WIZARAT_JSONL.exists():
        ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
        with open(WIZARAT_JSONL, 'r', encoding='utf-8') as f:
            patterns = [json.loads(line) for line in f if line.strip()]
        # Conversion du format {"text": ..., "label": ...} vers
        # {"label": ..., "pattern": "..."} attendu par l'EntityRuler.
        # Le phrase_matcher_attr="LOWER" se charge du tokenizing et du
        # lowercasing de chaque mot de la phrase.
        patterns = [
            {"label": p["label"], "pattern": p["text"]}
            if "text" in p else p
            for p in patterns
        ]
        if patterns:
            ruler.add_patterns(patterns)
    return nlp

def normalize_arabic_digits_in_text(text: str) -> str:
    """Convertit les chiffres arabes (٠-٩) en chiffres latins (0-9)."""
    arabic_to_latin = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for ar, la in arabic_to_latin.items():
        text = text.replace(ar, la)
    return text


def extract_legal_entities_ar(text: str, nlp=None):
    """
    Point d'entrée principal : extrait les références légales arabes d'un
    texte et retourne un Doc spaCy avec doc.ents peuplé (MINISTERE via
    EntityRuler + DAHIR/LOI/DECRET/ARRETE/BULLETIN_OFFICIEL via regex).

    Args:
        text: texte arabe nettoyé (voir cleaner_ar.py), typiquement le
            contenu d'un article (segmenter.py) ou d'une page. Attention :
            si tashkeel/tatweel n'ont pas encore été retirés par
            arabic_utils.normalize_arabic_text, les regex de
            loi_decrets_patterns_ar.py peuvent rater des correspondances
            (diacritiques insérés au milieu des mots-clés) — s'assurer que
            le texte passé ici sort bien de clean_arabic_text().
        nlp: pipeline spaCy à réutiliser (build_ar_nlp() par défaut).

    Returns:
        spacy.tokens.Doc avec les entités fusionnées dans doc.ents.
    """
    
    text = normalize_arabic_digits_in_text(text)

    if nlp is None:
        nlp = build_ar_nlp()

    regex_entities = regex_matches_to_entities(text, LEGAL_REFERENCE_PATTERNS_AR, lang="ar")
    date_entities = extract_dates_ar(text)
    institution_entities = extract_institutions_ar(text)

    return entities_to_spacy_doc(nlp, text, regex_entities + date_entities + institution_entities)


if __name__ == "__main__":
    sample = (
        "بناء على الظهير الشريف رقم 1.09.20 الصادر في 22 صفر 1430 "
        "الموافق طبقا لقانون رقم 03.25 يتعلق بهيئات التوظيف الجماعي، "
        "وعلى مرسوم رقم 2.08.562، وعلى قرار لوزير الصناعة والتجارة، "
        "المنشور بالجريدة الرسمية عدد 7499 بتاريخ 25 شوال 1447."
    )

    doc = extract_legal_entities_ar(sample)

    for ent in doc.ents:
        print(f"{ent.label_:20s} | {ent.text}")
