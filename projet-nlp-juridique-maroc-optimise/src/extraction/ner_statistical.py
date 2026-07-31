"""
src/extraction/ner_statistical.py
Étape 4b (FR) — NER statistique (PERSON, ORG).

Différence structurelle avec entity_ruler_builder_fr.py (étape 3) : on ne
peut plus utiliser spacy.blank(), il faut un pipeline ENTRAÎNÉ, ce qui
ajoute une vraie dépendance de téléchargement de modèle :
    python -m spacy download fr_core_news_md
"""
import spacy

_NLP = None


def get_ner_model():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("fr_core_news_md")
    return _NLP


def extract_persons_orgs(text: str):
    """Retourne (persons, orgs), chacune une liste de {text, start, end, label}."""
    nlp = get_ner_model()
    doc = nlp(text)
    persons, orgs = [], []
    for ent in doc.ents:
        if ent.label_ == "PER":
            persons.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": "PERSON"})
        elif ent.label_ == "ORG":
            orgs.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": "ORGANIZATION"})
    return persons, orgs