"""
entity_ruler_builder_fr.py
----------------------------
Construit un pipeline spaCy (français) et y attache deux sources
d'entités :

  1. Références légales (DAHIR, LOI, DECRET, ARRETE, BULLETIN_OFFICIEL) :
     détectées par regex dans loi_decrets_patterns.py (format numérique
     trop irrégulier pour un EntityRuler token-par-token — voir le
     docstring de ce fichier), puis converties en spans spaCy via
     doc.char_span() (entities.py).

  2. Ministères (MINISTERE) : mentions littérales ("ministère de
     l'intérieur", "ministre de la justice"...), assez régulières pour un
     EntityRuler classique chargé depuis patterns/fr/ministeres.jsonl.

On utilise nlp = spacy.blank("fr") plutôt qu'un modèle français entraîné
(fr_core_news_sm/md) : on n'a besoin ni de POS-tagging ni de NER statistique
ici (tout est basé règles), et ça évite une dépendance de téléchargement de
modèle pour cette étape. Le tokenizer français de spaCy suffit pour aligner
proprement les offsets de caractères sur des tokens.
"""

from pathlib import Path

import spacy

from src.extraction.entities import regex_matches_to_entities, entities_to_spacy_doc
from src.extraction.loi_decrets_patterns import LEGAL_REFERENCE_PATTERNS
from src.extraction.dates_patterns import extract_dates_fr

PATTERNS_DIR = Path(__file__).parent / "patterns" / "fr"
MINISTERES_JSONL = PATTERNS_DIR / "ministeres.jsonl"


def build_fr_nlp():
    """
    Construit et retourne un pipeline spaCy français avec un EntityRuler
    "MINISTERE" chargé depuis patterns/fr/ministeres.jsonl (si le fichier
    existe — sinon le pipeline fonctionne quand même, juste sans ce label).
    """
    nlp = spacy.blank("fr")

    if MINISTERES_JSONL.exists():
        ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
        ruler.from_disk(MINISTERES_JSONL)

    return nlp


def extract_legal_entities_fr(text: str, nlp=None):
    """
    Point d'entrée principal : extrait les références légales FR d'un
    texte et retourne un Doc spaCy avec doc.ents peuplé (MINISTERE via
    EntityRuler + DAHIR/LOI/DECRET/ARRETE/BULLETIN_OFFICIEL via regex).

    Args:
        text: texte français nettoyé (voir cleaner_fr.py), typiquement le
            contenu d'un article (segmenter.py) ou d'une page.
        nlp: pipeline spaCy à réutiliser (build_fr_nlp() par défaut). À
            construire une seule fois et à passer explicitement si on
            traite beaucoup de documents, pour éviter de recharger
            l'EntityRuler à chaque appel.

    Returns:
        spacy.tokens.Doc avec les entités fusionnées dans doc.ents.
    """
    if nlp is None:
        nlp = build_fr_nlp()

    regex_entities = regex_matches_to_entities(text, LEGAL_REFERENCE_PATTERNS, lang="fr")
    date_entities = extract_dates_fr(text)

    return entities_to_spacy_doc(nlp, text, regex_entities + date_entities)


if __name__ == "__main__":
    sample = (
        "Vu le dahir n° 1-09-20 du 22 safar 1430 (18 février 2009) et le "
        "dahir portant loi n° 1-73-255 du 27 chaoual 1393, en application "
        "de la loi n° 03-25 relative aux organismes de placement collectif, "
        "le décret n°2-08-562 abroge l'arrêté du ministre de l'industrie et "
        "du commerce, publié au « Bulletin officiel » n° 7499 du 25 chaoual "
        "1447 (13 avril 2026)."
    )

    doc = extract_legal_entities_fr(sample)

    for ent in doc.ents:
        print(f"{ent.label_:20s} | {ent.text}")
