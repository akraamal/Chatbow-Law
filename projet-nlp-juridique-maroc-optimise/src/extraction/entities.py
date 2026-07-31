"""
entities.py
------------
Structures partagées et fonctions utilitaires pour l'étape 3 (extraction
NLP) : conversion des correspondances regex (loi_decrets_patterns.py /
loi_decrets_patterns_ar.py) en entités exploitables, indépendamment de la
langue.

Pourquoi passer par spaCy alors que tout est déjà trouvé par regex ?
  - Un objet Doc/Span spaCy donne une représentation uniforme (position en
    tokens ET en caractères) réutilisable telle quelle par les étapes
    suivantes (classification, indexation FAISS) qui attendent des Doc
    spaCy.
  - doc.char_span() valide que les offsets regex tombent bien sur des
    limites de tokens ; sinon on le sait immédiatement au lieu de découvrir
    des entités mal alignées plus tard dans le pipeline.
"""

from dataclasses import dataclass, field
import spacy  # <-- AJOUT IMPORTANT
from spacy.tokens import Span

if not Span.has_extension("meta"):
    Span.set_extension("meta", default=None)

@dataclass
class LegalEntity:
    label: str          # "DAHIR", "LOI", "DECRET", "ARRETE", "BULLETIN_OFFICIEL"...
    text: str           # texte exact matché
    start: int      # offset de début dans le texte source
    end: int        # offset de fin dans le texte source
    lang: str            # "fr" ou "ar"
    meta: dict = field(default_factory=dict)  # données supplémentaires (ex : date associée, ministère, etc.)


def regex_matches_to_entities(text: str, patterns: dict, lang: str) -> list:
    """
    Applique chaque regex de `patterns` (dict label -> Pattern compilé) sur
    `text` et retourne la liste des LegalEntity trouvées, triées par
    position d'apparition dans le texte.

    Les chevauchements entre labels différents ne sont PAS résolus ici
    (rare en pratique vu la spécificité des mots-clés déclencheurs : un
    même passage ne matche normalement qu'un seul type de référence) — à
    surveiller si de nouveaux patterns sont ajoutés.
    """
    found = []

    for label, pattern in patterns.items():
        for match in pattern.finditer(text):
            found.append(
                LegalEntity(
                    label=label,
                    text=match.group(0).strip(),
                    start=match.start(),
                    end=match.end(),
                    lang=lang,
                )
            )

    found.sort(key=lambda e: e.start)
    return found


def entities_to_spacy_doc(nlp, text: str, entities: list, alignment_mode="expand"):
    """
    Exécute `nlp(text)` (ce qui déclenche au passage les éventuels
    composants de pipeline déjà attachés, ex : un EntityRuler pour les
    ministères), puis ajoute `entities` (liste de LegalEntity, typiquement
    issue de regex_matches_to_entities) comme entités supplémentaires dans
    doc.ents, via doc.char_span().

    Les entités déjà posées par le pipeline (EntityRuler) sont conservées
    en priorité ; une entité regex qui chevauche une entité déjà posée est
    ignorée (Doc.ents refuse les spans qui se recouvrent).

    alignment_mode="expand" : si les offsets regex ne tombent pas
    exactement sur une frontière de token spaCy, on élargit le span au
    token englobant plutôt que de rejeter l'entité — préférable ici car on
    veut auditer visuellement les résultats plutôt que perdre des
    correspondances trouvées par regex.
    """
    doc = nlp(text)

    final_spans = list(doc.ents)
    occupied = [(s.start, s.end) for s in final_spans]

    for ent in entities:
        span = doc.char_span(
            ent.start,
            ent.end,
            label=ent.label,
            alignment_mode="expand",
        )

        if span is None:
            # char_span peut renvoyer None sur du texte vide/whitespace en bord de match
            continue

        overlaps = any(span.start < o_end and span.end > o_start for o_start, o_end in occupied)
        if overlaps:
            continue

        final_spans.append(span)
        occupied.append((span.start, span.end))
    

    doc.ents = sorted(final_spans, key=lambda s: s.start)
    return doc

def extract_ruler_entities(doc: spacy.tokens.Doc, lang: str) -> list:
    """Extrait les entités posées par l'EntityRuler sous forme de LegalEntity."""
    ruler_entities = []
    for ent in doc.ents:
        # Vérifier si l'entité vient du EntityRuler (pas de regex)
        # Astuce : les entités du ruler ont souvent un attribut "source" non défini
        ruler_entities.append(
            LegalEntity(
                label=ent.label_,
                text=ent.text,
                start=ent.start,
                end=ent.end,
                lang=lang,
                meta={"source": "entity_ruler"}
            )
        )
    return ruler_entities

def validate_entity_offsets(text: str, entities: list) -> list:
    """Vérifie que les entités ne sortent pas des limites du texte."""
    valid = []
    for ent in entities:
        if 0 <= ent.start < len(text) and 0 <= ent.end <= len(text):
            valid.append(ent)
    return valid