"""
src/extraction/ner_merge.py
Fusion des entités statistiques (ner_statistical[_ar].py) avec les entités à
base de règles (MINISTERE/INSTITUTION, extraites par entity_ruler_builder_fr/ar.py
à l'étape 3).

Différence avec la fusion de l'étape 3 (entities.py / entities_to_spacy_doc) :
en cas de chevauchement, priorité aux entités RÈGLES (plus fiables, listes
fermées) sur les entités STATISTIQUES (plus bruitées) — c'est l'inverse de
la logique par défaut utilisée jusqu'ici.
"""


def merge_with_rule_based_entities(statistical_entities, rule_based_entities):
    def overlaps(a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]

    kept = list(rule_based_entities)
    for stat_ent in statistical_entities:
        if not any(overlaps(stat_ent, r) for r in rule_based_entities):
            kept.append(stat_ent)

    return sorted(kept, key=lambda e: e["start"])