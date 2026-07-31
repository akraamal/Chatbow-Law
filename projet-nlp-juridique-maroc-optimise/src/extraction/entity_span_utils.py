"""
src/extraction/entity_span_utils.py
Petit adaptateur : les entités produites par l'étape 3 (entities.py) peuvent
utiliser différentes conventions de clés pour les positions caractère
(start/end, start_char/end_char, span:[start,end]...). Ce module centralise
la lecture pour que citation_resolver.py, ner_merge.py et etape4_pipeline.py
n'aient pas à deviner.

Si ton format réel est différent des trois essayés ci-dessous, ajoute-le
dans _START_KEYS / _END_KEYS plutôt que de modifier chaque fichier.
"""

_START_KEYS = ("start", "start_char", "char_start")
_END_KEYS = ("end", "end_char", "char_end")


def get_start(entity: dict) -> int:
    for key in _START_KEYS:
        if key in entity:
            return entity[key]
    if "span" in entity and isinstance(entity["span"], (list, tuple)):
        return entity["span"][0]
    return -1


def get_end(entity: dict) -> int:
    for key in _END_KEYS:
        if key in entity:
            return entity[key]
    if "span" in entity and isinstance(entity["span"], (list, tuple)):
        return entity["span"][1]
    return -1


def normalize_entity(entity: dict) -> dict:
    """Retourne une copie de l'entité avec des clés 'start'/'end' garanties,
    en plus des clés d'origine (rien n'est supprimé)."""
    normalized = dict(entity)
    normalized["start"] = get_start(entity)
    normalized["end"] = get_end(entity)
    return normalized