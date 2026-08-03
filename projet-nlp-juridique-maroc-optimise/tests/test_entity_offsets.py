"""
test_entity_offsets.py
-----------------------
Régression : les offsets (start/end) d'une entité doivent pointer sur le
slice EXACT du texte source, de sorte que ``source.find(entity.text)``
fonctionne toujours.

Deux bugs corrigés :
  1. clean_entity_text() aplatissait les retours à la ligne en espaces
     ("الظهير الشريف رقم\n 1.08.49" → "الظهير الشريف رقم 1.08.49") SANS
     ajuster start/end : le text stocké différait du slice source et
     source.find() renvoyait -1 (audit BO_7408 : 175/1133 art., 17/105 préambule).
  2. les entités propagées depuis un préambule de décret étaient posées
     sur start/end du numéro seul mais gardaient le texte complet
     (ex. "قرار لوزير الافلحة والصيد البحري رقم 731.25" positionné sur
     "731.25") → span hors slice.

align_entity_text() restaure le slice exact dans les deux cas.

Usage:
    python -m pytest tests/test_entity_offsets.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_regular_entity_slice_invariant():
    """Pour une entité aux offsets valides, le slice source doit contenir
    le texte (avant correctif : texte aplati ≠ slice → find() = -1)."""
    from extraction.ner_filter import align_entity_text

    source = "الفقرة الأولى : الظهير الشريف رقم 1.08.49 الصادرة بتاريخ"
    start = source.find("الظهير الشريف رقم 1.08.49")
    assert start >= 0
    entity = {
        "label": "DAHIR",
        "text": "الظهير الشريف رقم\n1.08.49",   # aplatissement \n→espace
        "start": start,
        "end": start + len("الظهير الشريف رقم 1.08.49"),
    }
    aligned = align_entity_text(dict(entity), source)
    assert aligned is not None
    # Le slice doit être exactement le texte affiché.
    assert aligned["text"] == source[aligned["start"]:aligned["end"]]
    assert source.find(aligned["text"]) != -1


def test_newline_in_source_kept_in_restored_text():
    """Si le texte source du span contient réellement un retour à la ligne
    (le \n n'est PAS aplati car on restaure le slice exact), find() est
    trivial."""
    from extraction.ner_filter import align_entity_text

    source = "vu le ظهير شريف رقم\n 1.08.49 ;"
    start = source.find("ظهير شريف رقم")
    entity = {"label": "DAHIR", "text": "ظهير شريف رقم 1.08.49",
              "start": start, "end": source.find("49 ;") + 2}
    aligned = align_entity_text(dict(entity), source)
    assert aligned is not None
    assert source.find(aligned["text"]) != -1
    assert aligned["end"] - aligned["start"] == len(aligned["text"])


def test_propagated_entity_number_fallback():
    """Entité propagée (start=-1) dont le texte complet n'est pas présent
    dans l'article : align_entity_text la retrouve (texte ou numéro) et
    restaure un slice exact, ou la rejette (None) si introuvable."""
    from extraction.ner_filter import align_entity_text

    source = "قرار لوزير الفلاحة والصيد البحري رقم 731.25 يحدد القائمة."
    entity = {"label": "ARRETE",
              "text": "قرار لوزير الفلاحة والصيد البحري",
              "start": -1, "end": -1}
    aligned = align_entity_text(dict(entity), source)
    assert aligned is not None
    assert source.find(aligned["text"]) != -1
    assert aligned["text"] == source[aligned["start"]:aligned["end"]]

    # Texte totalement absent (ni texte ni numéro) → None (écarté).
    absent = {"label": "LOI", "text": "قانون رقم 9.99",
              "start": -1, "end": -1}
    assert align_entity_text(dict(absent), source) is None


def test_propagated_number_only_restores_number_slice():
    """Cas « 731.25 » de l'audit : le texte complet n'est pas dans
    l'article, seul le numéro l'est.  L'entité est positionnée sur le
    numéro et son text devient le slice exact du numéro (plus de texte
    fantôme couvrant tout le début de phrase)."""
    from extraction.ner_filter import align_entity_text

    source = "en application de l'arrêté n° 731.25 du 3 mars."
    entity = {"label": "ARRETE",
              "text": "l'arrêté du ministre n° 731.25 du 3 mars",
              "start": -1, "end": -1}
    aligned = align_entity_text(dict(entity), source)
    assert aligned is not None
    assert aligned["text"] == source[aligned["start"]:aligned["end"]]
    assert "731.25" in aligned["text"]
    assert source.find(aligned["text"]) != -1