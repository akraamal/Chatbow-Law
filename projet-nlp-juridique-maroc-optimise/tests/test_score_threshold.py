"""
test_score_threshold.py
------------------------
Verrouille le seuil de similarité anti-hallucination (#6) dans
src/rag/chatbot.py. Ne charge ni le LLM ni l'index : garde-fou de
régression sur la constante calibrée.

Historique de calibration (2026-08-03, 24 requêtes labelisées, index
1161 docs E5) :
    relevant top-1 : min 0.819 / median 0.833 / max 0.844
    off-topic top-1 : min 0.777 / median 0.801 / max 0.818
    seuil 0.82 → recall 11/12, fp 0/12, F1 0.957 (ancienne politique)

Politique actuelle (2026-08-09) : le seuil est un FILTRE DE QUALITÉ DU
CONTEXTE, plus le garde-fou principal — le garde-fou est désormais le
vérificateur de citations : toute réponse sans citation mécaniquement
vérifiée est refusée (logique "verrou à la sortie" dans answer()).
Un seuil élevé (0.82) bloque la plupart des questions réelles (scores
saturant 0.78-0.82 sur corpus juridique homogène) ; 0.75 filtre le
bruit tout en gardant un recall élevé, le refus en aval contrôlant le
risque d'hallucination.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.chatbot import DEFAULT_SCORE_THRESHOLD


def test_threshold_is_permissive_retrieval_filter():
    """0.75 : assez bas pour laisser passer les questions réelles (ancien
    min pertinent 0.819), assez haut pour couper le bruit manifeste."""
    assert DEFAULT_SCORE_THRESHOLD <= 0.78, (
        f"DEFAULT_SCORE_THRESHOLD={DEFAULT_SCORE_THRESHOLD} : un seuil élevé "
        f"réintroduit les refus de l'ancienne politique (0.82 bloquait la "
        f"plupart des questions réelles)."
    )


def test_threshold_still_filters_noise():
    """0.75 reste significatif : on ne charge pas le prompt avec des chunks
    totalement hors-sujet (scores < 0.75)."""
    assert DEFAULT_SCORE_THRESHOLD >= 0.70, (
        f"DEFAULT_SCORE_THRESHOLD={DEFAULT_SCORE_THRESHOLD} : sous 0.70 le "
        f"contexte serait pollué par du bruit."
    )


def test_threshold_not_inert_policy():
    """Garde : la constante est présente et documentée dans chatbot.py."""
    assert 0.70 <= DEFAULT_SCORE_THRESHOLD <= 0.78