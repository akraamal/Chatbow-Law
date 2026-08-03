"""
test_score_threshold.py
-----------------------
Verrouille la calibration empirique du seuil anti-hallucination (#6).
Ne charge pas le LLM ni l'index : simple garde-fou de régression sur la
constante calibrée dans src/rag/chatbot.py.

Calibration (2026-08-03, 24 requêtes labelisées, index 1161 docs E5) :
    relevant top-1 : min 0.819 / median 0.833 / max 0.844
    off-topic top-1 : min 0.777 / median 0.801 / max 0.818
    seuil 0.82 → recall 11/12, fp 0/12, F1 0.957 (meilleur compromis)
Un seuil <= 0.80 laisse passer la moitié des requêtes hors-sujet (fp 6/12).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.chatbot import DEFAULT_SCORE_THRESHOLD


def test_threshold_calibrated_not_inert():
    """0.82 bloque tout le hors-sujet testé ; un seuil <= 0.80 est inerte."""
    assert DEFAULT_SCORE_THRESHOLD == 0.82, (
        f"DEFAULT_SCORE_THRESHOLD={DEFAULT_SCORE_THRESHOLD} : la calibration "
        f"empirique #6 impose 0.82 (recall 11/12, fp 0/12 sur 24 requêtes)."
    )


def test_threshold_blocks_offtopic_distribution():
    """Le seuil doit être strictement au-dessus du max hors-sujet observé (0.818)."""
    off_topic_max = 0.818
    assert DEFAULT_SCORE_THRESHOLD > off_topic_max, (
        "Le seuil est sous le max des scores hors-sujet observés (0.818) : "
        "le garde-fou laisserait passer des questions hors domaine."
    )
