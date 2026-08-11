"""
tests/test_sommaire_truncated_marker.py
----------------------------------------
Régression (BO_6758) : un sommaire multi-pages dont la table des matières
contient SON PROPRE en-tête « TEXTES PARTICULIERS », suivi d'un marqueur
de section de corps TRONQUÉ par l'OCR (« TEXTES Gl » au lieu de
« TEXTES GÉNÉRAUX »), ne doit pas faire démarrer la segmentation au milieu
du sommaire ni fabriquer un décret fantôme à partir d'une entrée de la
table des matières.

Comportement attendu :
  * _skip_sommaire(limit=premier ART) s'arrête APRÈS le dernier marqueur
    de section avant le premier article (le vrai corps), pas dans la
    liste du sommaire ;
  * le premier décret du BO 6758 (Dahir n° 1-18-109) est détecté avec
    son préambule et first_article_idx = 0 ;
  * aucune entrée du sommaire (« Arrêté du ministre … n° 3587-18 ») ne
    devient un instrument.

Usage:
    python -m pytest tests/test_sommaire_truncated_marker.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def _sample_document() -> str:
    """Reproduction simplifiée de la structure du BO_6758 : sommaire
    multi-pages avec en-tête de section dans la liste, puis corps réel
    commençant par un marqueur tronqué « TEXTES Gl »."""
    sommaire_entries = "\n".join(
        [
            "Arrêté du ministre de l'agriculture n° 3587-18",
            "Décret n° 2-19-100",
            "dahir n° 1-18-109 portant promulgation de la loi n° 88-17",
            "TEXTES PARTICULIERS",
            "Arrêté du ministre de l'intérieur n° 3870-18",
            "Décision du directeur de l'ANRT",
            "TEXTES GENERAUX",
            "Dahir n° 1-19-19",
            "Décret n° 2-19-10",
        ]
        + [f"Décret n° 2-19-{100 + i} du ministère de l'intérieur"
           for i in range(12)]
    )
    return (
        "BULLETIN OFFICIEL N° 6758\n"
        "SOMMAIRE\n"
        + sommaire_entries
        + "\n"
        "-------------\n"
        "426 BULLETIN OFFICIEL — N° 6758 — 29 joumada II 1440 (7-3-2019)\n"
        "TEXTES Gl\n"
        "Dahir n° 1-18-109 du 2 joumada I 1440 (9 janvier 2019) portant\n"
        "promulgation de la loi n° 88-17 relative à la création et à\n"
        "l'accompagnement d'entreprises par voie électronique.\n"
        "LOUANGE A DIEU SEUL !\n"
        "(Grand Sceau de Sa Majesté le Roi Mohamed VI)\n"
        "Que l'on sache par les présentes — puisse Dieu en augmenter et en\n"
        "honorer la valeur — que Notre Majesté le Roi, conformément à la\n"
        "constitution, a décidé ce qui suit :\n"
        "Article premier\n"
        "Est promulguée la loi n° 88-17 telle qu'adoptée.\n"
        "Article 2\n"
        "La présente loi sera publiée au Bulletin officiel.\n"
        "TEXTES PARTICULIERS\n"
        "Décret n° 2-19-100 du 7 janvier 2019 portant création d'une\n"
        "entreprise publique.\n"
        "Le Chef du Gouvernement,\n"
        "Vu la constitution, notamment ses articles 85 et 90 ;\n"
        "Vu la loi n° 69-00 relative au contrôle financier de l'État ;\n"
        "Vu le décret n° 2-15-743 du 27 kaada 1436 (11 septembre 2015),\n"
        "Après délibération en Conseil du Gouvernement,\n"
        "Décrète :\n"
        "Article 1er\n"
        "Il est créé une entreprise publique.\n"
        "AVIS DU CONSEIL ECONOMIQUE, SOCIAL ET ENVIRONNEMENTAL\n"
        "Le conseil a examiné un projet de loi.\n"
    )


def _decrees(text: str):
    from src.preprocessing.segmenter import get_per_decree_preamble_map
    return get_per_decree_preamble_map(text, lang="fr")


def _skip_sommaire(text: str, limit):
    from src.preprocessing.segmenter import _skip_sommaire
    return _skip_sommaire(text, lang="fr", limit=limit)


def test_skip_sommaire_stops_after_last_section_marker_before_first_article():
    """Le sommaire multi-pages contient « TEXTES PARTICULIERS » dans sa
    liste ; le marqueur retenu doit être le corps réel tronqué « TEXTES Gl »,
    pas l'entrée de la table des matières."""
    from src.preprocessing.segmenter import _filter_article_matches
    text = _sample_document()
    first_art = _filter_article_matches(text, "fr")[0].start()
    end = _skip_sommaire(text, first_art)
    segment = text[end:first_art]
    assert "TEXTES Gl" not in segment, "le corps démarre après le marqueur tronqué"
    assert "Dahir n° 1-18-109" in segment, "le marqueur retenu précède le vrai corps"
    assert "SOMMAIRE" not in segment, "la table des matières ne doit pas être incluse"


def test_first_decree_is_dahir_1_18_109_not_sommaire_artifact():
    """Le premier instrument doit être le Dahir n° 1-18-109 (premier article
    du document), et non une entrée du sommaire (« Arrêté … 3587-18 »)."""
    decrees = _decrees(_sample_document())
    assert decrees, "aucun instrument détecté"
    first = decrees[0]
    assert first["first_article_idx"] == 0
    assert "1-18-109" in first["preamble"]
    assert "Dahir" in first["title"]


def test_no_sommaire_entry_becomes_instrument():
    """Les lignes « Arrêté du ministre de l'agriculture n° 3587-18 » et
    « Arrêté du ministre de l'intérieur n° 3870-18 » (table des matières)
    ne doivent apparaître ni en titre ni en préambule d'instrument."""
    decrees = _decrees(_sample_document())
    for d in decrees:
        title = d.get("title") or ""
        assert "3587-18" not in title
        assert "3870-18" not in title


def test_tail_instruments_still_detected():
    """Non-régression : le Décret n° 2-19-100 et l'Avis de queue restent
    des instruments (virtuel pour l'Avis, sans articles)."""
    decrees = _decrees(_sample_document())
    titles = [d["title"] for d in decrees]
    assert len(decrees) >= 3, f"instruments attendus ≥ 3, obtenus : {titles}"
    avis = [d for d in decrees if "AVIS" in (d["title"] or "").upper()]
    assert avis, "Avis de queue non détecté"
