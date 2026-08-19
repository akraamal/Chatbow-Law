"""
test_segmenter_ar_instrument_boundary.py
----------------------------------------
Régression pour la segmentation arabe du BO (src/preprocessing/segmenter.py).

Cas gelé : BO_7517_Ar (arrêtés 899.26–1043.26).  L'édition arabe du BO
compose les blocs de citation multi-lignes (« texte amendé ») en répétant
le guillemet OUVERT « au début de chaque ligne et en refermant par un seul
» en fin de bloc (vérifié sur les 26 blocs du document) — et le » de
fermeture manque parfois sous OCR.  La règle de « paire à la française »
(last « après le dernier ») filtrait donc à tort de VRAIS en-têtes
d'article situés après un bloc non refermé (ex. « المادة الثانية » du
1037.26) et provoquait l'avalement des instruments suivants.

Correctifs gelés ici :
1. guillemets arabes : un marqueur est cité ssi sa propre ligne commence
   par « ou » (règle de ligne, indépendante du sens d'ouverture) ;
2. références croisées en début de ligne (« المادة 2 منه؛ ») : citations
   de préambule, pas des en-têtes ;
3. frontière de titre : un article se termine au titre de l'instrument
   suivant (قرار/مرسوم/ظهير... en début de ligne) — plus d'avalement du
   titre + préambule du voisin.

Usage:
    python -m pytest tests/test_segmenter_ar_instrument_boundary.py -v
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.segmenter import (
    DOCUMENT_TITLE_PATTERN_AR,
    get_per_decree_preamble_map,
    segment_into_articles,
    segment_into_articles_ar,
)

AR_PROCESSED = Path("data/processed/ar")
BO7517 = AR_PROCESSED / "BO_7517_Ar.txt"

# Les 19 instruments qui n'avaient qu'UN article (premier article avalé) :
# l'arrêté 1.26.15, les arrêtés 899.26–910.26 et 1037.26, 1039.26–1043.26.
SINGLE_ARTICLE_REFS = [
    "1.26.15",
    "899.26", "900.26", "901.26", "902.26", "903.26", "904.26",
    "905.26", "906.26", "907.26", "908.26", "909.26", "910.26",
    "1037.26", "1039.26", "1040.26", "1041.26", "1042.26", "1043.26",
]


def _segment_7517():
    text = BO7517.read_text(encoding="utf-8")
    return text, segment_into_articles(text, lang="ar")


def test_7517_nineteen_instruments_get_two_articles():
    """Chaque instrument auparavant mono-article (BO_7517_Ar) doit avoir
    « المادة االولى » ET « المادة الثانية » comme deux articles distincts."""
    text, articles = _segment_7517()
    decrees = get_per_decree_preamble_map(text, lang="ar")

    for i, dec in enumerate(decrees):
        if not any(ref in dec["preamble"] for ref in SINGLE_ARTICLE_REFS):
            continue
        end = decrees[i + 1]["first_article_idx"] if i + 1 < len(decrees) else len(articles)
        n_articles = end - dec["first_article_idx"]
        assert n_articles >= 2, (
            f"instrument mono-article : {dec['preamble'][:70]!r} "
            f"({n_articles} article(s))"
        )
        first = articles[dec["first_article_idx"]]
        assert first.raw_header.startswith("المادة االولى"), (
            f"premier article inattendu : {first.raw_header!r}"
        )


def test_7517_quoted_amendment_is_not_a_boundary():
    """Le «المادة االولى. - تحدد...» cité dans le texte amendé ne crée PAS
    une frontière : il reste dans le corps de l'article réel."""
    text, articles = _segment_7517()
    decrees = get_per_decree_preamble_map(text, lang="ar")
    d899 = next(d for d in decrees if "899.26" in d["preamble"])
    first = articles[d899["first_article_idx"]]
    assert first.text.startswith("تتمم على النحو التالي"), (
        "l'article réel du 899.26 doit commencer par son vrai contenu"
    )
    assert "اللجنة القطاعية لعلوم الصحة.»" in first.text, (
        "le bloc cité «...» doit rester dans l'article (pas de frontière)"
    )


def test_7517_second_article_does_not_swallow_next_title():
    """« المادة الثانية » du 899.26 ne doit PAS avaler le titre + préambule
    du 900.26 (frontière de titre)."""
    text, articles = _segment_7517()
    decrees = get_per_decree_preamble_map(text, lang="ar")
    d899 = next(d for d in decrees if "899.26" in d["preamble"])
    second = articles[d899["first_article_idx"] + 1]
    assert second.raw_header.startswith("المادة الثانية")
    assert "900.26" not in second.text, "titre du 900.26 avalé par le 899.26"
    assert "قرار لوزير التعليم العالي" not in second.text, "titre avalé en fin d'article"


def test_7517_no_title_bleed_in_any_article():
    """Aucun article (BO_7517_Ar) ne contient une ligne de titre d'instrument."""
    text, articles = _segment_7517()
    for a in articles:
        assert not DOCUMENT_TITLE_PATTERN_AR.search(a.text), (
            f"titre avalé dans l'article {a.raw_header!r} @{a.start_pos}"
        )


def test_ar_quoted_marker_in_multi_line_block_not_boundary():
    """Bloc cité multi-lignes répétant « en tête de chaque ligne (convention
    réelle du BO arabe) : le «المادة االولى» cité ne doit pas créer d'article."""
    text = (
        "وزير التعليم،\n"
        "قرر ما يلي :\n"
        "المادة االولى\n"
        "تغير على النحو التالي المادة الرابعة من القرار المشار اليه ااعله :\n"
        "«المادة الرابعة. - اذا تغيب السيد لحسن معزيزي او عاقه عائق ناب\n"
        "«عنه السيد عثمان العداوي، المتصرف من الدرجة االولى بمديرية\n"
        "«الموارد البشرية لوزارة التجهيز والماء.\n"
        "المادة الثانية\n"
        "ينشر هذا القرار بالجريدة الرسمية.\n"
    )
    articles = segment_into_articles_ar(text)
    assert len(articles) == 2, [a.raw_header for a in articles]
    assert articles[0].raw_header.startswith("المادة االولى")
    assert "المادة الرابعة. - اذا تغيب" in articles[0].text


def test_ar_bb_repeating_quote_block_not_boundary():
    """Bloc cité multi-lignes répétant » en tête de chaque ligne (variante
    signalée par l'analyse manuelle) : même exigence, aucun faux article."""
    text = (
        "المادة االولى\n"
        "تتضمن اللائحة الموالية :\n"
        "»المادة االولى. - تحدد اللائحة الشهادات التي تعادل دبلوم التخصص\n"
        "»دبلوم التخصص في الطب، تخصص : امراض الكلي :\n"
        "»شهادة التخصص في االمراض الجلدية، مسلمة من جامعة سوسة.\n"
        "«\n"
        "المادة الثانية\n"
        "ينشر هذا القرار بالجريدة الرسمية.\n"
    )
    articles = segment_into_articles_ar(text)
    assert len(articles) == 2, [a.raw_header for a in articles]
    assert articles[0].raw_header.startswith("المادة االولى")
    assert "تتضمن اللائحة الموالية" in articles[0].text


def test_ar_unclosed_quote_block_aftermath_kept():
    """Bloc cité dont le » de fermeture manque sous OCR : le VRAI marqueur
    qui suit (« المادة الثانية ») doit être conservé (règle de ligne, pas
    de paire à la française)."""
    text = (
        "قرر ما يلي :\n"
        "المادة االولى\n"
        "تغير على النحو التالي المادة الرابعة :\n"
        "«المادة الرابعة. - اذا تغيب السيد لحسن معزيزي\n"
        "«عنه السيد عثمان العداوي.\n"
        "المادة الثانية\n"
        "ينشر هذا القرار بالجريدة الرسمية.\n"
    )
    articles = segment_into_articles_ar(text)
    assert len(articles) == 2, [a.raw_header for a in articles]
    assert articles[1].raw_header.startswith("المادة الثانية")


def test_ar_inline_quote_after_marker_not_swallowed():
    """Une citation courte «...» (style français) en cours d'article ne doit
    pas faire disparaître les marqueurs suivants."""
    text = (
        "المادة االولى\n"
        "يفوض الى السيد محمد الزهاوي االمضاء على االوامر المتعلقة بعبارة «فرض»\n"
        "في المواد التالية.\n"
        "المادة الثانية\n"
        "ينشر هذا القرار بالجريدة الرسمية.\n"
    )
    articles = segment_into_articles_ar(text)
    assert len(articles) == 2, [a.raw_header for a in articles]


def test_ar_crossref_not_a_boundary():
    """« المادة 2 منه؛ » en début de ligne (citation de préambule) n'est
    pas un en-tête d'article."""
    text = (
        "وزير التعليم العالي،\n"
        "بناء على المرسوم رقم 2.24.991 الصادر في 28 اكتوبر 2024، اول سيما\n"
        "المادة 2 منه؛\n"
        "وعلى قرار وزير التربية الوطنية رقم 753.06،\n"
        "قرر ما يلي :\n"
        "المادة االولى\n"
        "تتمم اللائحة الموالية.\n"
        "المادة الثانية\n"
        "ينشر هذا القرار بالجريدة الرسمية.\n"
    )
    articles = segment_into_articles_ar(text)
    assert len(articles) == 2, [a.raw_header for a in articles]
    assert articles[0].raw_header.startswith("المادة االولى")


def test_fr_guillemet_behavior_unchanged():
    """Français : la paire «...» continue de masquer les marqueurs cités."""
    text = (
        "Arrêté du ministre,\n"
        "Vu le dahir n° 1-15-52, « et notamment son article 2, qui prévoit\n"
        "la liste des diplômes équivalents » ;\n"
        "ART. 1. – Les diplômes visés sont les suivants :\n"
        "ART. 2. – Le présent arrêté sera publié au Bulletin officiel.\n"
    )
    articles = segment_into_articles(text, lang="fr")
    assert len(articles) == 2, [a.raw_header for a in articles]


def test_corpus_ar_no_title_bleed():
    """Invariant corpus-wide : aucun article arabe ne contient une ligne de
    titre d'instrument (رئيسي : مرسوم/قرار/ظهير en début de ligne)."""
    if not BO7517.exists():
        return
    total = 0
    for f in sorted(AR_PROCESSED.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        for a in segment_into_articles_ar(text):
            total += 1
            assert not DOCUMENT_TITLE_PATTERN_AR.search(a.text), (
                f"{f.name}: titre avalé dans l'article {a.raw_header!r}"
            )
    assert total > 0