"""
src/rag/query_routing.py
----------------------------
Aiguillage des questions vers le chemin adapté :

  - chemin « sémantique » (FAISS) : questions factuelles sur des extraits
    (« qui délivre le permis de construire ? », « quel est le taux de TVA ? ») ;
  - chemin « catalogue » (src/search_engine/catalog.py) : questions AGRÉGÉES
    ou par référence — « les dahirs les plus importants », « les décrets de
    2024 », « combien d'articles comporte le décret n° 2-25-1080 ? ».

L'aiguillage est purement lexical (rapide, déterministe, testable) : un
nom d'instrument (dahir, décret, arrêté, loi, décision — ou leur équivalent
arabe) associé à un signal d'agrégation (liste, importance, année,
référence numérique) déclenche le catalogue.
"""
from __future__ import annotations

import re
import unicodedata

_INSTRUMENT_ALIASES: list[tuple[str | None, dict[str, tuple[str, ...]]]] = [
    ("Dahir", {"fr": ("dahir", "dahirs"), "ar": ("الظهير", "الظهائر", "ظهائر", "ظهير")}),
    ("Loi", {"fr": ("loi", "lois"), "ar": ("القانون", "القوانين", "قانون", "قوانين")}),
    ("Décret", {"fr": ("décret", "décrets", "decret", "decrets"),
                "ar": ("المرسوم", "المراسيم", "مراسيم", "مرسوم")}),
    ("Arrêté", {"fr": ("arrêté", "arrêtés", "arrete", "arretes"),
                "ar": ("القرارات", "القرار", "قرارات", "قرار")}),
    ("Décision", {"fr": ("décision", "décisions", "decision", "decisions"),
                  "ar": ("المقررات", "المقرر", "مقررات", "مقرر")}),
    ("Avis", {"fr": ("avis",),
              "ar": ("التعليمات", "تعليمات", "البيانات")}),
    ("Ordonnance", {"fr": ("ordonnance", "ordonnances"),
                    "ar": ("الأمر", "أمر")}),
    # Instruments génériques : sans type — le catalogue répondra tous types.
    (None, {"fr": ("texte", "textes", "instrument", "instruments",
                   "norme", "normes", "textes règlementaires", "règlementaire"),
            "ar": ("النصوص", "نصوص", "التشريعات", "تشريعات")}),
]

# Signaux d'agrégation : la question demande une LISTE / un classement /
# une période plutôt qu'un fait unique.
_AGG_SIGNALS_FR = (
    "les plus", "plus importants", "plus importantes", "importants",
    "importantes", "majeurs", "majeures", "principaux", "principales",
    "récents", "récentes", "liste", "quels", "quelles", "tous", "toutes",
    "ensemble", "nombre", "adoptés", "publiés", "parus", "important",
    "importante", "majeur",
)
_AGG_SIGNALS_AR = (
    "أهم", "كل", "قائمة", "ما هي", "الأهم", "المهمة", "الرئيسية",
    "الصادرة", "المعتمدة", "المصادق", "المهم",
)

FR_REF_RE = re.compile(
    r"\b(?:n\s*[°o]?\s*)?[0-9]{1,2}(?:[-.][0-9]{1,4}){1,2}\b"
)
AR_REF_RE = re.compile(r"رقم\s*[0-9٠-٩]{1,4}(?:[-.][0-9٠-٩]{1,4}){1,2}")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")

# Les types « pluriel explicite » déclenchent l'agrégation même sans autre
# signal (« les dahirs », « les décrets »).
_PLURAL_SETS = {
    "fr": {"dahirs", "décrets", "decrets", "arrêtés", "arretes", "lois",
           "décisions", "decisions", "ordonnances", "avis"},
    "ar": {"الظهائر", "ظهائر", "المراسيم", "مراسيم", "القوانين", "قوانين",
           "القرارات", "قرارات", "المقررات", "مقررات", "النصوص", "نصوص",
           "التشريعات", "تشريعات"},
}

_ACCENTS = dict.fromkeys(
    ord(c) for c in unicodedata.normalize("NFD", "àâäéèêëîïôöùûüç")
    if unicodedata.category(c) == "Mn"
)


def _norm(text: str) -> str:
    if any("\u0600" <= c <= "\u06FF" for c in text):
        # NFD décomposerait les lettres hamza/inchoatives arabes (أ → ا +
        # U+0654) et casserait la correspondance avec l'alias.
        return text.lower()
    return unicodedata.normalize("NFD", text).translate(_ACCENTS).lower()


def _guess_lang(query: str) -> str:
    return "ar" if any("\u0600" <= c <= "\u06FF" for c in query) else "fr"


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


def _split_words(text: str) -> set[str]:
    return set(re.findall(r"[\w\u0600-\u06FF]+", text))


# Signaux de "vue d'ensemble" : la question porte sur le document/corpus
# comme un tout (résumé, thèmes, comparaison, structure), pas sur un fait
# ponctuel citable en une phrase.
_SYNTHESIS_SIGNALS_FR = (
    "résume", "résumé", "resume", "de quoi parle", "de quoi il s'agit",
    "de quoi ça parle", "sujet principal", "thèmes", "themes", "sujets",
    "à propos de quoi", "compare", "comparer", "comparaison",
    "différence entre", "quelle est la différence", "en quoi consiste",
    "que dit ce document", "que contient ce document", "structure de",
    "vue d'ensemble", "explique", "expliquer",
)
_SYNTHESIS_SIGNALS_AR = (
    "لخص", "ملخص", "عن ماذا يتحدث", "الموضوع الرئيسي", "المواضيع",
    "قارن", "المقارنة", "الفرق بين", "ماذا يتضمن", "بم يتعلق",
)


def _is_synthesis_query(q_norm: str, lang: str) -> bool:
    signals = _SYNTHESIS_SIGNALS_AR if lang == "ar" else _SYNTHESIS_SIGNALS_FR
    return any(s in q_norm for s in signals)


_AR_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _extract_references(q: str, lang: str, max_refs: int = 3) -> list[str]:
    """Isole les références numériques (ex. '2-25-1080') d'une question —
    plusieurs pour une comparaison ('compare le décret X et le décret Y'),
    dédupliquées et plafonnées à max_refs. Utilisé pour cibler
    get_document_chunks() en mode synthèse plutôt que de se limiter au
    top_k sémantique."""
    pattern = AR_REF_RE if lang == "ar" else FR_REF_RE
    seen: set[str] = set()
    out: list[str] = []
    for m in pattern.finditer(q):
        digits_only = re.search(r"[0-9\u0660-\u0669][0-9\u0660-\u0669\-.]*[0-9\u0660-\u0669]", m.group(0))
        if not digits_only:
            continue
        # Chiffres arabes-indiens (٢٥) → ASCII (25) : le catalogue indexe en ASCII.
        ref = digits_only.group(0).translate(_AR_TO_ASCII)
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
        if len(out) >= max_refs:
            break
    return out


def route_query(query: str, lang: str | None = None) -> dict:
    """
    Renvoie {"catalog": bool, "type": str|None, "year": int|None,
    "scope": str|None}.

    « catalog == True » signifie que la question doit être traitée par le
    catalogue d'instruments (agrégation / référence numérique), sinon elle
    suit le chemin sémantique classique.

    « scope == "synthesis" » signifie que la question porte sur une vue
    d'ensemble du contenu (résumé, thèmes, comparaison, structure) et doit
    être traitée par le mode synthèse (vérification d'ancrage allégée).
    """
    q = _norm(query).strip()
    if not q:
        return {
            "catalog": False,
            "type": None,
            "year": None,
            "scope": "synthesis" if _is_synthesis_query(q, lang) else None,
            "references": _extract_references(q, lang),
        }
    lang = lang or _guess_lang(query)
    words = _split_words(q)

    matched_type: str | None = None
    matched_word: str | None = None
    for canonical, aliases in _INSTRUMENT_ALIASES:
        lk = aliases.get(lang)
        if not lk:
            lk = aliases.get("fr", ())
        for w in lk:
            if _has_word(q, w):
                matched_type, matched_word = canonical, w
                break
        if matched_word:
            break

    if matched_word is None:
        return {
            "catalog": False,
            "type": None,
            "year": None,
            "scope": "synthesis" if _is_synthesis_query(q, lang) else None,
            "references": _extract_references(q, lang),
        }

    has_ref = bool(
        (AR_REF_RE if lang == "ar" else FR_REF_RE).search(q)
        or re.search(r"(?:19|20)\d{2}(?:[-.][0-9]{1,4}){1,2}", q)
    )
    m_year = YEAR_RE.search(q)
    year = int(m_year.group(0)) if m_year and len(m_year.group(0)) == 4 else None

    plural = matched_word in _PLURAL_SETS.get(lang, set())
    signals = _AGG_SIGNALS_AR if lang == "ar" else _AGG_SIGNALS_FR
    implied_agg = (
        has_ref
        or year is not None
        or plural
        or any(s in q for s in signals)
    )
    return {
        "catalog": bool(implied_agg),
        "type": matched_type,
        "year": year,
        "scope": "synthesis" if _is_synthesis_query(q, lang) else None,
        "references": _extract_references(q, lang),
    }