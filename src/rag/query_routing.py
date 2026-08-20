"""
src/rag/query_routing.py
----------------------------
Aiguillage des questions vers le chemin adapté :

  - chemin « sémantique » (FAISS) : questions factuelles sur des extraits
    (« qui délivre le permis de construire ? », « quel est le taux de TVA ? ») ;
  - chemin « catalogue » (src/search_engine/catalog.py) : questions AGRÉGÉES
    ou par référence — « les dahirs les plus importants », « les décrets de
    2024 », « combien d'articles comporte le décret n° 2-25-1080 ? ».

L'aiguillage principal est purement lexical (rapide, déterministe, testable) :
un nom d'instrument (dahir, décret, arrêté, loi, décision — ou leur équivalent
arabe) associé à un signal d'agrégation (liste, importance, année,
référence numérique) déclenche le catalogue.

Repli bas-coût : quand le routeur lexical ne détecte NI instrument NI
signal de vue d'ensemble (question sémantique « plate »), un second
classifieur rattrape les questions d'agrégation (liste, décompte) ou de
synthèse mal formulées — d'abord par heuristiques de phrases (le cas de
l'agrégation est protégé par un garde-fou : présence d'un nom de corpus,
pour ne pas dérouter les questions factuelles « quels sont les délais de
recours ? »), puis par similarité d'embedding contre des questions
canoniques en réutilisant le modèle déjà chargé pour la recherche (aucun
chargement supplémentaire). La décision expose un champ `signal` et un
journal de débogage est disponible via ADLI_ROUTING_DEBUG=1.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import unicodedata

import numpy as np

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
# une période plutôt qu'un fait unique. NB : « quels »/« quelles » nus sont
# volontairement ABSENTS — « quels sont les délais de recours ? » est une
# question factuelle. Une question de liste sans nom d'instrument est
# rattrapée par le repli (fallback), protégé par le garde-fou des noms de
# corpus (_CORPUS_NOUNS_FR).
_AGG_SIGNALS_FR = (
    "les plus", "plus importants", "plus importantes", "importants",
    "importantes", "majeurs", "majeures", "principaux", "principales",
    "récents", "récentes", "liste", "tous", "toutes",
    "ensemble", "nombre", "adoptés", "publiés", "parus", "important",
    "importante", "majeur",
)
_AGG_SIGNALS_AR = (
    "أهم", "كل", "قائمة", "ما هي", "الأهم", "المهمة", "الرئيسية",
    "الصادرة", "المعتمدة", "المصادق", "المهم",
)

# ---------------------------------------------------------------------------
# Repli bas-coût (« fallback ») pour les questions sémantiques « plates »
# ---------------------------------------------------------------------------
# Le routeur lexical ci-dessus ne déclenche le catalogue / la synthèse que si
# la question nomme un instrument ou un signal explicite. Le repli ci-dessous
# ne s'applique qu'aux questions qui restent en sémantique (catalog=False et
# scope=None) : il tente de rattraper les questions d'agrégation (liste,
# décompte) ou de vue d'ensemble formulées sans nom d'instrument.

_LOGGER = logging.getLogger("src.rag.query_routing")
_DEBUG = os.environ.get("ADLI_ROUTING_DEBUG") == "1"

# Garde-fou d'agrégation : une phrase d'agrégation ne déclenche le catalogue
# que si la question parle explicitement du corpus (documents, textes,
# instruments...). « quels sont les délais de recours ? » ne contient aucun
# de ces noms → reste en sémantique.
_CORPUS_NOUNS_FR = (
    "bulletin", "document", "documents", "texte", "textes", "corpus",
    "norme", "normes", "instrument", "instruments", "code", "codes",
)
_CORPUS_NOUNS_AR = (
    "الجريدة", "الوثيقة", "الوثائق", "النص", "النصوص", "المدونة",
    "التشريعات", "التشريع", "المواد", "الفصول",
)

# Phrases d'agrégation (formes normalisées — accents retirés par _norm).
# Pas de « quels »/« quelle est » nus : trop risqué (« quels sont les délais
# de recours ? ») — le garde-fou des noms de corpus fait le reste.
_FALLBACK_AGG_PHRASES_FR = (
    "quels sont", "quelles sont", "lesquels", "lesquelles", "combien",
    "liste des", "liste de", "lister", "enumere", "recenser", "recense",
    "l'ensemble des", "ensemble des", "tous les", "toutes les",
    "donne-moi la liste", "donne moi la liste",
)
_FALLBACK_AGG_PHRASES_AR = (
    "كم عدد", "كم", "ما هي", "قائمة", "جميع", "عدد",
)

# Phrases de vue d'ensemble — sans garde-fou : elles désignent explicitement
# le contenu comme un tout (résumé, objet, contenu).
_FALLBACK_SYNTH_PHRASES_FR = (
    "resumer", "le propos de", "l'objet de", "de quoi s'agit-il",
    "quel est le contenu", "que contient",
)
_FALLBACK_SYNTH_PHRASES_AR = (
    "لخص", "تحدث عن مضمون", "المضمون العام", "يتعلق بموضوع",
)

# Questions canoniques pour le repli par similarité d'embedding. Les seuils
# (_EMBED_AGG_THRESHOLD / _EMBED_SYNTH_THRESHOLD) sont calibrés sur ces
# exemples : score cosinus entre la requête et l'exemple le plus proche.
_FALLBACK_AGG_EXAMPLES = {
    "fr": (
        "quels sont les decrets les plus importants",
        "combien d'articles comporte le decret numero 2 25 1080",
        "liste des arretes publies recemment",
        "combien de lois ont ete adoptees cette annee",
        "quelles sont les dernieres decisions parues",
        "les textes les plus importants en matiere de transport",
    ),
    "ar": (
        "ما هي المراسيم المهمة",
        "كم عدد مواد المرسوم رقم 2.24.874",
        "أهم الظهائر الصادرة",
        "قائمة القوانين الجديدة",
    ),
}
_FALLBACK_SYNTH_EXAMPLES = {
    "fr": (
        "resume ce document",
        "de quoi parle ce bulletin officiel",
        "quels sont les themes principaux de ce texte",
        "compare le decret et l'arrete",
        "quelle est la difference entre ces deux lois",
        "quel est le propos de ce texte",
    ),
    "ar": (
        "لخص هذا الوثيقة",
        "عن ماذا يتحدث هذا المرسوم",
        "قارن بين المرسومين",
        "الفرق بين القانونين",
        "ماذا يتضمن هذا النص",
    ),
}

_EMBED_AGG_THRESHOLD = 0.40
_EMBED_SYNTH_THRESHOLD = 0.45

# Cache des vecteurs d'exemples par (type de repli, langue, instance
# d'embedder) : l'embedding des exemples est coûteux et immuable.
_EMBED_CACHE: dict[tuple, object] = {}
_EMBED_CACHE_LOCK = threading.Lock()

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


def _log_decision(route: dict, query: str, lang: str | None) -> None:
    """Trace la décision d'aiguillage (uniquement en mode ADLI_ROUTING_DEBUG)."""
    if not _DEBUG:
        return
    _LOGGER.debug(
        "routing | lang=%s signal=%s catalog=%s type=%s scope=%s year=%s refs=%s | query=%r",
        lang, route.get("signal"), route.get("catalog"), route.get("type"),
        route.get("scope"), route.get("year"), route.get("references"), query,
    )


def _example_vectors(kind: str, lang: str, embed_fn):
    """Vecteurs des questions canoniques (cache par instance d'embedder)."""
    key = (kind, lang, id(embed_fn.__self__))
    with _EMBED_CACHE_LOCK:
        cached = _EMBED_CACHE.get(key)
        if cached is not None:
            return cached
    examples = (
        _FALLBACK_AGG_EXAMPLES if kind == "agg" else _FALLBACK_SYNTH_EXAMPLES
    ).get(lang, ())
    try:
        vectors = np.stack([embed_fn(e) for e in examples])
    except Exception:
        vectors = None
    with _EMBED_CACHE_LOCK:
        _EMBED_CACHE[key] = vectors
    return vectors


def _embedding_hint(q: str, lang: str, embed_fn):
    """Similarités cosinus (max) de la requête contre les exemples canoniques
    d'agrégation et de synthèse. (0.0, 0.0) si l'embedding échoue."""
    if embed_fn is None:
        return (0.0, 0.0)
    try:
        qvec = np.asarray(embed_fn(q)).ravel()
    except Exception:
        return (0.0, 0.0)
    agg_sim = 0.0
    agg_vecs = _example_vectors("agg", lang, embed_fn)
    if agg_vecs is not None:
        agg_sim = float(np.max(np.dot(agg_vecs, qvec)))
    synth_sim = 0.0
    synth_vecs = _example_vectors("synthesis", lang, embed_fn)
    if synth_vecs is not None:
        synth_sim = float(np.max(np.dot(synth_vecs, qvec)))
    return (agg_sim, synth_sim)


def _fallback_classify(q: str, lang: str, embed_fn):
    """Repli bas-coût pour une question restée « plate » (pas d'instrument,
    pas de signal de synthèse lexical). Renvoie (route_partielle, signal).

    Ordre : heuristique d'agrégation (protégée par le garde-fou des noms de
    corpus) → heuristique de synthèse → similarité d'embedding (synthèse
    d'abord : « quels sont les thèmes principaux de ce texte » ressemble à
    de l'agrégation mais doit partir en synthèse).
    """
    nouns = _CORPUS_NOUNS_AR if lang == "ar" else _CORPUS_NOUNS_FR
    for p in _FALLBACK_AGG_PHRASES_AR if lang == "ar" else _FALLBACK_AGG_PHRASES_FR:
        if p in q and any(n in q for n in nouns):
            return {"catalog": True}, f"fallback:agg:phrase:{p}"
    for p in _FALLBACK_SYNTH_PHRASES_AR if lang == "ar" else _FALLBACK_SYNTH_PHRASES_FR:
        if p in q:
            return {"scope": "synthesis"}, f"fallback:synthesis:phrase:{p}"
    agg_sim, synth_sim = _embedding_hint(q, lang, embed_fn)
    if synth_sim >= _EMBED_SYNTH_THRESHOLD:
        return {"scope": "synthesis"}, f"fallback:synthesis:embed:{synth_sim:.2f}"
    if agg_sim >= _EMBED_AGG_THRESHOLD:
        return {"catalog": True}, f"fallback:agg:embed:{agg_sim:.2f}"
    return {}, "none"


def route_query(query: str, lang: str | None = None, embed_fn=None) -> dict:
    """
    Renvoie {"catalog": bool, "type": str|None, "year": int|None,
    "scope": str|None, "references": list[str], "signal": str}.

    « catalog == True » signifie que la question doit être traitée par le
    catalogue d'instruments (agrégation / référence numérique), sinon elle
    suit le chemin sémantique classique.

    « scope == "synthesis" » signifie que la question porte sur une vue
    d'ensemble du contenu (résumé, thèmes, comparaison, structure) et doit
    être traitée par le mode synthèse (vérification d'ancrage allégée).

    « signal » documente la règle qui a pris la décision : "none" (repli
    silencieux), un signal lexical (instrument + agrégation), ou une règle
    du repli bas-coût (« fallback:agg:phrase:... », « fallback:synthesis:
    embed:0.52 », ...).

    embed_fn (optionnel) : fonction d'embedding de requête (même contrat
    que Embedder.embed_query) pour le repli par similarité ; si None, seul
    le repli heuristique de phrases s'applique.
    """
    q = _norm(query).strip()
    lang = lang or _guess_lang(query)

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

    route: dict = {
        "catalog": False,
        "type": matched_type,
        "year": None,
        "scope": "synthesis" if _is_synthesis_query(q, lang) else None,
        "references": _extract_references(q, lang),
        "signal": "none",
    }
    if not q:
        _log_decision(route, query, lang)
        return route

    if matched_word is not None:
        has_ref = bool(
            (AR_REF_RE if lang == "ar" else FR_REF_RE).search(q)
            or re.search(r"(?:19|20)\d{2}(?:[-.][0-9]{1,4}){1,2}", q)
        )
        m_year = YEAR_RE.search(q)
        route["year"] = (
            int(m_year.group(0)) if m_year and len(m_year.group(0)) == 4 else None
        )
        plural = matched_word in _PLURAL_SETS.get(lang, set())
        signals = _AGG_SIGNALS_AR if lang == "ar" else _AGG_SIGNALS_FR
        route["catalog"] = bool(
            has_ref
            or route["year"] is not None
            or plural
            or any(s in q for s in signals)
        )

    # Repli bas-coût : uniquement pour les questions restées « plates »
    # (ni catalogue, ni synthèse lexicale) — pas de raison de le tenter sur
    # les autres chemins, déjà décidés par le routeur lexical.
    if not route["catalog"] and route["scope"] is None:
        partial, signal = _fallback_classify(q, lang, embed_fn)
        route.update(partial)
        route["signal"] = signal

    _log_decision(route, query, lang)
    return route