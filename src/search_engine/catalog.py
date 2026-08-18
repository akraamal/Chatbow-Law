"""
src/search_engine/catalog.py
-------------------------------
Catalogue d'instruments juridiques (Dahirs, Décrets, Arrêtés, Lois, Décisions...)
extrait des JSON enrichis de data/annotated/.

C'est la couche « structurée » qui complète la recherche sémantique :
là où FAISS ne peut répondre qu'à des questions factuelles sur des
extraits, le catalogue répond aux questions AGRÉGÉES du type
« les dahirs les plus importants », « les décrets de 2024 »,
« combien d'articles comporte le décret n° 2-25-1080 ? » :
filtrage par type / année / référence + score d'importance.

Le catalogue est produit par build_catalog() et persiste dans
data/index/catalog.json (à côté de l'index FAISS), chargé par
LegalRAGChatbot.

Chaque entrée contient le texte de son préambule dans le champ
« text » (title + preamble) : la vérification mécanique des citations
du LLM (src/rag/citation_verifier.py) fonctionne donc à l'identique
sur les réponses « catalogue ».
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

DEFAULT_ANNOTATED_DIR = "data/annotated"
DEFAULT_INDEX_DIR = "data/index"

CATALOG_FILENAME = "catalog.json"
CATALOG_VERSION = 1

# --- Détection du type d'instrument --------------------------------------

FR_TYPE_RE = re.compile(
    r"\b(dahir|décret|arrêté|loi|décision|avis|instruction|ordonnance)\b",
    re.IGNORECASE,
)
FR_CANONICAL = {
    "dahir": "Dahir", "décret": "Décret", "arrêté": "Arrêté",
    "loi": "Loi", "décision": "Décision", "avis": "Avis",
    "instruction": "Instruction", "ordonnance": "Ordonnance",
}
AR_TYPE_MAP = {
    "ظهير": "Dahir", "مرسوم": "Décret", "قانون": "Loi",
    "قرار": "Arrêté", "مقرر": "Décision", "تعليمات": "Instruction",
    "أمر": "Ordonnance",
}
AR_TYPE_RE = re.compile(r"(ظهير\s*شريف|ظهير|مرسوم|قانون|قرار|مقرر|تعليمات|أمر)")

FR_REF_RE = re.compile(r"\bn\s*[°o]?\s*([0-9]+(?:[\-.][0-9]+)+)")
AR_REF_RE = re.compile(r"رقم\s*([0-9٠-٩]+(?:[\-.][0-9٠-٩]+)+)")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Signaux « poids » d'un instrument : modification d'un autre texte,
# statut fondamental (loi organique, charte, code).
MODIFICATION_RE = re.compile(
    r"modifi|abroge|remplace|complétant|proroge|"
    r"يعدل|يلغي|يحل محل|يتمم|تعديل",
    re.IGNORECASE,
)
FOUNDATIONAL_RE = re.compile(
    r"organique|charte|civil|pénal|commercial|code des|code de|"
    r"مدونة|قانون تنظيمي|ميثاق",
    re.IGNORECASE,
)

# --- Normalisation --------------------------------------------------------

_ACCENTS_COMBINING = dict.fromkeys(
    ord(c) for c in unicodedata.normalize("NFD", "àâäéèêëîïôöùûüç")
    if unicodedata.category(c) == "Mn"
)


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).translate(_ACCENTS_COMBINING)


def _norm(text: str) -> str:
    if any("\u0600" <= c <= "\u06FF" for c in text):
        # NFD décomposerait les lettres hamza/inchoatives arabes (أ → ا +
        # U+0654) et casserait la correspondance avec le texte source.
        return text.lower()
    return _strip_accents(text).lower()


def _to_western_digits(s: str) -> str:
    return s.translate(_AR_DIGITS)


def _guess_lang(query: str) -> str:
    return "ar" if any("\u0600" <= c <= "\u06FF" for c in query) else "fr"


# --- Extraction par instrument -------------------------------------------

def _detect_type(text: str, lang: str) -> str:
    if lang == "ar":
        m = AR_TYPE_RE.search(text)
        if not m:
            return "Autre"
        return AR_TYPE_MAP.get(m.group(1), "Autre")
    m = FR_TYPE_RE.search(text)
    if not m:
        return "Autre"
    return FR_CANONICAL.get(m.group(1).lower(), "Autre")


def _extract_reference(text: str, lang: str) -> str:
    m = (AR_REF_RE if lang == "ar" else FR_REF_RE).search(text)
    if not m:
        return ""
    return _to_western_digits(m.group(1))


def _extract_year(text: str) -> int | None:
    m = YEAR_RE.search(text)
    return int(m.group(0)) if m else None


def _digits_only(text: str) -> str:
    return _to_western_digits(re.sub(r"\D", "", text or ""))


# --- Score d'importance ---------------------------------------------------

def compute_importance(entry: dict, ymin: int | None, ymax: int | None) -> int:
    """
    Heuristique d'importance (0-100) :
      - taille : nombre d'articles (log2) — un texte qui régit beaucoup
        de domaines vaut plus qu'un texte technique à 2 articles ;
      - portée : modifie/abroge un autre texte, ou statut fondamental
        (loi organique, charte, code) ;
      - actualité : recence relative dans le corpus.
    """
    score = 20.0
    n = max(int(entry.get("n_articles") or 1), 1)
    score += min(35.0, math.log2(n) * 9.0)
    hay = f"{entry.get('title') or ''} {entry.get('preamble') or ''}"
    if MODIFICATION_RE.search(hay):
        score += 15.0
    if FOUNDATIONAL_RE.search(hay):
        score += 10.0
    year = entry.get("year")
    if year and ymin is not None and ymax is not None:
        if ymax > ymin:
            score += 10.0 * (year - ymin) / (ymax - ymin)
        else:
            score += 5.0
    elif year is None:
        score += 2.0  # date inconnue : ni pénalisé ni favorisé
    return round(min(100.0, score))


# --- Déduplication des fichiers -------------------------------------------

def _canonical_rank(filename: str) -> int:
    """Les pipelines d'analyse produisent des doublons (_old/_new/_vN/
    suffixe hexa de 8 chiffres). Le fichier « nu » (nom plein) est la
    référence canonique."""
    if "_old" in filename:
        return 1
    if "_new" in filename:
        return 2
    if re.search(r"_v\d", filename):
        return 3
    if re.search(r"_[0-9a-f]{8}", filename):
        return 4
    return 0


# --- Construction ----------------------------------------------------------

def build_catalog(annotated_dir: str = DEFAULT_ANNOTATED_DIR) -> list[dict]:
    """
    Construit le catalogue des instruments à partir des JSON enrichis.
    Déduplique les fichiers redondants du même (langue, bulletin).
    """
    files = sorted(Path(annotated_dir).glob("**/*_entities.json"))
    docs: dict[tuple[str, str], tuple[str, dict]] = {}

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data.get("decrees") or not data.get("articles"):
            continue
        lang = data.get("lang") or "fr"
        bo = str(data.get("bo_number") or "")
        key = (lang, bo or p.stem)
        current = docs.get(key)
        rank, cand_rank = (
            _canonical_rank(current[0]) if current else 99,
            _canonical_rank(p.name),
        )
        n_arts = len(data["articles"])
        if current is None or cand_rank < rank or (
            cand_rank == rank and n_arts > len(current[1]["articles"])
        ):
            docs[key] = (p.name, data)

    entries: list[dict] = []
    for (lang, bo), (fname, data) in sorted(docs.items()):
        articles = data["articles"]
        decrees = sorted(
            [d for d in data.get("decrees", []) if isinstance(d, dict) and d.get("title")],
            key=lambda d: d.get("first_article_idx", 0),
        )
        idxs = sorted({
            d["first_article_idx"] for d in decrees
            if isinstance(d.get("first_article_idx"), int)
            and 0 <= d["first_article_idx"] < len(articles)
        })
        for i, decree in enumerate(decrees):
            idx = decree.get("first_article_idx")
            if not isinstance(idx, int) or not (0 <= idx < len(articles)):
                continue
            nxt = next((x for x in idxs if x > idx), None)
            n = (nxt - idx) if nxt is not None else len(articles) - idx
            n = max(1, n)

            text_src = " ".join(
                x for x in (decree.get("title") or "", decree.get("preamble") or "") if x
            ).strip()
            if len(text_src) < 30:
                continue

            art_slot = articles[idx]
            entries.append({
                "instrument_id": f"{bo}_{i}",
                "type": _detect_type(text_src, lang),
                "reference": _extract_reference(text_src, lang),
                "title": decree.get("title") or "",
                "lang": lang,
                "bo_number": bo,
                "doc_id": data.get("doc_id", fname.replace("_entities", "")),
                "date_publication": data.get("date_publication") or "",
                "year": _extract_year(text_src),
                "n_articles": n,
                "pdf_page": art_slot.get("pdf_page"),
                "printed_page": art_slot.get("printed_page"),
                "article_number": art_slot.get("number", ""),
                "preamble": text_src[:1200],
                "importance": 0,
            })

    years = [e["year"] for e in entries if e["year"]]
    ymin, ymax = (min(years), max(years)) if years else (None, None)
    for e in entries:
        e["importance"] = compute_importance(e, ymin, ymax)
        e["text"] = f"{e['title'] or ''}\n{e['preamble']}"
    return entries


def save_catalog(entries: list[dict], index_dir: str = DEFAULT_INDEX_DIR) -> Path:
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / CATALOG_FILENAME
    path.write_text(
        json.dumps({"version": CATALOG_VERSION, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_catalog(index_dir: str = DEFAULT_INDEX_DIR) -> list[dict] | None:
    path = Path(index_dir) / CATALOG_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != CATALOG_VERSION:
            return None
        return [dict(e, score=0.0) for e in data.get("entries", [])]
    except Exception:
        return None


# --- Recherche dans le catalogue -------------------------------------------

# Mots « de structure » de la question : inutiles pour scorer la pertinence.
AGGREGATE_WORDS_FR = frozenset(_norm(w) for w in (
    "les plus", "plus importants", "plus importantes", "importants",
    "importantes", "majeurs", "majeures", "principaux", "principales",
    "récents", "récentes", "liste", "quels", "quelles", "quelles sont",
    "quels sont", "tous", "toutes", "ensemble", "nombre", "adoptés",
    "publiés", "parus", "importante", "important",
))
AGGREGATE_WORDS_AR = frozenset((
    "أهم", "كل", "قائمة", "ما هي", "الأهم", "المهمة", "الرئيسية",
    "الصادرة", "المعتمدة", "المصادق", "المهم",
))
STOPWORDS_FR = frozenset(_norm(w) for w in (
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou",
    "en", "au", "aux", "est", "sont", "qui", "que", "quel", "quelle",
    "pour", "avec", "sur", "dans", "par", "comment", "combien", "donne",
    "moi", "citer", "cite", "enumerer", "énumérer", "recense", "lister",
    # Noms d'instruments : le filtre de type s'en charge, ils n'apportent
    # rien au scoring lexical (et gâcheraient le classement par importance).
    "dahir", "dahirs", "décret", "décret", "decret", "décrets", "decrets",
    "loi", "lois", "arrêté", "arrete", "arrêtés", "arretes",
    "décision", "decision", "décisions", "decisions", "avis",
    "ordonnance", "ordonnances", "texte", "textes", "instrument",
    "instruments", "norme", "normes",
))
# Miroir arabe : formes de types d'instruments (le filtre de type les gère).
STOPWORDS_AR = frozenset((
    "مرسوم", "المرسوم", "مراسيم", "المراسيم",
    "قانون", "القانون", "قوانين", "القوانين",
    "قرار", "القرار", "قرارات", "القرارات",
    "مقرر", "المقرر", "مقررات", "المقررات",
    "ظهير", "الظهير", "شريف", "ظهائر", "الظهائر",
    "نصوص", "النصوص", "تشريعات", "التشريعات",
))


def _tokenize(query: str, lang: str) -> list[str]:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", _norm(query))
    if lang == "ar":
        return [t for t in tokens if t not in AGGREGATE_WORDS_AR and t not in STOPWORDS_AR]
    out = []
    for t in tokens:
        if t in STOPWORDS_FR or t in AGGREGATE_WORDS_FR or (
            t.isdigit() and len(t) == 4
        ):
            continue
        if t.endswith("s") and len(t) > 3:  # pluriel français approximatif
            t = t[:-1]
        out.append(t)
    return out


def search_catalog(
    entries: list[dict],
    query: str,
    type_filter: str | None = None,
    year: int | None = None,
    lang: str | None = None,
    top_n: int = 8,
) -> list[dict]:
    """
    Renvoie les instruments correspondant à la question, triés par
    (pertinence, importance). Filtres : type d'instrument (Dahir, Décret…),
    année de publication, langue.

    Le score sémantique « lexical » (mots de la question dans le préambule)
    ne compte que comme un bonus : le poids principal vient du filtre de
    type/référence/année — sinon « les dahirs les plus importants » ne
    pourrait jamais retrouver des dahirs dont les mots ne figurent pas
    dans la question.
    """
    q_lang = _guess_lang(query) if lang is None else lang
    tokens = _tokenize(query, q_lang)
    ref_digits = _digits_only(query)

    scored: list[tuple[float, float, dict]] = []
    for e in entries:
        if lang and e.get("lang") != lang:
            continue
        if type_filter and e.get("type") != type_filter:
            continue
        if year is not None and e.get("year") != year:
            continue

        score = 0.0
        if type_filter:
            score += 50.0
        else:
            score += 20.0  # question générique : la pertinence = importance
        if year is not None and e.get("year") == year:
            score += 40.0
        ref = _digits_only(e.get("reference") or "")
        if len(ref_digits) >= 3 and ref and ref in ref_digits:
            score += 60.0

        if tokens:
            hay = _norm(f"{e.get('title') or ''} {e.get('preamble') or ''}")
            hits = sum(1 for t in tokens if t in hay)
            score += min(30.0, hits * 6.0)

        if score > 0:
            scored.append((score, float(e.get("importance") or 0), e))

    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [
        dict(e, score=round(t[0], 2))
        for t in scored[:top_n]
        for e in [t[2]]
    ]