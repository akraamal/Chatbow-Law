"""
src/rag/citation_verifier.py
---------------------------------
Vérification mécanique (caractère par caractère) des citations produites
par le LLM.

Principe (adapté de docuchat.app) : après la génération, on ne fait PAS
confiance aux guillemets du modèle — chaque citation est retrouvée
concrètement dans le chunk dont elle prétend provenir, et la page / le
span affiché est dérivé de la correspondance vérifiée, jamais d'une
assertion du modèle. Toute citation qui échoue est silencieusement
supprimée : elle n'est jamais montrée à l'utilisateur.

Pipeline de vérification (par citation) :
  1. Correspondance exacte case-sensitive dans le texte brut du chunk
     (offset réel retourné).
  2. Correspondance en texte « normalisé » (minuscules, apostrophes,
     suppression des guillemets « » interlacés — artefact de colonnes
     PDF —, fusion de l'espace blanc, normalisation des lettres arabes
     : alef, tachkeel, tatweel, chiffres). L'offset est renvoyé en
     coordonnées du texte brut via une table de correspondance.
  3. Correspondance « OCR-aware » (dernier recours) : on applique aux
     deux côtés le correcteur du corpus (ocr_corrector.py côté FR,
     clean_arabic_text() côté AR). Offsets non remontables ; exige une
     citation plus longue (les corrections multi-mots peuvent produire
     des faux positifs sur les passages courts).

Le format de sortie attendu du LLM est un bloc machine-parsable, neutre
en langue (FR et AR) :

    [[CITATIONS]]
    «texte mot à mot» [Source 2]
    «texte mot à mot» [Source 1]
    [[END]]
"""
from __future__ import annotations

import re
import unicodedata

# --- Constantes ----------------------------------------------------------

# Une citation plus courte que ça (en caractères normalisés) ne porte pas
# assez d'information pour être significative ("Article 3." n'est pas une
# citation vérifiable) — supprimée pour éviter les faux positifs triviaux.
MIN_QUOTE_CHARS = 12

# Longueur minimale supplémentaire pour accepter une correspondance obtenue
# par correction OCR (risque de faux positifs sur les passages courts).
MIN_OCR_QUOTE_CHARS = 40

CITATIONS_BLOCK_RE = re.compile(
    r"\[\[\s*CITATIONS\s*\]\](.*?)\[\[\s*END\s*\]\]", re.DOTALL | re.IGNORECASE
)
SOURCE_MARKER_RE = re.compile(r"\[?\s*source\s*(\d+)\s*\]?", re.IGNORECASE)

_GUILLEMETS = "«»“”„\""
_CONTROL_CC_CF = ("Cc", "Cf", "Co", "Cn")

# Tachkeel (diacritiques arabes) + tatweel (U+0640)
_AR_MARKS = frozenset("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0640")
# Variantes d'alef / lettre finales souvent normalisées
_ALEF_MAP = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ي"}
_AR_DIGITS = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
              "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}


# --- Normalisation -------------------------------------------------------

def _strip_ar_char(ch: str) -> str | None:
    """Renvoie la forme normalisée d'un caractère arabe (None = supprimé)."""
    if ch in _AR_MARKS:
        return None
    ch = unicodedata.normalize("NFC", ch)
    return _ALEF_MAP.get(ch, _AR_DIGITS.get(ch, ch))


def _mapped_norm(text: str, lang: str) -> tuple[str, list[int]]:
    """
    Construit (texte_normalisé, offsets) où offsets[i] = index dans le texte
    BRUT du caractère texte_normalisé[i]. Un seul espace sépare les mots.
    Les caractères supprimés par la normalisation n'apparaissent pas.
    """
    src = unicodedata.normalize("NFC", text)
    if lang != "ar":
        # Substitutions 1:1 en longueur, faites AVANT le parcours pour que
        # les indices restent synchrones.
        src = src.replace("’", "'").replace("‘", "'").replace("`", "'").lower()

    out: list[str] = []
    offs: list[int] = []
    for i, ch in enumerate(src):
        if lang == "ar":
            stripped = _strip_ar_char(ch)
            if stripped is None:
                continue
            ch = stripped
        else:
            if ch in _GUILLEMETS:
                continue
        # Attention à l'ordre : \n, \r, \t ont la catégorie Unicode "Cc"
        # (control) MAIS sont aussi des espaces. Il faut les traiter comme
        # de l'espace (fusion) AVANT le filtre des caractères de contrôle,
        # sinon les sauts de ligne du chunk/fear de PDF collent les mots.
        if ch.isspace():
            if out and out[-1] == " ":
                continue
            out.append(" ")
            offs.append(i)
            continue
        if unicodedata.category(ch) in _CONTROL_CC_CF:
            continue
        out.append(ch)
        offs.append(i)

    if out and out[-1] == " ":
        out.pop()
        offs.pop()
    return "".join(out), offs


def _norm_fr(text: str) -> str:
    norm, _ = _mapped_norm(text, "fr")
    return norm


def _norm_ar(text: str) -> str:
    norm, _ = _mapped_norm(text, "ar")
    return norm


def _norm_ocr(text: str, lang: str) -> str:
    """Normalisation « poussée » par correction OCR (offsets perdus)."""
    if lang == "ar":
        from src.preprocessing.cleaner_ar import clean_arabic_text
        return clean_arabic_text(text, remove_headers=False, fix_hijri_gregorian_parens=False)
    from src.extraction.ocr_corrector import correct_ocr
    return _norm_fr(correct_ocr(text))


def normalize_and_find(chunk_text: str, quote: str, lang: str = "fr") -> dict:
    """
    Retrouve `quote` dans `chunk_text`.

    Renvoie :
        {"char_start": int|None, "char_end": int|None, "exact": bool,
         "normalized": bool, "matched_quote": str}
    `char_start/char_end` sont en coordonnées du texte BRUT quand la
    correspondance est remontable (None en mode OCR).
    """
    # 1. Exact brut
    idx = chunk_text.find(quote)
    if idx != -1:
        return {"char_start": idx, "char_end": idx + len(quote),
                "exact": True, "normalized": False, "matched_quote": quote}

    # Garde de longueur (en normalisé)
    norm_q = _norm_fr(quote) if lang != "ar" else _norm_ar(quote)
    if not norm_q or len(norm_q) < MIN_QUOTE_CHARS:
        return {"char_start": None, "char_end": None, "exact": False,
                "normalized": True, "matched_quote": ""}

# 2. Normalisé remontable (coord. brutes)
    # Les lignes de tableau linéarisées commencent par "1. ", "2. " etc. :
    # le LLM les omet souvent dans la citation. On retire ce préfixe de
    # la QUOTE normalisée seulement — le "1. " reste dans le texte source,
    # mais la sous-chaîne "conditionnement: ..." s'y retrouve par find().
    norm_q = re.sub(r"^\d+[.)]\s+", "", norm_q)
    if not norm_q:
        return {"char_start": None, "char_end": None, "exact": False,
                "normalized": True, "matched_quote": ""}
    norm_chunk, offs = _mapped_norm(chunk_text, lang)
    i = norm_chunk.find(norm_q)
    if i != -1:
        if i + len(norm_q) - 1 < len(offs):
            start = offs[i]
            end = offs[i + len(norm_q) - 1]
            return {"char_start": start, "char_end": end + 1,
                    "exact": False, "normalized": True,
                    "matched_quote": chunk_text[start:end + 1]}
        return {"char_start": offs[i], "char_end": None,
                "exact": False, "normalized": True, "matched_quote": norm_q}

    # 3. OCR-aware (dernier recours, offsets non remontables)
    if len(norm_q) >= MIN_OCR_QUOTE_CHARS:
        chunk_ocr = _norm_ocr(chunk_text, lang)
        quote_ocr = _norm_ocr(quote, lang)
        j = chunk_ocr.find(quote_ocr)
        if j != -1:
            return {"char_start": None, "char_end": None, "exact": False,
                    "normalized": True, "ocr": True, "matched_quote": quote_ocr}

    return {"char_start": None, "char_end": None, "exact": False,
            "normalized": False, "matched_quote": ""}


def _outer_quote_start(before: str) -> int:
    """
    Retrouve le guillemet ouvrant EXTERNE du passage cité : on part du
    `marqueur` [Source N] et on remonte vers l'arrière en équilibrant « / ».
    Le premier `«` (en remontant) qui ferme l'équilibre ouvre le passage
    dans son intégralité — y compris quand la citation contient elle-même
    des guillemets imbriqués (fréquent dans le corpus : « la société « X »
    ... »).
    """
    depth = 0  # le `»` de fermeture de l'entrée est inclus dans `before`
    for i in range(len(before) - 1, -1, -1):
        ch = before[i]
        if ch == "»":
            depth += 1
        elif ch == "«":
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_citations(answer_text: str) -> tuple[str, list[dict]]:
    """
    Extrait le bloc [[CITATIONS]] ... [[END]] de `answer_text`.

    Renvoie (answer_text_sans_bloc, spans) où spans = [
        {"quote": str, "source": int}
    ].
    """
    m = CITATIONS_BLOCK_RE.search(answer_text)
    if not m:
        return answer_text, []
    block = m.group(1)
    after = answer_text[m.end():]
    clean = answer_text[: m.start()].rstrip()
    if after.strip():
        clean += "\n" + after.lstrip()

    spans = []
    for marker in SOURCE_MARKER_RE.finditer(block):
        before = block[: marker.start()].rstrip()
        start = _outer_quote_start(before)
        if start == -1:
            continue
        quote = before[start + 1:].strip().rstrip("»").strip()
        source = int(marker.group(1))
        if quote and source >= 1:
            spans.append({"quote": quote, "source": source})
    return clean, spans


def verify_citations(
    cited_spans: list[dict],
    retrieved_chunks: list[dict],
) -> tuple[list[dict], dict]:
    """
    Vérifie chaque citation et renvoie (verified, stats).

    retrieved_chunks : résultats de SemanticSearchEngine.search(), dans
        l'ordre — l'entrée i correspond à [Source i+1]. Chaque chunk doit
        contenir "text" (et idéalement article_id/doc_id/pdf_page/lang).

    verified : spans qui passent, enrichis de chunk_id/doc_id/page,
        char_start/char_end (coord. brutes), exact/normalized, matched_quote.
        Les spans qui échouent sont silencieusement supprimés.

    stats : {"claimed": int, "verified": int, "failed": int}.
    """
    verified: list[dict] = []
    failed = 0
    for span in cited_spans:
        src = span.get("source")
        if not src or src < 1 or src > len(retrieved_chunks):
            failed += 1
            continue
        chunk = retrieved_chunks[src - 1]
        # Le LLM génère à partir du contexte construit sur text_clean (texte +
        # tableaux linéarisés) quand disponible : c'est donc lui qu'on doit
        # vérifier, sinon une citation du tableau échouerait contre le texte
        # brut (qui ne contient pas les valeurs).
        text = (chunk.get("text_clean")
                or chunk.get("text")
                or chunk.get("raw_text")
                or "")
        lang = (chunk.get("lang") or "fr").lower()
        if text.strip():
            hit = normalize_and_find(text, span.get("quote", ""), lang=lang)
        else:
            hit = {"char_start": None}
        if hit["char_start"] is None:
            failed += 1
            continue
        verified.append({
            **span,
            "chunk_id": chunk.get("article_id", chunk.get("id")),
            "doc_id": chunk.get("doc_id"),
            "page": chunk.get("pdf_page"),
            "char_start": hit["char_start"],
            "char_end": hit["char_end"],
            "exact": hit["exact"],
            "normalized": hit["normalized"],
            "matched_quote": hit["matched_quote"],
            "verified": True,
        })

    stats = {
        "claimed": len(cited_spans),
        "verified": len(verified),
        "failed": failed,
    }
    return verified, stats


# --- Mode « synthèse » : vérification d'ancrage (existence) ----------------
# Contrairement aux citations mot à mot ([[CITATIONS]]), le mode synthèse
# demande au LLM un bloc [[GROUNDED-IN]] listant les numéros de sources
# réellement utilisées — la réponse étant une reformulation légitime, on ne
# vérifie que l'EXISTENCE de chaque source déclarée dans le contexte fourni.

GROUNDING_BLOCK_RE = re.compile(
    r"\[\[\s*GROUNDED-IN\s*\]\](.*?)\[\[\s*END\s*\]\]", re.DOTALL | re.IGNORECASE
)


def parse_grounding(answer_text: str) -> tuple[str, list[int]]:
    """Extrait le bloc [[GROUNDED-IN]] d'une réponse en mode synthèse."""
    m = GROUNDING_BLOCK_RE.search(answer_text)
    if not m:
        return answer_text, []
    block = m.group(1)
    after = answer_text[m.end():]
    clean = answer_text[: m.start()].rstrip()
    if after.strip():
        clean += "\n" + after.lstrip()
    return clean, [int(n) for n in re.findall(r"\d+", block)]


def verify_grounding(
    source_indices: list[int],
    retrieved_chunks: list[dict],
) -> tuple[list[int], dict]:
    """
    Vérification allégée pour le mode synthèse : contrairement à
    verify_citations() (correspondance mot à mot), on vérifie seulement que
    chaque numéro de source déclaré existe dans le contexte réellement
    fourni — la réponse elle-même est une reformulation légitime, pas une
    citation. Les données précises (règle 2 du prompt synthèse) ne sont
    PAS re-vérifiées ici mécaniquement.
    """
    valid = sorted({i for i in source_indices if 1 <= i <= len(retrieved_chunks)})
    stats = {
        "claimed": len(source_indices),
        "verified": len(valid),
        "failed": len(source_indices) - len(valid),
    }
    return valid, stats


# --- Vérification des données chiffrées (mode synthèse) ------------------
#
# Complète verify_grounding() (qui ne vérifie que l'EXISTENCE des sources
# citées, pas leur contenu) par un contrôle ciblé de la RÈGLE 2 du prompt
# de synthèse : les références (n° de décret/dahir/loi) et les années ne
# peuvent pas être paraphrasées, contrairement au reste d'une réponse de
# synthèse — elles doivent apparaître telles quelles dans le contexte
# fourni. Ne couvre PAS les dates complètes ni les petits nombres
# génériques ("3 articles") : trop de formes valides pour un contrôle
# regex fiable sans faux positifs.

_REF_NUMBER_RE = re.compile(r"\b[0-9]{1,2}(?:[-.][0-9]{1,4}){1,2}\b")
_YEAR_NUMBER_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def extract_numeric_claims(text: str) -> list[str]:
    """Extrait les références (ex. '2-25-1080') et années à 4 chiffres
    d'un texte — les seuls éléments chiffrés que la règle 2 du prompt de
    synthèse interdit de paraphraser."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _REF_NUMBER_RE.findall(text) + _YEAR_NUMBER_RE.findall(text):
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def verify_numeric_claims(answer_text: str, retrieved_chunks: list[dict]) -> dict:
    """
    Vérifie que chaque référence/année mentionnée dans `answer_text`
    apparaît textuellement dans AU MOINS UNE des sources fournies. Ne
    vérifie pas la source précise de provenance (contrairement à
    verify_citations) — juste l'existence du nombre quelque part dans le
    contexte réellement donné au LLM, ce qui suffit à détecter un numéro
    inventé ou déformé par la synthèse.

    Renvoie {"claimed": [...], "failed": [...]} — `failed` liste les
    tokens absents de toutes les sources (candidats hallucination).
    """
    claims = extract_numeric_claims(answer_text)
    if not claims:
        return {"claimed": [], "failed": []}
    haystack = "\n".join(
        (c.get("text_clean") or c.get("text") or "") for c in retrieved_chunks
    )
    return {"claimed": claims, "failed": [t for t in claims if t not in haystack]}