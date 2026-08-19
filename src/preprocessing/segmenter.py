"""
segmenter.py
--------------
Découpe un texte juridique nettoyé (voir cleaner_fr.py / cleaner_ar.py) en
articles individuels, en détectant les marqueurs "ARTICLE X" / "ART. X" en
français et "المادة" en arabe.

Basé sur les patterns réels observés dans le Bulletin Officiel marocain :
    "ART. 2. – En application des dispositions..."
    "ART. 5. – Le présent arrêté sera publié au Bulletin officiel."

Le format "ART. N. – texte" (avec un tiret cadratin/en-dash après le point)
est la convention dominante observée dans les décrets/arrêtés du BO.
"""

import re
from dataclasses import dataclass


@dataclass
class Article:
    number: str          # "1", "2", "premier"... tel qu'écrit dans le texte
    text: str            # contenu de l'article (sans le marqueur "ART. N. –")
    raw_header: str      # le marqueur original, ex: "ART. 2. –"
    start_pos: int        # position de début dans le texte source (utile pour audit)
    preamble: str = ""    # texte du préambule qui précède cet article (visas, considérants, DÉCRÈTE, etc.)


# Sécurité : aucun article ne doit dépasser cette taille (~7500 tokens).
# Un article plus long qu'un arrêté normal est soit une annexe entière qui
# n'a pas été correctement découpée (pattern "ANNEXE" manquant), soit un
# artefact d'extraction. On le scinde en sous-parties.
MAX_ARTICLE_CHARS = 30_000


# Patterns de marqueurs structurels pour les documents sans articles (avis,
# annexes, appendices) — on détecte des sous-sections à l'intérieur d'un
# article trop long, pour le découper en chunks sémantiquement cohérents.
_STRUCTURAL_HEADING_RE = re.compile(
    r"\n(?:"
    r"Chapitre\s+(?:[IVXL]+|\d+)|"
    r"Section\s+\d+|"
    r"Paragraphe\s+\d+|"
    r"TITRE\s+(?:[IVXL]+|\d+)"
    r")\b[.\s:]",
    re.IGNORECASE,
)

_ROMAN_NUMERAL_RE = re.compile(
    r"\n([IVXL]+)\.[\s]",
)

_NUMBERED_LIST_RE = re.compile(
    r"\n(\d{1,2})\.[\s](?=[A-Z\u00C0-\u024F])",
)


def _split_by_structural_markers(article):
    """
    Détecte les marqueurs structurels (Chapitre, Section, titres en chiffres
    romains, listes numérotées) dans le texte d'un article et le scinde à
    ces positions.  Renvoie une liste d'Articles.

    Si moins de 2 marqueurs sont trouvés, renvoie [article] inchangé.
    """
    text = article.text
    markers = []

    for m in _STRUCTURAL_HEADING_RE.finditer(text):
        markers.append(m.start())

    for m in _ROMAN_NUMERAL_RE.finditer(text):
        markers.append(m.start())

    for m in _NUMBERED_LIST_RE.finditer(text):
        markers.append(m.start())

    if len(markers) < 2:
        return [article]

    markers.sort()
    chunks = []
    prev = 0
    for pos in markers:
        chunk_text = text[prev:pos].strip()
        if chunk_text:
            chunks.append(chunk_text)
        prev = pos
    last = text[prev:].strip()
    if last:
        chunks.append(last)

    if not chunks:
        return [article]

    return [
        Article(
            number=article.number + (f"-{i+1}" if i else ""),
            text=chunk,
            raw_header=article.raw_header if i == 0 else "",
            start_pos=article.start_pos,
        )
        for i, chunk in enumerate(chunks)
    ]


# Seuil réduit pour les textes non structurés en articles (avis, annexes) :
# si le segment entre deux marqueurs ART. N. dépasse cette taille et qu'aucun
# marqueur structurel n'est trouvé, on force un découpage fixe.
NON_ARTICLE_CHUNK_SIZE = 5000


def _split_oversized_article(article, max_chars=MAX_ARTICLE_CHARS):
    """
    Découpe un article dont le texte dépasse `max_chars` en plusieurs
    sous-articles, en essayant les séparateurs naturels dans l'ordre :
      1. Marqueurs structurels (Chapitre, Section, listes numérotées, etc.)
      2. Sauts de page (\\f / \\x0c) — limites de page PDF
      3. Titres en capitales (TITRE, CHAPITRE, SECTION, ANNEXE)
      4. Dernière frontière de phrase avant max_chars
      5. Découpage fixe à NON_ARTICLE_CHUNK_SIZE pour les textes non
         structurés (avis, annexes)

    Returns:
        Liste d'objets Article (peut n'en contenir qu'un si le texte
        est déjà sous la limite ou insécable).
    """
    if len(article.text) <= max_chars:
        return [article]

    # 0) Try structural marker split first — catches avis, annexes,
    #    appendices that have no ART. N. markers but do have section
    #    headings.
    structural = _split_by_structural_markers(article)
    if len(structural) > 1:
        return structural

    chunks = []
    remaining = article.text

    while len(remaining) > max_chars:
        chunk = remaining[:max_chars]

        # 1) Try to split on form feed (PDF page break)
        ff_pos = chunk.rfind("\f")
        if ff_pos > max_chars // 2:
            head, remaining = remaining[:ff_pos], remaining[ff_pos:]
            chunks.append(head.strip())
            continue

        # 2) Try to split on an all-caps heading line (last occurrence).
        #    (Avant : le pattern était cherché dans chunk[::-1] — la chaîne
        #    inversée — donc ne matchait JAMAIS : branche morte.)
        caps_heading = None
        for m in re.finditer(r"\n(?:TITRE|CHAPITRE|SECTION|ANNEXE|APPENDICE)\b", chunk, re.IGNORECASE):
            caps_heading = m
        if caps_heading:
            pos = caps_heading.start()
            head, remaining = remaining[:pos], remaining[pos:]
            chunks.append(head.strip())
            continue

        # 3) Fallback: last sentence boundary before max_chars
        sentence_end = max(
            chunk.rfind(". "),
            chunk.rfind(".\n"),
            chunk.rfind("!\n"),
            chunk.rfind("?\n"),
        )
        if sentence_end > max_chars // 2:
            split_at = sentence_end + 1
            head, remaining = remaining[:split_at], remaining[split_at:]
            chunks.append(head.strip())
            continue

        # 4) Last resort: hard split at max_chars boundary
        chunks.append(chunk.strip())
        remaining = remaining[max_chars:]

    if remaining.strip():
        chunks.append(remaining.strip())

    # If the resulting chunks are still very large (> NON_ARTICLE_CHUNK_SIZE),
    # this is probably unstructured text (avis, opinion text, etc.).  Force a
    # finer-grained split at NON_ARTICLE_CHUNK_SIZE.
    if len(chunks) <= 2:
        refined = []
        for ch in chunks:
            if len(ch) > NON_ARTICLE_CHUNK_SIZE:
                for i in range(0, len(ch), NON_ARTICLE_CHUNK_SIZE):
                    refined.append(ch[i:i + NON_ARTICLE_CHUNK_SIZE].strip())
            else:
                refined.append(ch)
        chunks = refined

    return [
        Article(
            number=article.number + (f"-{i+1}" if i else ""),
            text=chunk,
            raw_header=article.raw_header if i == 0 else "",
            start_pos=article.start_pos,
        )
        for i, chunk in enumerate(chunks) if chunk
    ]


# --- Français ---
# Couvre "ART. 2. –", "Article 2 :", "ARTICLE PREMIER –", "art. 544-1.–", etc.
# Variante OCR "ART, 2. –" (virgule au lieu du point après ART — observée
# sur BO_6718/6758/6804/6822) : le `[.,]?` tolère les deux.
# The trailing (?![\dA-Za-z]) rejects false matches like "article 214-III-A"
# where the hyphen is part of a CGI subsection reference, not an article
# separator.  Matches that begin mid-sentence ("l'article ...") are also
# filtered out in segment_into_articles_fr.
# The separator is optional: many LOI-CADRE articles use "Article premier\\n"
# with no trailing punctuation.  The segment_into_articles_fr function filters
# out mid-sentence matches (preceded by a non-newline character).
ARTICLE_PATTERN_FR = re.compile(
    r"(ART(?:ICLE)?[.,]?\s*(?:PREMIER|1er|UNIQUE|\d+(?:-\d+)?)\s*\.?\s*[-–.:—]?(?![\dA-Za-z\u00C0-\u024F]))",
    re.IGNORECASE,
)

# Détecte le début du TITRE d'un nouveau texte juridique (Arrêté / Décret /
# Dahir / Décision) en tête de ligne.
#
# Bug corrigé : segment_into_articles_fr() découpait uniquement sur les
# marqueurs "ART. N. –". Comme un arrêté à 2 articles (très fréquent dans
# les séries "Equivalences de diplômes" et "Textes particuliers" du BO)
# n'a pas d'autre marqueur d'article entre son "ART. 2." et le titre du
# texte juridique SUIVANT, tout le titre + les visas + "ARRÊTE :" du texte
# suivant se retrouvaient absorbés dans le texte de l'article précédent
# (confirmé sur BO_7510_Fr : l'ART. 2 de l'arrêté n° 408-26 contenait déjà
# tout le titre de l'arrêté n° 409-26 qui le suit immédiatement).
#
# Dans le corpus du BO, une ligne qui commence directement par "Arrêté",
# "Décret", "Dahir" ou "Décision" est TOUJOURS le début du titre d'un
# nouveau texte — les références à un texte existant à l'intérieur d'un
# article sont systématiquement introduites par "Vu " ou apparaissent en
# milieu de phrase ("... du décret n°..."), jamais en début de ligne.
DOCUMENT_TITLE_PATTERN_FR = re.compile(
    r"^[ \t]*(?:Arr[eê]t[eé]|D[eé]cret|Dahir|D[eé]cision|R[eè]glement|Avis|Circulaire|Annexe)\b",
    re.MULTILINE | re.IGNORECASE,
)

# Marqueur de section du Bulletin officiel (« TEXTES GENERAUX »,
# « TEXTES PARTICULIERS ») : il sépare des rubriques entières du BO et
# ne fait pas partie du corps de l'article qui le précède.  Ex. le
# « TEXTES PARTICULIERS » qui suit l'arrêté n° 1197-26 (BO_7522) était
# absorbé en fin d'article.
SECTION_MARKER_RE = re.compile(
    r"^[ \t]*TEXTES[ \t]+(?:G[ÉE]N[ÉE]RAUX|PARTICULIERS)\b",
    re.MULTILINE,
)

# Équivalent arabe pour les titres de textes juridiques du BO (ظهير/مرسوم/
# قرار/قانون/أمر/منشور/بلاغ) en tête de ligne. L'arabe n'a pas de casse :
# l'ancrage en début de ligne (^) suffit à écarter les citations en milieu
# de phrase (« وكذا القانون رقم ... »).
DOCUMENT_TITLE_PATTERN_AR = re.compile(
    r"^[ \t]*(?:ظهير|مرسوم|قرار|مقرر|قانون|أمر|منشور|بلاغ)\b",
    re.MULTILINE,
)


def _is_doc_title_match(text: str, match, lang: str = "fr") -> bool:
    """Un match de DOCUMENT_TITLE_PATTERN est-il un VRAI titre de texte ?

    Français : un titre commence normalement par une majuscule
    (« Décret n° ... ») — les minuscules en tête de ligne sont en général
    des citations (« loi n° 17-99 portant ... »).  MAIS l'extraction PDF
    minusculise parfois de vrais titres (« décret n° 2-26-489 du ... ») :
    dans ce cas on accepte le titre si la ligne précédente ne se termine
    pas par une lettre minuscule — sinon c'est une continuation de phrase
    du type « Vu le\\ndécret n° 2-15-743 ... », qui ne doit PAS créer de
    frontière d'instrument.  Arabe : pas de casse, tout match est accepté.
    """
    if lang == "ar":
        return True
    title = match.group().strip()
    if not title:
        return False
    if title[0].isupper():
        return True
    line_start = text.rfind("\n", 0, match.start()) + 1
    if line_start == 0:
        return True  # tout début de texte : pas de contexte de continuation
    # dernier caractère NON-espace de la ligne précédente (le "\n" lui-même
    # est à line_start - 1)
    j = line_start - 2
    while j >= 0 and text[j] in " \t\r":
        j -= 1
    return j < 0 or not text[j].islower()


# --- Arabe ---
# Couvre "المادة 1." / "المادة الأولى" (première) / "المادة الثانية" (deuxième, rare
# car les textes juridiques utilisent presque toujours les chiffres au-delà de 1)
# Tous les nombres ordinaux arabes jusqu'à la dixième.
#
# Real OCR evidence (BO_7517_Ar, arrêtés 899.26–1043.26): "الأولى" is not
# just missing its hamza under OCR — the whole lam/hamza cluster gets
# transposed (observed: "االولى" instead of "الأولى"). A single added
# spelling doesn't cover it; needs a tolerant shape match instead. أ est
# dans la classe [اوأ] : sans lui, la forme standard «الأولى» (hamza sur
# l'alef) ne matcherait plus — vérifié contre les trois orthographes.
_AR_ORDINAL_FIRST = r"[اولأ]{2,5}لى"
_AR_ORDINALS = (
    rf"{_AR_ORDINAL_FIRST}|"
    r"الثانية|"
    r"الثالثة|"
    r"الرابعة|"
    r"الخامسة|"
    r"السادسة|"
    r"السابعة|"
    r"الثامنة|"
    r"التاسعة|"
    r"العاشرة"
)
ARTICLE_PATTERN_AR = re.compile(
    rf"""
    (
        (?:المادة|املادة)
        \s*
        (
            [0-9٠-٩]+                              # chiffres (latins ou arabes)
            |
            {_AR_ORDINALS}                          # première / deuxième / … / dixième
        )
        \s*
        [\.:\-–]?
    )
    """,
    re.VERBOSE | re.UNICODE,
)

# Suite d'en-tête typique d'une RÉFÉRENCE CROISÉE arabe en début de ligne
# (« المادة 2 منه؛ », « المادة 4 من القانون رقم ... ») — une citation
# dans un préambule, pas un vrai en-tête d'article.
_AR_XREF_RE = re.compile(r"^(?:منه|منها|اعلاه|ااعله|المشار اليه|من القانون|من المرسوم|من القرار|من الظهير|من النص)\b|^؛")

def normalize_arabic_digits(text: str) -> str:
    """Convertit les chiffres arabes (٠-٩) en chiffres latins (0-9)."""
    arabic_to_latin = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for ar, la in arabic_to_latin.items():
        text = text.replace(ar, la)
    return text



def _is_stray_fragment(text):
    """Un fragment est un texte qui n'a pas la structure d'un article
    autonome : vide, très court (< 15 caractères), ou commençant par
    une minuscule ou un chiffre (signe d'une continuation de phrase)."""
    if not text:
        return True
    if len(text) < 15:
        return True
    if text[0].islower() or text[0].isdigit():
        return True
    return False


def _merge_short_articles(articles, min_chars=50):
    """Fusionne les articles dont le texte est trop court ou fragmentaire
    dans l'article précédent.

    Ces fragments apparaissent quand un marqueur d'article mal formé
    (ex: "article 28-" où le `-` est en fait le début du nombre
    composé "28-1") est détecté comme un en-tête autonome avec un
    texte vide ou quasi vide, ou quand la colonne de page sépare
    un en-tête de son contenu.
    """
    if not articles or len(articles) < 2:
        return articles

    merged = []
    for art in articles:
        if merged and len(art.text) < min_chars and _is_stray_fragment(art.text):
            prev = merged.pop()
            merged_text = (prev.text + "\n" + art.raw_header + " " + art.text).strip()
            merged.append(Article(
                number=prev.number,
                text=merged_text,
                raw_header=prev.raw_header,
                start_pos=prev.start_pos,
            ))
        else:
            merged.append(art)
    return merged


def _extract_article_number_fr(header: str) -> str:
    """Extrait juste le numéro (ou 'premier') depuis l'en-tête matché en français."""
    match = re.search(r"(PREMIER|1er|UNIQUE|\d+(?:-\d+)?)", header, re.IGNORECASE)
    return match.group(1) if match else header.strip()


_ANNEXE_HEADER_RE = re.compile(r"^(Annexe\s*\d*)", re.MULTILINE | re.IGNORECASE)

def _extract_annexe_number(gap_text: str) -> str:
    """Extract a label from the first annexe/header line in gap text."""
    m = _ANNEXE_HEADER_RE.search(gap_text)
    if m:
        return m.group(1).strip()
    # Not an annexe header: return a generic label
    return "(annexe)"

def _extract_annexe_header(gap_text: str) -> str:
    """First meaningful line of the gap text."""
    for line in gap_text.split("\n"):
        line = line.strip()
        if line and len(line) > 3:
            return line
    return "(annexe)"


def _extract_article_number_ar(header: str) -> str:
    """Extrait le numéro et normalise les chiffres arabes."""
    m = re.search(
        rf"(?:المادة|املادة)\s*([0-9٠-٩]+|{_AR_ORDINALS})",
        header,
    )
    if m:
        number = m.group(1)
        # Normalisation des chiffres arabes
        return normalize_arabic_digits(number)
    return header.strip()


def _inside_guillemets(text: str, pos: int, lang: str = "fr") -> bool:
    """Check if *pos* is inside «...» (guillemet-quoted text).

    The pairing rule (last « before *pos* after the last ») is only
    reliable for the French edition.  The Arabic edition of the BO
    typesets multi-line quoted amendment blocks with the opening «
    repeated at the start of every line and closes with a single »
    (verified on BO_7517_Ar: all 16 multi-line quote blocks), and the
    closing » is sometimes lost under OCR — so the last-«-after-last-»
    pairing would swallow the *real* article marker that follows an
    unclosed block.  Since article markers are line-anchored and quoted
    markers always sit on a line that itself starts with « or », the
    robust Arabic rule is line-based: the marker is quoted iff its own
    line starts with a guillemet (works whichever of « / » is used as
    the per-line repetition character).
    """
    if lang == "ar":
        line_start = text.rfind("\n", 0, pos) + 1
        j = line_start
        while j < pos and text[j] in " \t":
            j += 1
        return j < pos and text[j] in '\u00AB\u00BB'
    last_ab = text.rfind('\u00AB', 0, pos)  # «
    last_bb = text.rfind('\u00BB', 0, pos)  # »
    return last_ab > last_bb       # « opens, » closes (French convention)


# Marqueurs de sommaire et de section, par langue du BO.  L'édition
# arabe utilise « فهرست » pour l'en-tête du sommaire et « نصوص عامة »
# / « نصوص خاصة » pour les rubriques de section.
SOMMAIRE_MARKERS = {"fr": "SOMMAIRE", "ar": "فهرست"}
SECTION_MARKERS = {
    "fr": ["TEXTES GÉNÉRAUX", "TEXTES GENERAUX", "TEXTES PARTICULIERS"],
    "ar": ["نصوص عامة", "نصوص خاصة"],
}


def _skip_sommaire(text: str, lang: str = "fr", limit: int | None = None) -> int:
    """Find the position where actual legal text starts, after the sommaire.

    Strategy: locate the sommaire heading (``SOMMAIRE`` / ``فهرست``), then
    find a section marker (``TEXTES GÉNÉRAUX`` / ``TEXTES GENERAUX`` /
    ``TEXTES PARTICULIERS`` / ``نصوص عامة`` / ``نصوص خاصة``) that appears
    **after** the sommaire content (500+ chars from the sommaire line).
    This avoids matching the marker inside the table-of-contents list.

    Un sommaire multi-pages (OCR 2 colonnes) contient souvent SES PROPRES
    en-têtes de section dans sa liste (« TEXTES PARTICULIERS » dans la
    table des matières, BO_6758) : le premier marqueur après
    ``sommaire_pos + 500`` peut donc tomber DANS le sommaire.  Règle
    retenue :

      * on préfère les marqueurs de type « GÉNÉRAUX » (« TEXTES G… »,
        « نصوص عامة ») — le corps des BO s'ouvre presque toujours par la
        rubrique générale ; les entrées du sommaire qui sont de type
        « PARTICULIERS » arrivent généralement en fin de table des
        matières (après le garde-fou des 500 caractères) ;
      * parmi les marqueurs du type préféré, on retient le DERNIER avant
        *limit* (typiquement le premier marqueur d'article) : en
        français, si le corps réel démarre par « TEXTES GÉNÉRAUX » mais
        que la liste du sommaire contient aussi « TEXTES GENERAUX »,
        c'est le marqueur le plus proche du premier article qui est le
        bon ;
      * en repli (aucun marqueur « GÉNÉRAUX »), dernier marqueur de
        n'importe quel type avant *limit*.

    Les marqueurs tronqués par l'extraction PDF (« TEXTES Gl » au lieu de
    « TEXTES GÉNÉRAUX ») sont acceptés tant que la ligne reste courte
    (≤ 30 caractères, ce qui exclut les entrées du sommaire fusionnées
    avec du texte).
    """
    sommaire_marker = SOMMAIRE_MARKERS.get(lang, SOMMAIRE_MARKERS["fr"])
    sommaire_pos = text.find(sommaire_marker)
    if sommaire_pos == -1:
        return 0
    search_from = sommaire_pos + 500
    if limit is None:
        limit = len(text)

    if lang == "ar":
        general_re = re.compile(r"^[ \t]*نصوص\s+عامة[^\n]*", re.MULTILINE)
        any_re = re.compile(r"^[ \t]*نصوص\s+(?:عامة|خاصة)[^\n]*", re.MULTILINE)
    else:
        general_re = re.compile(r"^[ \t]*TEXTES[ \t]+G[^\n]*", re.MULTILINE)
        any_re = re.compile(r"^[ \t]*TEXTES[ \t]+[GP][^\n]*", re.MULTILINE)

    def _candidates(pattern):
        return [
            m for m in pattern.finditer(text, search_from, limit)
            if len(m.group().strip()) <= 30
        ]

    candidates = _candidates(general_re) or _candidates(any_re)
    if candidates:
        return candidates[-1].end()

    # Repli rétrocompatible : marqueurs exacts.
    for marker in SECTION_MARKERS.get(lang, SECTION_MARKERS["fr"]):
        pos = text.find(marker, search_from)
        if pos != -1 and pos < limit:
            return pos + len(marker)
    return 0


def _filter_article_matches(text: str, lang: str = "fr") -> list:
    """Return filtered ARTICLE_PATTERN matches.

    Filters out:
    1. Mid-sentence matches (not at line start)
    2. Lowercase keyword (cross-references like "article 34")
    3. Matches inside guillemet-quoted text
    """
    pattern = ARTICLE_PATTERN_FR if lang == "fr" else ARTICLE_PATTERN_AR
    matches = list(pattern.finditer(text))

    filtered = []
    for m in matches:
        pos = m.start()
        # Must be at line start
        if not (pos == 0 or text[pos - 1] in "\n\r"):
            continue
        # Keyword must start with uppercase — French only. Arabic has no
        # case distinction: .isupper() is always False on Arabic letters,
        # which rejected every Arabic marker and made get_preamble() return
        # the whole document (and get_per_decree_preamble_map() nothing).
        if lang == "fr":
            match_text = m.group(1).strip()
            first_word = match_text.split()[0] if match_text else ""
            if not first_word or not first_word[0].isupper():
                continue
        # Must not be inside guillemets
        if _inside_guillemets(text, pos, lang=lang):
            continue
        if lang == "ar":
            # Arabic has no case distinction, so the uppercase check above
            # does not apply.  OCR line breaks put cross-references like
            # "المادة 2 منه؛" / "المادة 4 من القانون رقم ..." at line
            # start inside preambles (citations, not article headers):
            # filter them by what follows the header.
            rest = text[m.end():m.end() + 40].lstrip()
            if _AR_XREF_RE.match(rest):
                continue
        filtered.append(m)
    return filtered


def segment_into_articles_fr(text: str) -> list:
    """
    Découpe un texte français en articles, à partir des marqueurs "ART. N. –".

    Returns:
        Liste d'objets Article. Le texte AVANT le premier marqueur trouvé
        (préambule, visas, considérants) n'est pas inclus dans la liste — à
        récupérer séparément via get_preamble() si besoin.
    """
    matches = _filter_article_matches(text)

    articles = []
    pending_preamble = ""  # preamble deferred from a previous boundary

    for i, match in enumerate(matches):
        header = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Si le titre d'un nouveau texte juridique (Arrêté/Décret/Dahir/
        # Décision) apparaît avant le prochain marqueur d'article, il ne
        # fait pas partie de l'article courant : on coupe là, pas au
        # prochain "ART. N.–" (qui appartient au texte suivant).  Un
        # marqueur de section (TEXTES GENERAUX / PARTICULIERS) coupe
        # également : ce qui suit appartient aux rubriques suivantes.
        original_end = end
        boundary_match = None
        candidates = []
        for bm in DOCUMENT_TITLE_PATTERN_FR.finditer(text, start, end):
            if _is_doc_title_match(text, bm, "fr"):
                candidates.append(bm)
                break
        for sm in SECTION_MARKER_RE.finditer(text, start, end):
            candidates.append(sm)
            break
        if candidates:
            boundary_match = min(candidates, key=lambda m: m.start())
        if boundary_match:
            end = boundary_match.start()

        content = text[start:end].strip()

        articles.append(
            Article(
                number=_extract_article_number_fr(header),
                text=content,
                raw_header=header.strip(),
                start_pos=match.start(),
                preamble=pending_preamble,
            )
        )
        pending_preamble = ""

        # Capture gap content between the truncated boundary and the next
        # ART marker (or end of text).
        #
        # * If the boundary is "Annexe" → the gap is the annexe content
        #   itself (append as a separate Article chunk).
        # * Otherwise → the gap is the next decree's preamble; defer it
        #   to attach to the next article (it will be set above on the
        #   next loop iteration).
        if boundary_match and original_end > end:
            boundary_title = text[boundary_match.start():boundary_match.start() + 20]
            gap_text = text[end:original_end]
            if re.match(r"^\s*Annexe", boundary_title, re.IGNORECASE):
                # Le gap commence à l'en-tête "Annexe".  Il peut contenir
                # PLUSIEURS annexes consécutives (chacune devient une
                # entrée Article séparée), puis le titre de l'instrument
                # suivant : tout ce qui suit la dernière annexe est le
                # préambule de cet instrument et est reporté à l'article
                # suivant (champ `.preamble`).
                gap_matches = list(DOCUMENT_TITLE_PATTERN_FR.finditer(gap_text))
                pos = 0
                mi = 0
                while mi < len(gap_matches) and re.match(
                    r"^\s*Annexe", gap_matches[mi].group().strip(), re.IGNORECASE
                ):
                    next_pos = (gap_matches[mi + 1].start() if mi + 1 < len(gap_matches)
                                else len(gap_text))
                    annexe_text = gap_text[pos:next_pos].strip()
                    if len(annexe_text) > 200:
                        articles.append(
                            Article(
                                number=_extract_annexe_number(annexe_text),
                                text=annexe_text,
                                raw_header=_extract_annexe_header(annexe_text),
                                start_pos=end + pos,
                            )
                        )
                    pos = next_pos
                    mi += 1
                pending_preamble = gap_text[pos:].strip()
            else:
                # Not an annexe — this gap is the preamble of the next
                # decree/arrêté.  Defer it so the following article
                # receives it as its `.preamble` field.
                pending_preamble = gap_text.strip()

    articles = _merge_truncated_articles(_merge_short_articles(articles))
    return [chunk for a in articles for chunk in _split_oversized_article(a)]


def _merge_truncated_articles(articles, max_chars=100):
    """Fusionne les articles courts (< max_chars) dont le texte se termine
    sans ponctuation de fin de phrase avec l'article suivant si celui-ci
    commence par une minuscule (signe de continuation de phrase).

    Corrige les cas résiduels où l'extraction PDF a coupé une phrase au
    milieu. La limite max_chars évite les fausses fusions sur des articles
    longs dont le texte manque naturellement de point final (fréquent dans
    l'OCR du BO).
    """
    if not articles or len(articles) < 2:
        return articles

    merged = []
    i = 0
    while i < len(articles):
        if i + 1 < len(articles):
            cur = articles[i]
            nxt = articles[i + 1]
            cur_end = cur.text.rstrip()

            if (len(cur.text) < max_chars
                and len(nxt.text) < max_chars
                and cur_end
                and cur_end[-1] not in '.!?;'
                and nxt.text
                and nxt.text[0].islower()):
                merged_text = (cur.text + " " + nxt.text).strip()
                merged.append(Article(
                    number=cur.number,
                    text=merged_text,
                    raw_header=cur.raw_header,
                    start_pos=cur.start_pos,
                ))
                i += 2
                continue

        merged.append(articles[i])
        i += 1

    return merged


def segment_into_articles_ar(text: str) -> list:
    """
    Découpe un texte arabe OCRisé en articles.

    Compatible avec :
        المادة 1
        المادة ١
        المادة الأولى
        المادة الثانية
        املادة ...
    """
    # segment_into_articles_ar() bypassed the guillemet/line-start filter
    # that get_preamble()/get_per_decree_preamble_map() already apply via
    # _filter_article_matches() — so a quoted "«المادة الأولى...»" inside
    # amendment text (common: this arrêté cites the OLD article text it's
    # replacing) was creating a spurious article boundary. Route through
    # the same filter here.
    matches = _filter_article_matches(text, lang="ar")
    articles = []

    title_positions = [
        m.start() for m in DOCUMENT_TITLE_PATTERN_AR.finditer(text)
        if _is_doc_title_match(text, m, lang="ar")
    ]

    for i, match in enumerate(matches):
        header = match.group(1)
        start = match.end()
        next_art_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        next_title = next(
            (p for p in title_positions if start < p < next_art_start), None
        )
        end = next_title if next_title is not None else next_art_start
        content = text[start:end].strip()

        articles.append(
            Article(
                number=_extract_article_number_ar(header),
                text=content,
                raw_header=header.strip(),
                start_pos=match.start(),
            )
        )

    articles = _merge_truncated_articles(_merge_short_articles(articles))
    return [chunk for a in articles for chunk in _split_oversized_article(a)]


def get_preamble(text: str, lang: str = "fr") -> str:
    """
    Retourne le texte AVANT le premier marqueur d'article détecté (visas,
    considérants, titre du décret/loi) — souvent riche en références
    législatives utiles pour l'extraction d'entités (étape 3).
    """
    matches = _filter_article_matches(text, lang)
    if matches:
        start = _skip_sommaire(text, lang, limit=matches[0].start())
        first_art = matches[0].start()
        if first_art > start:
            return text[start:first_art].strip()
        return text[:first_art].strip()
    return text.strip()


def get_sommaire(text: str, lang: str = "fr") -> str:
    """
    Retourne le sommaire (table des matières) du Bulletin officiel,
    s'il existe : de l'en-tête ``SOMMAIRE`` (ou ``فهرست`` en arabe) jusqu'au
    début du contenu réel (première section TEXTES GÉNÉRAUX / PARTICULIERS,
    ou ``نصوص عامة`` / ``نصوص خاصة``, hors liste), même logique de
    délimitation que _skip_sommaire.  Chaîne vide si le document n'a pas
    de sommaire.
    """
    sommaire_marker = SOMMAIRE_MARKERS.get(lang, SOMMAIRE_MARKERS["fr"])
    sommaire_pos = text.find(sommaire_marker)
    if sommaire_pos == -1:
        return ""
    matches = _filter_article_matches(text, lang)
    limit = matches[0].start() if matches else None
    end = _skip_sommaire(text, lang, limit=limit)
    if end <= sommaire_pos:
        return ""
    # _skip_sommaire renvoie la position APRÈS le premier marqueur de
    # section (ex. « TEXTES GENERAUX ») : ce marqueur marque le début du
    # contenu réel et ne fait pas partie du sommaire lui-même.
    for marker in SECTION_MARKERS.get(lang, SECTION_MARKERS["fr"]):
        if text[end - len(marker):end] == marker:
            end -= len(marker)
            break
    else:
        # Marqueur tronqué par l'extraction PDF (« TEXTES Gl ») : on le
        # retire aussi du sommaire s'il se termine exactement à `end`.
        m = re.search(r"(?:TEXTES[ \t]+[GP][^\n]*|\u0646\u0635\u0648\u0635\s+(?:\u0639\u0627\u0645\u0629|\u062e\u0627\u0635\u0629))$",
                      text[max(0, end - 40):end])
        if m:
            end -= m.end()
    return text[sommaire_pos:end].strip()


def get_per_decree_preamble_map(text: str, lang: str = "fr") -> list[dict]:
    """
    Extrait les préambules par décret à partir des articles segmentés.

    Au lieu de balayer le texte avec des titres de document (qui ramassent
    aussi les entrées du sommaire), on utilise les frontières entre articles
    : pour chaque article, on cherche s'il est précédé d'un titre de
    document (Arrêté/Décret/Dahir/Décision) dans le texte entre la fin de
    l'article précédent et le début de l'article courant.  Si oui, le
    préambule de ce décret est le texte entre ce titre et l'article.

    Les ``first_article_idx`` référencent le tableau FINAL des articles
    (après fusion des fragments courts, découpage des annexes et des
    articles trop longs) — pas les indices bruts des marqueurs ART., qui
    dérivent dès qu'une annexe est insérée ou qu'un article est scindé.
    Les instruments de queue sans article (décisions courtes, avis,
    annexes) reçoivent ``first_article_idx = len(articles)`` (un cran
    au-delà du tableau : instrument virtuel sans article).

    Retourne une liste de dicts :
        {"title": str, "preamble": str, "first_article_idx": int}
    """
    matches = _filter_article_matches(text, lang)
    if not matches:
        return []

    final_articles = segment_into_articles(text, lang)
    start_to_idx: dict[int, int] = {}
    for idx, art in enumerate(final_articles):
        start_to_idx.setdefault(art.start_pos, idx)
    final_len = len(final_articles)

    title_pattern = DOCUMENT_TITLE_PATTERN_FR if lang == "fr" else DOCUMENT_TITLE_PATTERN_AR

    def _title_matches(region: str) -> list:
        return [
            m for m in title_pattern.finditer(region)
            if lang == "ar" or _is_doc_title_match(region, m, lang)
        ]

    sommaire_end = _skip_sommaire(text, lang, limit=matches[0].start())
    decrees = []

    for i, m in enumerate(matches):
        art_start = m.start()
        if i == 0:
            search_start = sommaire_end
        else:
            # Previous article text region ends at the position just before
            # the next doc title OR the next ART marker, whichever comes first
            prev_end = matches[i - 1].end()
            # Check if there was a doc boundary between prev article and this one
            search_start = prev_end

        gap = text[search_start:art_start]

        # Les titres "Annexe" en tête de gap appartiennent à l'instrument
        # PRÉCÉDENT (annexes jointes à ses articles) — le titre de
        # l'instrument suivant arrive APRÈS.  On ne retient une "Annexe"
        # que si le gap ne contient aucun autre titre.
        raw_titles = _title_matches(gap)
        non_annexe = [t for t in raw_titles
                      if not t.group().strip().lower().startswith("annexe")]
        doc_matches_in_gap = non_annexe or raw_titles
        if doc_matches_in_gap:
            first_doc = doc_matches_in_gap[0]
            title = first_doc.group().strip()
            # Preamble = from the doc title to the next doc title
            # (or ART marker if no next doc title in the gap)
            preamble_start = search_start + first_doc.start()
            if len(doc_matches_in_gap) > 1:
                second_doc = doc_matches_in_gap[1]
                preamble_end = search_start + second_doc.start()
            else:
                preamble_end = art_start
            preamble = text[preamble_start:preamble_end].strip()
            if preamble and len(preamble) > 100:
                decrees.append({
                    "title": title,
                    "preamble": preamble,
                    "first_article_idx": start_to_idx.get(m.start(), i),
                })

    # Post-process: scan text after the last article for document titles
    # that have no article markers (e.g. short CSCA decisions, AVIS, Annexes).
    # Add them as article-less decrees: first_article_idx = len(articles)
    # (one past the end of the array — "virtual instrument").
    if matches:
        last_art_end = matches[-1].end()
        remaining = text[last_art_end:]
        # Titres après le dernier article (décisions courtes, avis, annexes) :
        # on exige une majuscule initiale en français.  Les citations
        # bibliographiques en minuscule (« dahir n° 1-16-115 … portant
        # promulgation de la loi n° 01-16 », note de bas de page de l'Avis du
        # CESE, BO_6758) ne doivent PAS créer d'instrument fantôme
        # (n_articles = 0) en fin de document.
        remaining_titles = [
            dm for dm in _title_matches(remaining)
            if lang == "ar" or dm.group().strip()[:1].isupper()
        ]
        for ti, dm in enumerate(remaining_titles):
            dm_start = last_art_end + dm.start()
            if ti + 1 < len(remaining_titles):
                dm_end = last_art_end + remaining_titles[ti + 1].start()
            else:
                dm_end = len(text)
            preamble = text[dm_start:dm_end].strip()
            if preamble and len(preamble) > 50:
                decrees.append({
                    "title": dm.group().strip(),
                    "preamble": preamble,
                    "first_article_idx": final_len,
                })

    return decrees


def segment_into_articles(text: str, lang: str = "fr") -> list:
    """Point d'entrée unique : route vers la bonne fonction selon la langue."""
    if lang == "fr":
        return segment_into_articles_fr(text)
    elif lang == "ar":
        return segment_into_articles_ar(text)
    else:
        raise ValueError(f"Langue non supportée : {lang}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python segmenter.py <chemin_vers_fichier_nettoye.txt> [fr|ar]")
        sys.exit(1)

    lang_arg = sys.argv[2] if len(sys.argv) > 2 else "fr"
    with open(sys.argv[1], encoding="utf-8") as f:
        text_content = f.read()

    preamble = get_preamble(text_content, lang=lang_arg)
    articles = segment_into_articles(text_content, lang=lang_arg)

    print(f"Préambule ({len(preamble)} caractères) :")
    print(preamble[:300])
    print(f"\n{len(articles)} article(s) détecté(s) :\n")

    for art in articles:
        print(f"--- Article {art.number} (marqueur : '{art.raw_header}') ---")
        print(art.text[:200])
        print()