"""
src/extraction/gazetteer_filter.py
Additional post-NER filter for PERSON false positives.

Adds a second layer of filtering after ``filter_persons()`` in
etape4_pipeline.py.  Catches patterns that survive the first pass:
cross-reference markers ("Op.cit", "Ibid"), OCR noise ("Cetet situatoin"),
preposition sequences, and other known false positives.
"""

import re

# ── Gazetteer of known non-person text fragments ───────────────────────
# These are text spans that the statistical NER flags as PERSON but are
# never actual names in a legal BO context.
_GAZETTEER_BLOCKLIST = {
    # Cross-reference and citation markers
    "op. cit", "op.cit", "ibid", "ibid.", "cf.", "cf", "supra", "infra",
    "idem", "loc. cit", "loc.cit",
    # Verb/noun fragments mistaken as names
    "mettez les cales", "mettez les",
    # OCR noise / meaningless fragments
    "cetet situatoin", "cetet", "situatoin",
    "aller principal", "principal",
    # Administrative headers mistaken for names
    "recto", "verso", "filigrane", "ballon", "format",
    "tifinagh", "diamètre", "frappe", "cesar",
    "chesterfield", "glamour", "fortuna", "marquise", "lights",
    # Generic legal fragments
    "vu", "vule", "parrêté", "pamélioration", "péconomie",
    "paccomplissement", "leconseilconsidère", "ilimporte", "finforma",
    "monte", "téservés", "mo", "rem",
    "n", "ro", "de",
    # Street / place names
    "sidi rbat",
    # University fragments mistaken for persons
    "derjavin", "derzhavine", "pavlov de riazan",
    # Moroccan provinces/places mistaken for persons
    "al haouz",
}

# ── Toponymes arabes (Maroc) ─────────────────────────────────────────────
# Le NER camel-tools étiquette régulièrement des noms de lieux marocains
# comme PERSON ("الفقيه", "بن صالح الغرب", "ملال") ou comme ORG.
# 1) Préfixes toponymiques : une entité PERSON dont le texte commence par
#    l'un de ces préfixes est un lieu-dit / tribu / région, pas un nom.
_AR_TOPO_PREFIXES = (
    "بني ", "أيت ", "ايت ", "أولاد ", "اولاد ", "الفقيه ",
    "القصر ", "القصبة ", "دوار ", "المدينة ", "جهة ", "عمالة ",
    "إقليم ", "اقليم ", "تيزي ", "سيدي إفني", "أكدز",
)
# 2) Noms de villes/provinces connus — liste non exhaustive des villes les
#    plus fréquentes dans les BO (le contexte d'un BO juridique : un nom de
#    ville étiqueté PERSON est toujours un faux positif).
_AR_TOPO_NAMES = {
    "الدار البيضاء", "الرباط", "سلا", "مراكش", "فاس", "مكناس", "طنجة",
    "تطوان", "أكادير", "اغادير", "أغادير", "وجدة", "العرائش", "القنيطرة",
    "الجديدة", "آسفي", "أسفي", "خريبكة", "خنيفرة", "الرشيدية", "ورزازات",
    "الناظور", "الحسيمة", "بركان", "تازة", "إفران", "أفران", "الصويرة",
    "تارودانت", "تزنيت", "طانطان", "كلميم", "العيون", "الداخلة", "سطات",
    "برشيد", "خميسات", "سيدي قاسم", "سيدي سليمان", "المحمدية", "تمارة",
    "الصخيرات", "بوسكورة", "بني ملال", "الفقيه بن صالح", "اليوسفية",
    "سيدي بنور", "أزمور", "السوالم", "الغرب", "الشاوية", "دكالة",
    "الشرق", "سوس ماسة", "جهة فاس مكناس", "بوجدور", "إقليم الحوز",
    "أزيلال", "جرادة", "فكيك", "تاونات", "قلعة السراغنة", "الرحامنة",
    "ميدلت", "تنغير", "السمارة", "طرفاية",
}

# Contextes qui signalent qu'un nom de lieu suit (position, adresse,
# signature de préambule) — "ب" de lieu ("بوجدة", "بالدار البيضاء").
_AR_PLACE_CONTEXT_RE = re.compile(
    r"(?:^(?:ب|في|بال|من)\s*|محل\s*|إقامة\s*|جهة\s*|مدينة\s*)",
    re.UNICODE,
)


def _looks_like_ar_toponym(text: str, full_text: str = "", start: int = 0) -> bool:
    """True si *text* (span PERSON) est un nom de lieu marocain."""
    norm = text.strip()
    if not norm:
        return False
    if norm in _AR_TOPO_NAMES:
        return True
    upper = norm.upper()
    for p in _AR_TOPO_PREFIXES:
        if norm.startswith(p) or upper.startswith(p.upper()):
            return True
    # "ملال" seul (fragment de "بني ملال") : trop court et ne ressemble
    # pas à un nom de personne — le NER le détecte comme ORG/PERS isolé.
    if len(norm.split()) == 1 and norm in {"ملال", "الفقيه", "الغرب"}:
        return True
    # Contexte précédent : "الفقيه بن صالح" est une ville — le NER scinde
    # souvent le span en deux ("الفقيه" + "بن صالح الغرب"). On rejette
    # aussi le complément si le texte immédiatement précédent contient
    # "الفقيه" (titre du toponyme).
    if full_text and start > 0:
        preceding = full_text[max(0, start - 30):start]
        if "الفقيه" in preceding and (
            norm.startswith("بن ") or norm.startswith("أولاد ")
        ):
            return True
    return False


# Cross-reference marker PREFIXES — when the NER merges a marker with a
# following word ("Op.cit P"), we reject the whole span if its lowercase
# form starts with one of these tokens.  This is more robust than listing
# every possible combination.
_CROSS_REF_PREFIXES = ("op.cit", "op. cit", "ibid", "cf.", "loc.cit", "loc. cit", "supra", "infra", "idem")

# Prepositions and articles that when alone or in pairs are never names
_PREPOSITIONS = {"de", "du", "des", "d'", "le", "la", "les",
                 "sur", "sous", "dans", "pour", "par", "avec",
                 "sans", "chez", "entre", "selon", "outre"}

_OCR_MARKER = re.compile(r"[0-9]|[\u200E\u200F\u202A-\u202E]")

# A heuristic for missing-word-boundary OCR: a lowercase letter followed
# by an uppercase letter without a space (e.g. "deL'arrêté" → bad).
_CAMEL_OCR = re.compile(r"[a-zéèêëàâîïôûùç][A-ZÉÈÊËÀÂÎÏÔÛÙÇ]")

# ── Contexte taxonomique ────────────────────────────────────────────────
# "Gelidium Sesquipedale" a exactement la même forme qu'un vrai nom de
# personne (Capitalisé Capitalisé) : aucun regex sur le texte de l'entité
# seule ne peut les distinguer. Ce qui les distingue, c'est le CONTEXTE :
# ces noms scientifiques n'apparaissent dans ces BO que dans des listes
# d'espèces (algues, poissons, coquillages...), jamais à côté d'un verbe
# d'action ou d'un titre comme le serait un vrai nom de personne.
# Confirmé sur BO_7510_Fr : "les algues rouges « Gracilaria Gracilis »,
# « Gelidium Sesquipedale » et « Grateloupia filicina »" — le mot-clé
# taxonomique apparaît systématiquement dans les ~80 caractères précédents
# (une ligne ou deux dans le texte source, à cause du retour à la ligne).
_SPECIES_CONTEXT_WORDS = (
    "algue", "algues", "espèce", "espèces", "poisson", "poissons",
    "crustacé", "crustacés", "coquillage", "coquillages", "mollusque",
    "mollusques", "variété", "variétés",
)
_SPECIES_CONTEXT_WINDOW = 80

# Mots-clés indiquant un nom de rue/voie (ex. "avenue Mohammed VI")
_STREET_CONTEXT_WORDS = (
    "avenue", "rue", "boulevard", "place", "impasse", "route",
    "allée", "cité", "square", "passage", "chemin",
)
_STREET_CONTEXT_WINDOW = 50

# Mots-clés indiquant un contexte universitaire (ex. "Université d'Etat
# de Tambov nommée d'après G.R. Derjavin")
_UNIVERSITY_CONTEXT_WORDS = (
    "université", "faculté", "académie", "institut", "école", "campus",
    "département", "laboratoire", "centre de recherche",
    # Variantes sans accent / OCR
    "universite", "faculte", "academie", "institut", "ecole",
)
_UNIVERSITY_CONTEXT_WINDOW = 100


def _has_species_context(text: str, start: int, full_text: str) -> bool:
    if not full_text:
        return False
    window_start = max(0, start - _SPECIES_CONTEXT_WINDOW)
    preceding = full_text[window_start:start].lower()
    return any(w in preceding for w in _SPECIES_CONTEXT_WORDS)


def _has_street_context(text: str, start: int, full_text: str) -> bool:
    """Reject if the entity is preceded by a street-name keyword."""
    if not full_text:
        return False
    window_start = max(0, start - _STREET_CONTEXT_WINDOW)
    preceding = full_text[window_start:start].lower()
    return any(w in preceding for w in _STREET_CONTEXT_WORDS)


def _has_university_context(text: str, start: int, end: int, full_text: str) -> bool:
    """Reject if the entity is near a university/institution keyword."""
    if not full_text:
        return False
    # Check before
    before_start = max(0, start - _UNIVERSITY_CONTEXT_WINDOW)
    before = full_text[before_start:start].lower()
    if any(w in before for w in _UNIVERSITY_CONTEXT_WORDS):
        return True
    # Check after
    after = full_text[end:end + _UNIVERSITY_CONTEXT_WINDOW].lower()
    if any(w in after for w in _UNIVERSITY_CONTEXT_WORDS):
        return True
    return False


def gazetteer_filter_persons(persons: list[dict], full_text: str = "") -> list[dict]:
    """
    Second-pass filter for PERSON entities.

    Rejects entities whose text:
      1. Matches a known non-person token (cross-ref markers, OCR noise)
      2. Is a sequence of only prepositions/articles (e.g. "de la", "du")
      3. Contains digit or Unicode-direction control characters
      4. Contains camelCase OCR (merged word boundary)
      5. Is a single lowercase word not in MINISTERS_WHITELIST
      6. Is preceded by a taxonomic context word (species list), e.g.
         "Gelidium Sesquipedale" appearing right after "les algues rouges"

    Args:
        persons: List of PERSON entity dicts with at least ``text`` and
                 ``label`` keys (and ideally ``start`` for the context check).
        full_text: The article's full text, used to look at the context
                   preceding each entity (needed for check 6 only — pass
                   "" or omit to skip it, e.g. for callers without offsets).

    Returns:
        Filtered list of persons.
    """
    filtered = []
    for p in persons:
        text = p.get("text", "").strip()
        if not text:
            continue

        norm = text.lower().strip(".,;:!?\"'«»()[]{}")

        # 1. Gazetteer blocklist
        if norm in _GAZETTEER_BLOCKLIST:
            continue

        # 1b. Cross-reference prefix match — catches spans like "Op.cit P"
        #     where a known marker starts the entity but has trailing text.
        if norm.startswith(_CROSS_REF_PREFIXES):
            continue

        # 2. Preposition/article sequence
        words = norm.split()
        if len(words) <= 3 and all(w in _PREPOSITIONS for w in words):
            continue

        # 3. Digit or control characters
        if _OCR_MARKER.search(text):
            continue

        # 4. CamelCase OCR (merged words)
        if _CAMEL_OCR.search(text):
            continue

        # 5. Single lowercase word (unlikely to be a proper name without
        #    context, and context here is a legal BO document)
        if len(words) == 1 and text[0].islower():
            continue

        # 6. Entirely uppercase single word > 4 letters (likely an acronym
        #    or header fragment, not a person)
        if len(words) == 1 and text.isupper() and len(text) > 4:
            continue

        # 7. Taxonomie / contexte d'espèce
        if _has_species_context(text, p.get("start", 0), full_text):
            continue

        # 8. Contexte de rue / lieu-dit (e.g. "avenue Mohammed VI")
        if _has_street_context(text, p.get("start", 0), full_text):
            continue

        # 9. Contexte universitaire (e.g. "Université ... G.R. Derjavin")
        if _has_university_context(
            text, p.get("start", 0), p.get("end", 0), full_text,
        ):
            continue

        # 10. Noms de douars/villages marocains fréquemment étiquetés PERSON
        #     par le NER statistique — "Sidi Ghiate", "Aït Ourir",
        #     "Sidi Abdellah Ghiat", "Haj Kaddour" sont des toponymes,
        #     pas des personnes.
        _DOUAR_PREFIXES = ("SIDI ", "AÏT ", "AIT ", "HAJ ", "HADJ ",
                           "DOUAR ", "DOUAR ", "KSAR ", "KASBAH ",
                           "MELLAH ", "MEDINA ")
        upper_text = text.upper()
        if any(upper_text.startswith(p) for p in _DOUAR_PREFIXES):
            continue

        # 11. Toponymes arabes marocains étiquetés PERSON ("الفقيه بن صالح",
        #     "بني ملال", "سيدي قاسم", "الغرب") — jamais un nom de personne
        #     dans un BO juridique.
        if _looks_like_ar_toponym(text, full_text, p.get("start", 0)):
            continue

        filtered.append(p)

    return filtered