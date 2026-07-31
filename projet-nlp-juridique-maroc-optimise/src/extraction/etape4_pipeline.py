"""
src/extraction/etape4_pipeline.py
Orchestration de l'étape 4 : 4a -> 4a-bis -> 4b -> 4c -> JSON enrichi.

Version améliorée avec :
- Filtrage des faux positifs (personnes, organisations)
- Extraction des dates (hégiriennes + grégoriennes)
- Logging détaillé
- Copie profonde pour éviter les effets de bord
"""

import copy
import logging
import re
from typing import List, Dict, Any

from . import article_citation_patterns as citations_fr
from . import article_citation_patterns_ar as citations_ar
from . import ner_statistical as ner_fr
from . import ner_statistical_ar as ner_ar
from .citation_resolver import resolve_citations
from .ner_merge import merge_with_rule_based_entities
from .entity_span_utils import normalize_entity
from .gazetteer_filter import gazetteer_filter_persons
from .dates_patterns import MOIS_GREGORIEN_FR, MOIS_HIJRI_FR
from .dates_patterns_ar import MOIS_GREGORIEN_AR, MOIS_HIJRI_AR

# Configuration du logging
logger = logging.getLogger(__name__)

# Constantes
LEGAL_TEXT_LABELS = ("LOI", "DECRET", "ARRETE", "DAHIR", "BULLETIN_OFFICIEL")
DATE_LABELS = ("DATE_HIJRI", "DATE_GREGORIAN")

# Modules par langue
_CITATION_MODULES = {"fr": citations_fr, "ar": citations_ar}
_NER_MODULES = {"fr": ner_fr, "ar": ner_ar}
_NER_UNAVAILABLE_WARNED = set()  # langues déjà signalées comme sans NER statistique (voir enrich_article_json)

# ============================================================================
# FILTRES POUR LES PERSONNES ET ORGANISATIONS
# ============================================================================

# Mots-clés communs en arabe qui ne sont pas des noms propres
COMMON_TITLES_AR = {
    # Mis à jour pour correspondre à l'orthographe désormais correcte
    # (voir cleaner_ar.fix_lam_meem_transposition) — ces mots apparaissaient
    # auparavant comme "امل..." à cause d'un bug d'extraction depuis
    # corrigé ; le texte en amont produit maintenant "الم..." correctement.
    "المعهد", "المدير", "المترشح", "المكلف", "المنتدب", "المعني",
    "المساعد", "المتمرن", "المديران", "المترشحين", "المنتدبين",
    "المشرع", "المنفذ", "المراقب", "المختص", "المؤهل",
    "المكلف\nالمتعلق",
}

# Mots-clés communs en français qui ne sont pas des noms propres
COMMON_TITLES_FR = {
    "le", "la", "les", "directeur", "directrice", "candidat", "candidate",
    "responsable", "chargé", "chargée", "inspecteur", "inspectrice",
    "président", "présidente", "ministre", "secrétaire", "général",
    "chapitre", "article", "annexe", "ballon", "format", "filigrane",
    "mixture", "recto", "tifinagh", "diamètre", "frappe", "cesar",
    "chesterfield", "glamour", "fortuna", "marquise", "lights",
    "approche", "thématique", "satellite", "orientation",
    "douar", "lambert", "roi", "fait", "verso", "centre", "fond", "liste",
    "thème", "crassostrea", "ruditapes", "palais", "portail",
    "arabesque", "ornement", "modernité", "souveraineté", "ouverture",
    "développement", "violet", "faciale", "polymère", "substrat",
    "fenêtre", "transparente", "perspective", "réseau", "fibre",
    "numérique", "transformation", "digitale", "stylisation",
    "rayon", "solaire", "partie", "supérieure", "inférieure",
    "dénomination", "institut", "émission", "représentation", "projet",
    "génie logiciel", "sciences de l'eau",
    # Mots de liaison / prépositions souvent mal classifiés comme PERSON
    "dans", "sur", "pour", "avec", "sans", "selon", "chez",
    "cette", "ces", "tous", "toutes", "chaque", "entre",
    "après", "avant", "durant", "pendant", "non", "outre",
    "notamment", "toutefois", "cependant", "néanmoins", "par", "sous",
    # Faux positifs supplémentaires observés sur BO 6822
    "vule", "technique", "montant", "largeur", "crassostrea", "crassosfrea",
    "lhuître", "environnemental", "homme", "parcelle", "bornes", "borne",
    "latitude", "longitude", "zone", "ancien", "argoub", "rem",
    "calastropaiques", "déchets", "deraton",
    # Noms scientifiques / taxonomiques (coquillages, poissons, etc.)
    "pecten", "maximus", "perna", "gracilaria", "gracili",
    "mytilus", "galloprovincialis", "pecten",
    # Noms de lieux / adresses souvent classifiés comme PERSON
    "saint", "jacques", "azib", "labrareq", "moulay", "rachid", "bloc",
    "ancien", "argoub", "hay",
    # Mots techniques / en-têtes de tableau
    "borncs", "pêche", "pêche maritime", "pamélioration",
    "leconseilconsidère", "ilimporte",  "finforma", "paccomplissement",
    "monte", "crassostrea", "développement",
    "vu", "arrêté", "parrêté", "péconomie", "téservés", "mo",
    # Prépositions / noms communs supplémentaires
    "de", "royal", "rabat",
    # Termes administratifs/commerciaux (importés depuis BLACKLIST_PERSON)
    "matériel", "charrue", "tracteur", "broyeur", "semoir",
    "moissonneuse", "batteuse", "faucheuse", "récolteuse",
    "type", "plafond", "taux", "nombre", "unité",
    "engagement", "paiement", "décision", "état", "ordre",
    "facture", "bon", "quittance", "reçu", "procès-verbal",
    "avenant", "marché", "contrat", "convention", "bordereau",
    "décompte", "attestation", "certificat", "copie", "extrait",
    "jugement", "arrêté", "dahir", "décret", "loi", "bulletin",
    "n", "ro",
}

# Mots-clés pour les organisations en arabe (vrais institutions)
# Mots-clés pour les organisations en arabe (vrais institutions) — alignés
# sur INSTITUTION_KEYWORDS_AR (institution_patterns_ar.py) pour que ce
# filtre ne rejette pas des organisations statistiques (camel-tools) qui
# utilisent les mêmes mots-clés mais n'ont pas été posées par la regex.
from .institution_patterns_ar import INSTITUTION_KEYWORDS_AR
ORG_KEYWORDS_AR = set(INSTITUTION_KEYWORDS_AR) | {
    "مكتب", "وكالة", "هيئة", "مديرية", "مصلحة", "لجنة", "مجلس", "مؤسسة",
}

# Mots-clés pour les organisations en français
ORG_KEYWORDS_FR = {"office", "agence", "autorité", "direction", "ministère", "service", "commission",
                   "conseil", "banque", "caisse", "fond", "fonds", "institut", "association",
                   "société", "entreprise", "établissement", "organisation", "comité", "délégation",
                   "chambre", "fédération", "union", "régie"}

# Patterns pré-compilés pour l'évaluation des mots-clés d'organisations
# (évite de recompiler re.escape(kw) dans une boucle pour chaque entité)
_ORG_PATTERNS_AR = [re.compile(rf"\b{re.escape(kw)}\b") for kw in ORG_KEYWORDS_AR]
_ORG_PATTERNS_FR = [re.compile(rf"\b{re.escape(kw)}\b") for kw in ORG_KEYWORDS_FR]

# Mots-clés à exclure des organisations (faux positifs)
COMMON_ORG_WORDS_AR = {"دوار", "سعيد", "بن", "ايت", "مطير", "بنحسين", "البيضاء", "الشق", "احسين", "حسين", "نعمان"}
COMMON_ORG_WORDS_FR = {"village", "commune", "mairie", "préfecture", "région", "département"}

# Whitelist des ministres marocains (pour améliorer la précision)
MINISTERS_WHITELIST = {
    "ar": {
        "عزيز اخنوش", "فوزي لقجع", "نزار بركة", "عبد الصمد قيوح", "نادية فتاح",
        "احمد البواري", "عز الدين املداوي", "عبد اللطيف وهبي", "رياض مزور",
        "محمد املهدي بنسعيد", "فتيحة اللحيان", "كريمة الحمياني"
    },
    "fr": {
        "Aziz Akhannouch", "Fouzi Lekjaa", "Nizar Baraka", "Abdessamad Qioh", "Nadia Fettah",
        "Ahmed El Bouari", "Azzedine El Madoui", "Abdellatif Ouahbi", "Ryad Mezzour",
        "Mohamed Mehdi Bensaid", "Fatiha El Lahyan", "Karima El Hamiani"
    }
}

# (supprimé — fusionné dans COMMON_TITLES_FR ci-dessus)

def filter_persons(persons: List[Dict], lang: str) -> List[Dict]:
    """
    Filtre les personnes pour éliminer les faux positifs.
    """
    filtered = []
    for p in persons:
        text = p.get("text", "").strip()
        if not text:
            continue

        # 1. Supprimer les entités trop courtes (1 mot) qui sont des titres communs
        words = text.split()
        if lang == "fr":
            if len(words) == 1:
                if text.lower() in COMMON_TITLES_FR:
                    continue
                # Supprimer les mots français isolés qui ne sont pas des noms propres
                if len(text) <= 2 or re.match(r"^[A-Z]{3,}$", text):
                    continue
            # Pour les entités multi-mots, filtrer si un mot est dans les titres communs
            # (évite les phrases OCR comme "Latitude Longitude Bornes")
            common_hits = sum(1 for w in words if w.lower() in COMMON_TITLES_FR)
            if common_hits >= len(words) - 1:  # tous ou presque sont des mots communs
                continue

        # 2. Supprimer le bruit OCR (textes avec des chiffres, symboles Unicode,
        #    sauts de ligne, pipes, etc.)
        if re.search(r'[0-9]|[‏‎\n\r|]', text):
            continue

        # 3. Supprimer les entités trop longues (plus de 5 mots) - souvent des phrases entières
        if len(words) > 5:
            continue

        # 4. Garder les noms dans la whitelist des ministres
        if lang in MINISTERS_WHITELIST:
            if any(name in text for name in MINISTERS_WHITELIST[lang]):
                filtered.append(p)
                continue

        # 5. Règle générale : garder les entités avec au moins 2 mots et qui contiennent une majuscule (fr) ou un nom propre (ar)
        if lang == "fr" and not any(c.isupper() for c in text):
            continue
        if lang == "ar" and len(words) < 2:
            continue

        filtered.append(p)

    return filtered

_VALID_MONTHS_BY_LANG = {
    "fr": set(MOIS_GREGORIEN_FR) | set(MOIS_HIJRI_FR),
    "ar": set(MOIS_GREGORIEN_AR) | set(MOIS_HIJRI_AR),
}


def filter_dates(dates: List[Dict], lang: str = "fr") -> List[Dict]:
    """
    Garde uniquement les dates dont le nom de mois est reconnu (grégorien ou
    hégirien) pour la langue de l'article — évite de garder de faux
    positifs tout en ne filtrant pas à tort les dates valides d'une langue
    au prétexte qu'elles ne correspondent pas aux mois d'une autre langue.
    """
    valid_months = _VALID_MONTHS_BY_LANG.get(lang, set())
    filtered = []
    for d in dates:
        text = d.get("text", "").lower()
        if any(m in text for m in valid_months):
            filtered.append(d)
    return filtered

_OCR_NOISE = re.compile(r"[‏‎]")

# ── Détection des noms de société ────────────────────────────────────────
# "Zniber Seafarm" et "Uni Fiber" ont la même forme qu'un vrai nom de
# personne (Capitalisé Capitalisé) : le texte de l'entité seul ne permet
# pas de les distinguer. Ce qui les trahit, c'est le CONTEXTE : dans ces
# BO, un nom de société est systématiquement introduit par "société" (avec
# ou sans guillemets — confirmé sur BO_7510_Fr : "la société Uni Fiber",
# "la société « ZNIBER SEAFARM Sarl AU »") ou suivi d'une forme juridique
# (Sarl, SA, Sarl AU...). Un nom de personne n'apparaît jamais dans ce
# contexte précis. Un gazetteer de noms d'entreprises connues ne
# fonctionnerait pas : chaque nouveau numéro du BO introduit de nouvelles
# sociétés (TAIBA SEAFOOD, EXTRAMER, IMAR AQUA, AQUADUNE... déjà 5
# nouvelles rien que sur ce document).
_COMPANY_CONTEXT_WINDOW = 80
_COMPANY_SUFFIX_WINDOW = 40
_COMPANY_SUFFIX_RE = re.compile(r"^\s*[»\"')\]>]*\s*(Sarl|SARL|S\.A\.?R\.?L\.?|SA|S\.A\.|S\.A\.R\.L|Sarl AU|SARL AU|S\.A\.R\.L\.? AU)\b", re.IGNORECASE)

# Mots caractéristiques des noms d'entreprise (secteur, forme juridique,
# suffixe courant) qui n'apparaissent quasiment jamais dans un nom de
# personne. Permet de reclasser les entités PERSON qui ne sont pas
# précédées de "société" (cas "Uni Fiber").
_COMPANY_INDICATORS = {
    "seafarm", "fiber", "telecom", "seafood", "aquaculture", "holding",
    "technologies", "systems", "solutions", "logistics", "industries",
    "corporation", "incorporated", "limited", "company", "group",
    "maroc", "farma", "agricole",
}

# Pattern for ALL-CAPS company names that spaCy NER ignores entirely
# (e.g. "TAIBA SEAFOOD", "EXTRAMER", "IMAR AQUA", "AQUADUNE")
# Matches 2-6 uppercase tokens separated by spaces, preceded or followed
# by company context or wrapped in angle quotes.
_ALL_CAPS_COMPANY_RE = re.compile(
    r'(?:[Ss]oci[eé]t[eé]|[Ss]ociete)\s*[«"\'<]?\s*'
    r'([A-Z][A-Z\s]{2,60}?)'
    r'(?=\s*[»"\'>]'
    r'|\s+(?:Sarl|SARL|SA|S\.A\.)\b'
    r'|\s[a-z]'        # space + lowercase = next word starts
    r'|[,;:.\n]|$)',
)

# Standalone all-caps names in angle quotes — the name may be followed
# by "Sarl", "SA", "Sarl AU" etc. before the closing quote.  We capture
# only the ALL-CAPS portion by requiring a boundary (lowercase letter or
# close-quote) right after the captured text.
_STANDALONE_COMPANY_RE = re.compile(
    r'[«"\'<]\s*([A-Z][A-Z\s]{2,60}?)'
    r'(?=\s*[»"\'>]|\s+(?:Sarl|SARL|SA|S\.A\.)\b)',
)

def _has_company_context(text: str, start: int, end: int, full_text: str) -> bool:
    """Vérifie si l'entité est précédée de "société" ou suivie d'une forme
    juridique (Sarl, SA...) — signe qu'il s'agit d'un nom d'entreprise et
    non d'une personne, quelle que soit la forme du texte de l'entité."""
    if not full_text:
        return False

    # Reject entity text that looks like a place/address rather than a
    # company name — these are typically comma-separated fragments after
    # the company name in registration articles ("société …, adresse …").
    _ADDRESS_INDICATORS = {
        "lotissement", "douar", "oulad", "ouled", "résidence",
        "residence", "immeuble", "appartement", "annexe", "domaine",
        "coopérative", "cooperative", "dhar", "bled", "commune",
    }
    words_lower = set(w.lower().strip(".,;:»\"'") for w in text.split())
    if words_lower & _ADDRESS_INDICATORS:
        return False

    preceding = full_text[max(0, start - _COMPANY_CONTEXT_WINDOW):start].lower()
    if "société" in preceding or "societe" in preceding:
        return True
    following = full_text[end:end + _COMPANY_SUFFIX_WINDOW]
    if _COMPANY_SUFFIX_RE.search(following):
        return True
    # Fallback: bare all-caps name (e.g. "TAIBA SEAFOOD") — if the
    # entity itself is all-caps multi-word, it's practically never a
    # person name in a legal BO context.
    # Exception: Moroccan minister/person names in all caps at the end
    # of articles (signature blocks) — e.g. "AHMED EL BOUARI",
    # "ABDESSAMAD KAYOUH", "AZZEDDINE EL MIDAOUI".
    _MOROCCAN_PERSON_MARKERS = ("EL ", "BEN ", "ABD", "AL ",
                                "MOHAMED", "MOHAMMED", "HASSAN",
                                "ABDEL", "ABDOU", "MOULAY")
    if re.match(r"^[A-Z][A-Z\s]{2,}$", text) and len(text.split()) >= 2:
        upper_words = text.upper().split()
        is_moroccan_person = any(
            any(w.startswith(m.rstrip()) for w in upper_words)
            for m in _MOROCCAN_PERSON_MARKERS
        )
        if is_moroccan_person:
            return False
        return True
    # Company-indicative words: entity text contains a word that is
    # characteristic of company names but never of person names,
    # e.g. "Fiber" in "Uni Fiber", "Seafarm" in "Zniber Seafarm".
    words = set(w.lower() for w in text.split())
    if words & _COMPANY_INDICATORS:
        return True
    return False


def _extract_company_names_regex(full_text: str, lang: str = "fr") -> list[dict]:
    """
    Extrait les noms de société que le NER statistique ne détecte pas
    (typiquement les noms en capitales comme TAIBA SEAFOOD, EXTRAMER,
    IMAR AQUA, AQUADUNE), en utilisant des patterns regex sur le texte
    brut indépendants du passage NER.

    Returns:
        Liste d'entités ORG avec les clés {text, start, end, label}.
    """
    companies = []
    for m in _ALL_CAPS_COMPANY_RE.finditer(full_text):
        name = m.group(1).strip()
        words = name.split()
        if len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 6):
            companies.append({
                "text": name, "start": m.start(1), "end": m.end(1),
                "label": "ORG",
            })
    for m in _STANDALONE_COMPANY_RE.finditer(full_text):
        name = m.group(1).strip()
        words = name.split()
        if len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 6):
            # Deduplicate with already-found companies
            dup = False
            for c in companies:
                if c["start"] == m.start(1) or c["text"] == name:
                    dup = True
                    break
            if not dup:
                companies.append({
                    "text": name, "start": m.start(1), "end": m.end(1),
                    "label": "ORG",
                })
    return companies


def reclassify_company_names(persons: List[Dict], full_text: str) -> tuple[List[Dict], List[Dict]]:
    """
    Sépare les entités PERSON qui sont en réalité des noms de société
    (contexte "société ..." ou suffixe Sarl/SA) du reste.

    Sans cette étape, un nom comme "Zniber Seafarm" ou "Uni Fiber" reste
    étiqueté PERSON (faux positif) OU serait perdu s'il était simplement
    retiré — il doit être RECLASSÉ en organisation, pas juste supprimé.

    Utilise deux signaux :
    1. Contexte local (société/suffixe/casse) via _has_company_context()
    2. Correspondance avec les sociétés trouvées par regex ALL-CAPS dans
       le texte (cas "Zniber Seafarm" dont une occurrence plus haut dans
       le texte est en capitales entre guillemets après "société").

    Returns:
        (persons_restants, noms_de_société_reclassés_en_ORG)
    """
    # Build lowercased set of company names from regex extraction
    regex_companies = _extract_company_names_regex(full_text)
    regex_company_texts = set(c["text"].lower() for c in regex_companies)

    kept, reclassified = [], []
    for p in persons:
        text = p.get("text", "").strip()
        if not text:
            continue
        if _has_company_context(text, p.get("start", 0), p.get("end", 0), full_text):
            reclassified.append({**p, "label": "ORG"})
        elif text.lower() in regex_company_texts:
            # Entity text matches a company name found by regex elsewhere
            # in the full text (e.g. title-case name matches ALL-CAPS
            # occurrence after "société").
            reclassified.append({**p, "label": "ORG"})
        else:
            kept.append(p)
    return kept, reclassified


def _merge_unique_orgs(orgs: List[Dict], extra: List[Dict]) -> List[Dict]:
    """Ajoute `extra` à `orgs` en évitant les doublons (même texte + position)."""
    seen = {(o.get("text", ""), o.get("start", 0)) for o in orgs}
    merged = list(orgs)
    for o in extra:
        key = (o.get("text", ""), o.get("start", 0))
        if key not in seen:
            merged.append(o)
            seen.add(key)
    return merged


def filter_organizations(orgs: List[Dict], lang: str) -> List[Dict]:
    """
    Filtre les organisations pour éliminer les faux positifs.
    Utilise les patterns ORG pré-compilés (_ORG_PATTERNS_AR/_ORG_PATTERNS_FR).
    """
    filtered = []
    patterns = _ORG_PATTERNS_AR if lang == "ar" else _ORG_PATTERNS_FR
    keywords = ORG_KEYWORDS_AR if lang == "ar" else ORG_KEYWORDS_FR

    for org in orgs:
        text = org.get("text", "").strip()
        if not text:
            continue

        # 1. Supprimer les entités trop courtes (1 mot) qui sont des lieux ou prénoms
        words = text.split()
        if len(words) == 1:
            if lang == "ar" and text in COMMON_ORG_WORDS_AR:
                continue
            if lang == "fr" and text.lower() in COMMON_ORG_WORDS_FR:
                continue

        # 2. Supprimer le bruit OCR
        if _OCR_NOISE.search(text):
            continue

        # 3. Garder seulement les organisations qui contiennent des mots-clés institutionnels
        if not any(p.search(text) for p in patterns):
            continue

        # 4. Supprimer les organisations trop longues (> 5 mots) sauf si elles contiennent des mots-clés
        if len(words) > 5 and not any(p.search(text) for p in patterns):
            continue

        filtered.append(org)

    return filtered


def _has_legal_entities(article: dict) -> bool:
    """Vérifie si l'article contient au moins une entité juridique
    (LOI, DECRET, DAHIR, ARRETE, BULLETIN_OFFICIEL)."""
    return any(
        e.get("label") in LEGAL_TEXT_LABELS
        for e in article.get("entities", [])
    )


def _batch_extract_persons_orgs(
    article_texts: list[str],
    lang: str,
) -> list[tuple[list, list]]:
    """
    Exécute le NER statistique UNE SEULE FOIS sur tous les textes d'articles
    concaténés, puis distribue les entités (personnes, organisations) à chaque
    article selon leur position dans le texte combiné.

    Retourne une liste de tuples (persons, orgs_stat) de même longueur que
    article_texts — les positions des entités sont ajustées pour être
    relatives au texte de chaque article.

    Si le NER statistique est indisponible (ImportError), retourne des
    listes vides sans lever d'exception.
    """
    if not article_texts or not any(t.strip() for t in article_texts):
        return [([], []) for _ in article_texts]

    # Concaténer les textes avec un séparateur et calculer les offsets
    SEP = "\n<ARTICLE_BOUNDARY>\n"
    combined = SEP.join(article_texts)

    offsets = [0]
    for i, text in enumerate(article_texts):
        if i > 0:
            offsets.append(offsets[-1] + len(SEP))
        offsets[-1] += len(text)
    # offsets[i] = fin du texte de l'article i dans combined

    try:
        all_persons, all_orgs = _NER_MODULES[lang].extract_persons_orgs(combined)
    except ImportError:
        if lang not in _NER_UNAVAILABLE_WARNED:
            _NER_UNAVAILABLE_WARNED.add(lang)
            logger.warning(
                f"NER statistique ({lang}) indisponible — "
                f"personnes/organisations omises."
            )
        return [([], []) for _ in article_texts]

    # Déterminer à quel article appartient chaque entité
    def _article_index(char_pos: int) -> int:
        for i in range(len(article_texts)):
            start = 0 if i == 0 else offsets[i - 1] + len(SEP)
            end = offsets[i]
            if start <= char_pos < end:
                return i
        return -1

    results = [([], []) for _ in article_texts]
    for p in all_persons:
        idx = _article_index(p.get("start", 0))
        if idx >= 0:
            offset = 0 if idx == 0 else offsets[idx - 1] + len(SEP)
            p = {**p, "start": p.get("start", 0) - offset, "end": p.get("end", 0) - offset}
            results[idx][0].append(p)
    for o in all_orgs:
        idx = _article_index(o.get("start", 0))
        if idx >= 0:
            offset = 0 if idx == 0 else offsets[idx - 1] + len(SEP)
            o = {**o, "start": o.get("start", 0) - offset, "end": o.get("end", 0) - offset}
            results[idx][1].append(o)

    return results


# ============================================================================
# FONCTIONS PRINCIPALES
# ============================================================================

def extract_dates_from_entities(article: Dict, lang: str) -> List[Dict]:
    """
    Extrait les dates des entités de l'article.
    """
    date_entities = [
        normalize_entity(e) for e in article.get("entities", []) 
        if e.get("label") in DATE_LABELS
    ]
    return date_entities


def enrich_article_json(
    article: dict,
    full_text: str,
    doc_id: str,
    lang: str = "fr",
    pre_extracted_persons: list | None = None,
    pre_extracted_orgs: list | None = None,
) -> dict:
    """
    Enrichit un article avec citations résolues, personnes, organisations et dates.

    Optimisations :
      - Si pre_extracted_persons/orgs sont fournis (batch NER), le NER
        statistique n'est pas ré-exécuté.
      - Si l'article n'a aucune entité juridique (LOI/DECRET/DAHIR/ARRETE/
        BULLETIN_OFFICIEL), le NER statistique est sauté (~60% des articles
        dans un BO typique, où le bruit NER serait de toute façon prédominant).

    Args:
        article: dictionnaire de l'article (sortie de l'étape 3)
        full_text: texte brut de l'article
        doc_id: identifiant du document source
        lang: "fr" ou "ar"
        pre_extracted_persons: résultats NER personnes pré-calculés (batch)
        pre_extracted_orgs: résultats NER organisations pré-calculés (batch)

    Returns:
        Dictionnaire enrichi avec citations, personnes, organisations, dates
    """
    article = copy.deepcopy(article)

    if lang not in _CITATION_MODULES:
        raise ValueError(f"Langue non supportée : {lang!r} (attendu 'fr' ou 'ar')")

    logger.info(f"Enrichissement de l'article {article.get('number', '?')} (lang={lang})")

    # ============================================================
    # 4a : Extraction des citations
    # ============================================================
    raw_citations = [
        {"text": t, "start": s, "end": e}
        for s, e, t in _CITATION_MODULES[lang].find_article_citations(full_text)
    ]
    logger.debug(f"  {len(raw_citations)} citations brutes trouvées")

    # ============================================================
    # 4a-bis : Résolution des citations
    # ============================================================
    legal_entities = [
        normalize_entity(e) for e in article.get("entities", []) if e.get("label") in LEGAL_TEXT_LABELS
    ]
    resolved = resolve_citations(
        raw_citations, legal_entities, doc_id=doc_id, full_text=full_text, lang=lang
    )
    citations_json = [
        {
            "text": c.text,
            "target_label": c.target_label,
            "target_text": c.target_text,
            "resolved": c.resolved,
        }
        for c in resolved
    ]
    logger.debug(f"  {sum(1 for c in resolved if c.resolved)} citations résolues")

    # ============================================================
    # 4b : NER statistique (personnes + organisations)
    # ============================================================
    if pre_extracted_persons is not None and pre_extracted_orgs is not None:
        persons, orgs_stat = pre_extracted_persons, pre_extracted_orgs
    elif not _has_legal_entities(article):
        # Aucune entité juridique → le NER statistique ne donnerait
        # quasiment que du bruit → on saute l'appel coûteux
        persons, orgs_stat = [], []
        logger.debug(f"    Pas d'entité juridique → NER statistique sauté")
    else:
        try:
            persons, orgs_stat = _NER_MODULES[lang].extract_persons_orgs(full_text)
        except ImportError as e:
            if lang not in _NER_UNAVAILABLE_WARNED:
                _NER_UNAVAILABLE_WARNED.add(lang)
                logger.warning(
                    f"  NER statistique ({lang}) indisponible ({e}) — "
                    f"personnes/organisations statistiques omises pour tous les "
                    f"articles de cette langue, le reste de l'enrichissement "
                    f"(dates, citations, institutions) continue normalement. "
                    f"(ce message ne s'affiche qu'une fois par langue par exécution)"
                )
            persons, orgs_stat = [], []

    # Organisations déjà connues via les règles (MINISTERE, INSTITUTION)
    existing_orgs = [
        normalize_entity(e) for e in article["entities"]
        if e.get("label") in ("MINISTERE", "INSTITUTION")
    ]
    orgs_merged = merge_with_rule_based_entities(orgs_stat, existing_orgs)

    # ============================================================
    # 4c : Extraction des dates (à partir des entités)
    # ============================================================
    dates = extract_dates_from_entities(article, lang)
    dates = filter_dates(dates, lang)

    # ============================================================
    # 4d : Filtrage des personnes et organisations
    # ============================================================
    # Extract company names via regex (catches ALL-CAPS names like
    # "TAIBA SEAFOOD" that spaCy NER never flags) and merge into orgs.
    regex_companies = _extract_company_names_regex(full_text, lang=lang)
    orgs_merged = _merge_unique_orgs(orgs_merged, regex_companies)

    # Reclassify company names BEFORE filter_persons() — the statistical NER
    # frequently mislabels long company names (e.g. "Société Nouvelle des
    # Entreprises de Construction...", 9+ words) as PERSON.  If
    # filter_persons() runs first, its >5‑word rejection rule drops them
    # before reclassify_company_names() ever gets a chance to relabel them
    # to ORG.
    persons, company_orgs = reclassify_company_names(persons, full_text)
    persons = filter_persons(persons, lang)
    persons = gazetteer_filter_persons(persons, full_text)
    orgs_merged = filter_organizations(orgs_merged, lang)
    # Les noms de société reclassés sont déjà validés par leur contexte
    # ("société ...") — ils n'ont pas besoin de repasser par
    # filter_organizations(), dont le filtre par mots-clés institutionnels
    # (office, ministère...) les rejetterait à tort.
    orgs_merged = _merge_unique_orgs(orgs_merged, company_orgs)

    # ============================================================
    # 4e : Normalisation finale
    # ============================================================
    persons = [normalize_entity(p) for p in persons]
    orgs_merged = [normalize_entity(o) for o in orgs_merged]
    dates = [normalize_entity(d) for d in dates]

    # Deduplicate organizations that differ only by casing (e.g.
    # "ministre de l'économie et des finances" vs "MINISTRE DE
    # L'ÉCONOMIE ET DES FINANCES").  Keep the first-encountered variant.
    _seen_org: set[str] = set()
    _deduped_orgs = []
    for o in orgs_merged:
        key = o.get("text", "").lower()
        if key not in _seen_org:
            _seen_org.add(key)
            _deduped_orgs.append(o)
    orgs_merged = _deduped_orgs

    # Deduplicate ALL entities that differ only by casing (e.g.
    # "ministre de l'économie et des finances" vs "MINISTRE DE
    # L'ÉCONOMIE ET DES FINANCES"). Keep the first-encountered casing
    # variant for each lowercase key.
    _seen_ent: set[str] = set()
    _deduped_ents = []
    for e in article.get("entities", []):
        key = e.get("text", "").lower()
        if key not in _seen_ent:
            _seen_ent.add(key)
            _deduped_ents.append(e)
    article["entities"] = _deduped_ents

    # ============================================================
    # 4f : Mise à jour de l'article
    # ============================================================
    article["text"] = full_text
    article["citations"] = citations_json
    article["persons"] = persons
    article["organizations"] = orgs_merged
    article["dates"] = dates

    logger.info(f"  {len(persons)} personnes, {len(orgs_merged)} organisations, {len(dates)} dates (après filtrage)")
    return article


def enrich_articles_batch(
    articles: list[dict],
    article_texts: list[str],
    doc_id: str,
    lang: str = "fr",
) -> list[dict]:
    """
    Enrichit tous les articles d'un document en une seule passe.

    Optimisations :
      - Le NER statistique est exécuté UNE SEULE FOIS sur l'ensemble des
        textes d'articles (via _batch_extract_persons_orgs) et les
        résultats sont distribués à chaque article.
      - Les articles sans entité juridique ne reçoivent pas d'appel NER
        du tout (voir _has_legal_entities).

    Args:
        articles: liste de dictionnaires articles (sortie de l'étape 3)
        article_texts: liste des textes bruts correspondants (même ordre)
        doc_id: identifiant du document source
        lang: "fr" ou "ar"

    Returns:
        Liste de dictionnaires enrichis (même ordre que articles).
    """
    # 1) Batch NER — une seule passe pour tous les articles
    batch_results = _batch_extract_persons_orgs(article_texts, lang)

    # 2) Enrichir chaque article avec ses résultats NER pré-calculés
    enriched = []
    for article, text, (persons, orgs) in zip(articles, article_texts, batch_results):
        enriched.append(
            enrich_article_json(
                article, text,
                doc_id=doc_id, lang=lang,
                pre_extracted_persons=persons,
                pre_extracted_orgs=orgs,
            )
        )

    return enriched


# ============================================================================
# TEST UNITAIRE (pour débogage)
# ============================================================================

if __name__ == "__main__":
    import json

    # Exemple d'article pour test
    sample_article = {
        "number": "5",
        "entities": [
            {"text": "loi n° 03-25", "label": "LOI", "start": 40, "end": 52},
            {"text": "7 ماي 2026", "label": "DATE_HIJRI", "start": 100, "end": 110}
        ],
    }
    sample_text = (
        "Conformément à l'article 5 de la loi n° 03-25, le ministre "
        "de l'Intérieur, Ahmed Benali, informe l'article 12 ci-dessus. "
        "La loi a été publiée le 7 ماي 2026."
    )
    
    # Test en français
    result_fr = enrich_article_json(sample_article, sample_text, doc_id="BO_7506_Fr", lang="fr")
    print("=== RÉSULTAT FRANÇAIS ===")
    print(json.dumps(result_fr, ensure_ascii=False, indent=2))

    # Test en arabe (version simplifiée)
    sample_article_ar = {
        "number": "الثانية",
        "entities": [
            {"text": "قانون رقم 03.25", "label": "LOI", "start": 40, "end": 52},
            {"text": "7 ماي 2026", "label": "DATE_HIJRI", "start": 100, "end": 110}
        ],
    }
    sample_text_ar = (
        "طبقاً للمادة 5 من القانون رقم 03.25، الوزير "
        "عزيز اخنوش، يبلغ المادة 12 أعلاه. "
        "نشر القانون في 7 ماي 2026."
    )
    
    result_ar = enrich_article_json(sample_article_ar, sample_text_ar, doc_id="BO_7506_Ar", lang="ar")
    print("\n=== RÉSULTAT ARABE ===")
    print(json.dumps(result_ar, ensure_ascii=False, indent=2))