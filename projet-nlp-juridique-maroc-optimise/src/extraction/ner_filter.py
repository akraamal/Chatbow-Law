"""
Filtre post-NER commun. Placez dans src/extraction/ner_filter.py
"""

import re

# Numéros de page collés à la fin d'une entité (ex: "صندوق...7497").
# Se déclenche UNIQUEMENT si les chiffres sont immédiatement accolés au
# texte (pas d'espace avant), pour ne pas toucher aux années des dates
# comme "12 septembre 2019".
_TRAILING_PAGE_NUMBER = re.compile(r"(?<![\s\d_\-])(\d{3,4})\s*$")

# Patterns de fragments incomplets (entité qui se termine par un préfixe
# sans valeur, ex: "loi n°" au lieu de "loi n° 03-25")
_INCOMPLETE_ENTITY_PATTERN = re.compile(
    r"(?:n[°oº]\s*|رقم\s*|عدد\s*)$",
)

# Listes noires (minuscules)
BLACKLIST_PERSON = {
    "matériel", "charrue", "tracteur", "broyeur", "semoir",
    "moissonneuse", "batteuse", "faucheuse", "récolteuse",
    "type", "plafond", "taux", "montant", "nombre", "unité",
    "engagement", "paiement", "décision", "état", "ordre",
    "facture", "bon", "quittance", "reçu", "procès-verbal",
    "avenant", "marché", "contrat", "convention", "bordereau",
    "décompte", "attestation", "certificat", "copie", "extrait",
    "jugement", "arrêté", "dahir", "décret", "loi", "bulletin",
    "n", "ro", "vule",  # fragments isolés
    "approche", "thématique", "satellite", "orientation",
    "douar", "lambert", "fait", "recto", "verso", "centre",
    "crassostrea", "ruditapes", "palais", "portail", "ornement",
    "modernité", "souveraineté", "ouverture", "violet",
    "polymère", "substrat", "perspective", "réseau", "fibre",
    "numérique", "digitale", "stylisation", "rayon", "solaire",
    "dénomination", "institut", "émission", "représentation",
    "projet", "thème", "liste", "arabesque", "faciale",
    "transparente", "transformation", "supérieure", "inférieure",
    # Mots de liaison / prépositions françaises souvent mal classifiés
    "dans", "sur", "pour", "avec", "sans", "selon", "chez",
    "cette", "ces", "tous", "toutes", "chaque", "entre",
    "après", "avant", "durant", "pendant", "non", "outre",
    "notamment", "toutefois", "cependant", "néanmoins", "par", "sous",
    # Mots techniques / en-têtes de tableau / coordonnées
    "technique", "montant", "largeur", "latitude", "longitude",
    "parcelle", "bornes", "borne", "borncs", "zone",
    "ancien", "argoub", "azib", "labrareq", "moulay", "rachid",
    "bloc", "hay", "saint", "jacques", "rabat", "royal",
    "pêche", "développement", "environnemental", "homme",
    "calastropaiques", "déchets", "deraton", "lhuître",
    "crassostrea", "crassosfrea", "pecten", "maximus", "perna",
    "gracilaria", "gracili", "mytilus", "galloprovincialis",
    "mo", "rem", "vu", "téservés",
    "pamélioration", "péconomie", "paccomplissement",
    "leconseilconsidère", "ilimporte", "finforma",
    "monte", "parrêté", "pêche maritime",
    "de", "roi", "fait",
}

BLACKLIST_ORG = {
    "ordre de service", "ordre de mission",
    "fiche d'engagement", "état d'engagement", "fiche navette",
    "bordereau des prix", "procès-verbal", "compte d'emploi",
    "état des sommes dues", "bulletin officiel",
    "trésorerie", "budget", "fonds", "recette", "dépense",
    "crédit", "emprunt", "ministère", "direction", "service",
}

# Références légales et dates : intrinsèquement porteuses de numéros et de
# dates ("dahir n° 1-14-139 du 16 chaoual 1435 (13 août 2014)" est à moins
# de 50 % de lettres).  Le ratio alphabétique ne s'applique PAS à elles,
# sinon toutes ces entités sont rejetées (0 LOI/DAHIR/DATE conservés —
# observé sur BO_7522 : 26 entités seulement pour 141 pages).
_DIGIT_HEAVY_LABELS = {
    "DAHIR", "LOI", "DECRET", "ARRETE", "BULLETIN_OFFICIEL",
    "CIRCULAIRE", "DATE_HIJRI", "DATE_GREGORIAN", "MONEY",
}


# Espacements internes (retours à la ligne sautés par un pattern — ex.
# une classe négative [^»] matche \n) : à aplatir dans le TEXTE affiché,
# pas dans le pattern (start/end restent les offsets dans le texte source).
_WS_RUN = re.compile(r"\s+")


def clean_entity_text(entity: dict) -> dict:
    """
    Nettoie le texte d'une entité :
      - aplatit les espaces internes (retours à la ligne d'extraction) ;
      - retire les numéros de page résiduels (4 chiffres collés à la fin,
        courants dans la table des matières du BO).
    start/end restent des offsets valides dans le texte source.
    """
    text = entity.get("text", "") or ""
    text = _WS_RUN.sub(" ", text).strip()
    entity["text"] = text
    m = _TRAILING_PAGE_NUMBER.search(text)
    if m:
        trailing = m.group(1)
        # Ne pas couper si le texte entier n'est QUE le numéro
        if text.strip() != trailing:
            new_text = text[:m.start()].strip()
            entity["text"] = new_text
            entity["end"] = entity["start"] + len(new_text)
    return entity


def is_valid_entity(text: str, label: str, context: str = "") -> bool:
    text = text.strip()
    norm = text.lower().strip(".,;:!?")
    
    # Trop court
    if len(text) < 3:
        return False

    # Fragment incomplet (ex: "loi n°" sans numéro)
    if _INCOMPLETE_ENTITY_PATTERN.search(norm):
        return False
    
    # Pas assez alphabétique (uniquement pour les entités "génériques" :
    # les références légales et dates contiennent légitimement des numéros)
    if (label not in _DIGIT_HEAVY_LABELS
            and sum(1 for c in text if c.isalpha()) / len(text) < 0.5):
        return False
    
    # Liste noire
    if label == "PERSON" and norm in BLACKLIST_PERSON:
        return False
    if label == "ORGANIZATION" and norm in BLACKLIST_ORG:
        return False
    
    # Juste un nombre
    if text.isdigit():
        return False
    
    return True


def filter_entities(entities: list, context: str = "") -> list:
    cleaned = [clean_entity_text(e) for e in entities]
    return [e for e in cleaned if is_valid_entity(e.get("text", ""), e.get("label", ""), context)]


def align_entity_text(entity: dict, source_text: str):
    """
    Repositionne une entité dans *source_text* et restaure son ``text``
    comme le slice EXACT ``source_text[start:end]``.

    Ceci rétablit l'invariant requis par les consommateurs
    (``source_text.find(entity["text"])``) :
      - si les offsets fournis sont déjà dans le texte, on restaure le
        slice exact (avant correctif, ``clean_entity_text`` aplatissait les
        retours à la ligne en espaces SANS ajuster les offsets → le text
        stocké différait du slice source et find() renvoyait -1) ;
      - si ``start == -1`` (entité propagée depuis un préambule de décret),
        une recherche tolérante aux espaces retrouve le fragment ;
      - en dernier recours, une recherche par le numéro de référence.

    Returns:
        l'entité mise à jour, ou ``None`` si introuvable (à écarter).
    """
    start = entity.get("start", -1)
    end = entity.get("end", -1)

    def _restore(s, e):
        entity["start"] = s
        entity["end"] = e
        entity["text"] = source_text[s:e]
        return entity

    # Chemin rapide : offsets déjà dans le texte source.
    if (start != -1 and end != -1 and 0 <= start < len(source_text)
            and end > start and end <= len(source_text)):
        return _restore(start, end)

    text = (entity.get("text") or "").strip()
    if not text:
        return None

    # Le texte aplati (retours à la ligne → espaces) ne matche pas
    # `source_text` tel quel : recherche tolérante (n'importe quels \s+).
    parts = _WS_RUN.split(text)
    if not parts:
        return None
    pattern = re.compile(r"\s+".join(re.escape(p) for p in parts))
    m = pattern.search(source_text)
    if m:
        return _restore(m.start(), m.end())

    # Dernier recours : le numéro de référence seul (ex. "731.25").
    nums = re.findall(r"[\d]+(?:[-–.][\d]+){1,2}", text)
    for n in nums:
        pos = source_text.find(n)
        if pos >= 0:
            return _restore(pos, pos + len(n))
    return None