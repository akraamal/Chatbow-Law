"""
src/extraction/ocr_corrector.py
Post-OCR correction for French legal BO texts.

Applies a dictionary of known OCR error patterns observed in Moroccan
Bulletin Officiel PDFs.  Designed to be called as a lightweight
post-processing step after cleaner_fr.py.

Usage:
    from src.extraction.ocr_corrector import correct_ocr
    text = correct_ocr(text)
"""

import re

# ── Single-word substitutions ─────────────────────────────────────────────
# Format: {wrong_word: correct_word}  (case-insensitive matching)
_SINGLE_WORD_FIXES = {
    # Letter transpositions (common OCR swapping)
    "fgurant": "figurant",
    "scientifqiue": "scientifique",
    "Offcie": "Office",
    "publqiue": "publique",
    "techniqe": "technique",
    "techniqeus": "techniciens",
    "juridiqe": "juridique",
    "économiqe": "économique",
    "économiqeu": "économique",
    "admiistratif": "administratif",
    "admiistrative": "administrative",
    "adminitratif": "administratif",
    "adminitrative": "administrative",
    "conformément": "conformément",
    "dispositon": "disposition",
    "dispostion": "disposition",
    "dispostions": "dispositions",
    "péciale": "spéciale",
    "pécial": "spécial",
    "publié": "publié",
    "publie": "publié",
    "décrèt": "décret",
    "décret": "décret",
    "arrêt": "arrêté",
    "minisère": "ministère",
    "minitère": "ministère",
    "prévoit": "prévoit",
    "prévoient": "prévoient",
    "conditons": "conditions",
    "conditon": "condition",
    "conformté": "conformité",
    "conformté": "conformité",
    "exécuton": "exécution",
    "exécuton": "exécution",
    "modificaton": "modification",
    "modificatons": "modifications",
    "publcaton": "publication",
    "publcatons": "publications",
    "déclaráon": "déclaration",
    "déclaraton": "déclaration",
    "déterminaton": "détermination",
    "autorston": "autorisation",
    "indemnston": "indemnisation",
    "organisaton": "organisation",
    "organisatons": "organisations",
    "remplacé": "remplacé",
    "remplace": "remplace",
    "pénalté": "pénalité",
    "pénaltés": "pénalités",
    "responsablté": "responsabilité",
    "responsabltés": "responsabilités",
    "obligaton": "obligation",
    "obligatons": "obligations",
    "sancton": "sanction",
    "sanctons": "sanctions",
    "dispostif": "dispositif",
    "contradctoire": "contradictoire",
    "contradctoires": "contradictoires",
    "procédre": "procédure",
    "procédures": "procédures",
    "pérequis": "prérequis",
    "cotnractant": "contractant",
    "cotnractants": "contractants",
    "soustraitant": "sous-traitant",
    "soustraitants": "sous-traitants",
    "cotntrat": "contrat",
    "cotntrats": "contrats",
    "cahier": "cahier",
    "clasue": "clause",
    "clasues": "clauses",
    "conventon": "convention",
    "conventons": "conventions",
    "délbération": "délibération",
    "délbérations": "délibérations",
    "cotntrole": "contrôle",
    "cotntroler": "contrôler",
    "vérficaton": "vérification",
    "vérficatons": "vérifications",
    "cotnseil": "conseil",
    "cotnseiller": "conseiller",
    "inspecton": "inspection",
    "inspectons": "inspections",
    "évaluaton": "évaluation",
    "évaluatons": "évaluations",
    "cotnstaté": "constaté",
    "cotnstater": "constater",
    "cotnstatation": "constatation",
    "cotnstatations": "constatations",
    "sanciton": "sanction",
    "sancitons": "sanctions",
    "mutatoin": "mutation",
    "mutatoins": "mutations",
    "confgiuratoin": "configuration",
    "confgiuratoins": "configurations",
    "efeft": "effet",
    "efefts": "effets",
    "tsisu": "tissu",
    "défsi": "défi",
    "défsis": "défis",
    "modalté": "modalité",
    "modaltés": "modalités",
    "compétnce": "compétence",
    "compétnces": "compétences",
    "délbératon": "délibération",
    "délbératons": "délibérations",
    "cotnclure": "conclure",
    "cotnclu": "conclu",
    "cotnclusion": "conclusion",
    "cotnclusions": "conclusions",
    "prorogaton": "prorogation",
    "prorogatons": "prorogations",
    "abrogaton": "abrogation",
    "abrogatons": "abrogations",
    "modificaton": "modification",
    "modificatons": "modifications",
    "ratifcation": "ratification",
    "ratifcations": "ratifications",
    "homologaton": "homologation",
    "homologatons": "homologations",
    "enregstrement": "enregistrement",
    "enregstrements": "enregistrements",
    "transcpton": "transcription",
    "transcptons": "transcriptions",
    "publcaton": "publication",
    "publcatons": "publications",
    "notficaton": "notification",
    "notficatons": "notifications",
    "signficaton": "signification",
    "signficatons": "significations",
    "oppositon": "opposition",
    "oppositons": "oppositions",
    "réclamaton": "réclamation",
    "réclamatons": "réclamations",
    "cotntestaton": "contestation",
    "cotntestatons": "contestations",
    "cotnforme": "conforme",
    "cotnformément": "conformément",
    "cotnformté": "conformité",
    "dérogaton": "dérogation",
    "dérogatons": "dérogations",
    "excepton": "exception",
    "exceptons": "exceptions",
    "dérogaton": "dérogation",
    "dérogatons": "dérogations",
    "dérogatoir": "dérogatoire",
    "dérogatoirs": "dérogatoires",
    "transiton": "transition",
    "transitons": "transitions",
    "transitoir": "transitoire",
    "transitoirs": "transitoires",
    "défnton": "définition",
    "défntons": "définitions",
    "caractérstique": "caractéristique",
    "caractérstiques": "caractéristiques",
    "spécfcité": "spécificité",
    "spécfcités": "spécificités",
    "partcularité": "particularité",
    "partcularités": "particularités",
    "cotnstitue": "constitue",
    "cotnstituent": "constituent",
    "cotnstitué": "constitué",
    "cotnstitution": "constitution",
    "cotnstitutionnel": "constitutionnel",
    "cotnstitutionnelles": "constitutionnelles",
    "insttution": "institution",
    "insttutions": "institutions",
    "insttutionnel": "institutionnel",
    "insttutionnels": "institutionnels",
    "cotntenu": "contenu",
    "cotntenus": "contenus",
    "cotntenance": "contenance",
    "cotntenances": "contenances",
    "cotmpositon": "composition",
    "cotmposé": "composé",
    "cotmprend": "comprend",
    "cotmptetnt": "compétent",
    "cotmptetnce": "compétence",
    "cotmptences": "compétences",
}

# ── Pattern-based corrections ─────────────────────────────────────────────
# Applied after single-word fixes.  Each is (compiled_regex, replacement).
# These catch systematic OCR errors that can't be listed exhaustively.
_PATTERN_FIXES = [
    # "rn" → "m" confusion (common in OCR where "m" is split as "rn")
    (re.compile(r"\binterrnational\b"), "international"),
    (re.compile(r"\binterrnationale\b"), "internationale"),
    (re.compile(r"\binterrnationaux\b"), "internationaux"),
    (re.compile(r"\bgouvernement\b"), "gouvernement"),
    (re.compile(r"\bgouvernemental\b"), "gouvernemental"),
    (re.compile(r"\bgouvernementaux\b"), "gouvernementaux"),
    (re.compile(r"\bgouvernementales\b"), "gouvernementales"),

    # "ii" → "n" confusion
    (re.compile(r"\bcomiiission\b"), "commission"),
    (re.compile(r"\bcomiiissions\b"), "commissions"),
    (re.compile(r"\bprogrammme\b"), "programme"),
    (re.compile(r"\bprogrammmes\b"), "programmes"),

    # "e" → "c" / "c" → "e" confusions
    (re.compile(r"\bOffcie\b"), "Office"),
    (re.compile(r"\boffcie\b"), "office"),
    (re.compile(r"\bpratique\b"), "pratique"),
    (re.compile(r"\bpratiques\b"), "pratiques"),

    # Accent restoration patterns
    (re.compile(r"\bdecret\b", re.IGNORECASE), lambda m: "décret" if m.group(0).islower() else "Décret"),
    (re.compile(r"\barrete\b", re.IGNORECASE), lambda m: "arrêté" if m.group(0).islower() else "Arrêté"),
    (re.compile(r"\beconomie\b", re.IGNORECASE), "économie"),
    (re.compile(r"\beconomique\b", re.IGNORECASE), "économique"),
    (re.compile(r"\beconomiques\b", re.IGNORECASE), "économiques"),
    (re.compile(r"\bregion\b", re.IGNORECASE), "région"),
    (re.compile(r"\bregionale\b", re.IGNORECASE), "régionale"),
    (re.compile(r"\bregionaux\b", re.IGNORECASE), "régionaux"),
    (re.compile(r"\bregionales\b", re.IGNORECASE), "régionales"),
    (re.compile(r"\bprevu\b", re.IGNORECASE), "prévu"),
    (re.compile(r"\bprevue\b", re.IGNORECASE), "prévue"),
    (re.compile(r"\bprevus\b", re.IGNORECASE), "prévus"),
    (re.compile(r"\bprevues\b", re.IGNORECASE), "prévues"),
    (re.compile(r"\bprocede\b", re.IGNORECASE), "procédé"),
    (re.compile(r"\bproceder\b", re.IGNORECASE), "procéder"),
    (re.compile(r"\bprocede\b", re.IGNORECASE), "procédé"),
    (re.compile(r"\bprocedural\b", re.IGNORECASE), "procédural"),
    (re.compile(r"\bprocedurale\b", re.IGNORECASE), "procédurale"),
    (re.compile(r"\bproceduraux\b", re.IGNORECASE), "procéduraux"),
    (re.compile(r"\bprocedurales\b", re.IGNORECASE), "procédurales"),
    (re.compile(r"\belaborer\b", re.IGNORECASE), "élaborer"),
    (re.compile(r"\belabore\b", re.IGNORECASE), "élaboré"),
    (re.compile(r"\belaboree\b", re.IGNORECASE), "élaborée"),
    (re.compile(r"\bexecuter\b", re.IGNORECASE), "exécuter"),
    (re.compile(r"\bexecute\b", re.IGNORECASE), "exécuté"),
    (re.compile(r"\bexecution\b", re.IGNORECASE), "exécution"),
    (re.compile(r"\bexecutif\b", re.IGNORECASE), "exécutif"),
    (re.compile(r"\bexecutive\b", re.IGNORECASE), "exécutive"),
]

# ── Compile all single-word fixes into a single regex ─────────────────────
_WORD_PATTERNS = [
    (re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), correct)
    for wrong, correct in _SINGLE_WORD_FIXES.items()
]


def _apply_word_fixes(text: str) -> str:
    for pattern, replacement in _WORD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _apply_pattern_fixes(text: str) -> str:
    for pattern, replacement in _PATTERN_FIXES:
        text = pattern.sub(replacement, text)
    return text


def correct_ocr(text: str) -> str:
    """
    Apply known OCR corrections to a French legal text.

    This is a lightweight post-processing step that fixes character
    transpositions, common misspellings, and accent-loss patterns
    observed in PDF-extracted Moroccan BO texts.

    Args:
        text: Cleaned French text (output of cleaner_fr.py or similar).

    Returns:
        Corrected text.
    """
    text = _apply_word_fixes(text)
    text = _apply_pattern_fixes(text)
    return text
