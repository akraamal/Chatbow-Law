"""
cleaner_ar.py
--------------
Pipeline complet de nettoyage du texte arabe extrait des PDF juridiques
marocains, similaire à cleaner_fr.py mais adapté aux spécificités de
l'arabe (voir arabic_utils.py pour le détail des normalisations).
"""

import re
import unicodedata

from src.preprocessing.arabic_utils import normalize_arabic_text

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200E\u200F]")

# Séquences de points au milieu du texte (artefact d'extraction de
# tables des matières/sommaires mêlés au corps du texte)
INLINE_DOT_SEQUENCE = re.compile(r"\.{4,}")

# En-tête récurrent typique du Bulletin Officiel en arabe :
# "الجريدة الرسمية" (BULLETIN OFFICIEL en arabe) et la ligne de référence
# numéro/date qui l'accompagne généralement.
_HEADER_COMBINED_AR = re.compile(
    r"^\s*(?:"
    r"الجريدة\s+الرسمية"           # "الجريدة الرسمية" (BULLETIN OFFICIEL en arabe)
    r"|"
    r"\d{1,4}"                     # numéro de page seul
    r")\s*$",
    re.MULTILINE,
)

# Correction du bug de transposition ال/م confirmé sur l'ensemble du corpus
# arabe (data/raw/ar/*.pdf, 6 documents, 900+ occurrences rien que pour
# "مالادة" au lieu de "المادة") : tout mot qui devrait commencer par "الم"
# (article défini ال + mot commençant par م — المادة, المدير, المكلف,
# المالية, ...) est systématiquement extrait avec les 3 premières lettres
# permutées en "مال" (confirmé aussi bien pour du vocabulaire institutionnel
# que pour "المالية"/finances, un mot très fréquent dans ce corpus). Motif
# stable sur des centaines d'occurrences sans exception, ce qui indique une
# table ToUnicode corrompue pour cette combinaison de lettres dans la
# police intégrée au PDF — pas un problème d'outil d'extraction.
#
# Remarque historique : une version antérieure de ce correctif ciblait le
# motif inverse ("امل" -> "الم"), documenté comme confirmé sur
# BO_7506_Ar.pdf. Le motif réellement produit a changé après l'introduction
# de la reconstruction BiDi caractère par caractère dans
# pdf_extractor._fix_bidi_line() (qui corrige un bug distinct, l'ordre des
# mots dans les lignes RTL) : cette reconstruction interagit avec la même
# corruption de police sous-jacente et en modifie la manifestation. Cette
# version cible donc le motif "مال" effectivement observé aujourd'hui.
#
# Portée : on ne corrige "مال" qu'EN DÉBUT DE MOT (après un espace, un
# début de texte, ou un préfixe arabe d'une à trois lettres collé —
# و/ف/ب/ك/ل, "et/donc/avec/comme/pour"), ET seulement quand au moins une
# lettre arabe supplémentaire suit (pour ne jamais toucher un éventuel
# "مال" isolé, le mot "argent" à part entière) — un "مال" isolé n'apparaît
# d'ailleurs jamais dans ce corpus (vérifié), tout comme aucune occurrence
# légitime de "مالية" sans article n'y a été trouvée : chaque occurrence de
# "مالية" observée fait partie du mot corrompu "مالالية" (= "المالية" avec
# les 3 premières lettres permutées).
_WORD_BOUNDARY = r"(?:^|[\s«»\"'.\-\)\(:؛,،])"

LAM_MEEM_SWAP_PATTERN = re.compile(
    rf"({_WORD_BOUNDARY})([وفبكل]{{0,3}})مال(?=[\u0621-\u064A])", re.MULTILINE
)


def fix_lam_meem_transposition(text: str) -> str:
    """
    Corrige la transposition ال/م en début de mot (voir commentaire
    ci-dessus). Idempotent par construction : une fois "مال" remplacé par
    "الم" à une position donnée, cette position ne peut plus matcher à
    nouveau (contrairement à l'ancienne version, dont le garde-fou
    vérifiait la présence de "الم" n'importe où dans TOUT le texte — cette
    condition étant presque toujours vraie ailleurs dans un texte de
    plusieurs milliers de caractères, elle désactivait silencieusement le
    correctif sur la quasi-totalité des documents réels).
    """
    return LAM_MEEM_SWAP_PATTERN.sub(lambda m: m.group(1) + m.group(2) + "الم", text)


# Correction OCR : "الئحة" (3 lettres : ا ل ئ ح ة) → "اللائحة" (5 lettres : ا ل ل ا ئ ح ة)
# Le shadda sur le ل de "اللائحة" fait que Tesseract saute le ل, produisant "الئحة".
# On utilise une substitution exacte (pas de regex) pour rester conservateur.
LAAIHA_CORRECTIONS = {
    "الئحة": "اللائحة",
    "بالئحة": "باللائحة",
    "والئحة": "واللائحة",
    "فلئحة": "فللائحة",
}

def fix_laaiha_ocr_artifact(text: str) -> str:
    for wrong, correct in LAAIHA_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_unicode_ar(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = CONTROL_CHARS.sub("", text)
    return text


def remove_page_headers_footers_ar(text: str) -> str:
    """
    Retire l'en-tête récurrent "الجريدة الرسمية" (BULLETIN OFFICIEL) et les
    numéros de page isolés en un seul passage regex. À compléter si d'autres
    formulations récurrentes sont observées sur un vrai corpus arabe (le motif
    exact peut varier selon l'édition du Bulletin Officiel).
    """
    return _HEADER_COMBINED_AR.sub("", text)


def remove_dot_leader_lines_ar(text: str) -> str:
    """
    Remplace les séquences de points (4+) par un espace — ces artefacts
    proviennent de tables des matières/sommaires du BO mêlés au texte lors
    de l'extraction PDF. Exemple : "المادة......................... 12"
    """
    return INLINE_DOT_SEQUENCE.sub(" ", text)


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


_GREG_MONTHS_AR = (
    "يناير", "فبراير", "مارس", "ابريل", "ماي", "مايو", "يونيو", "يوليوز",
    "يوليو", "غشت", "اغسطس", "شتنبر", "سبتمبر", "اكتوبر", "نونبر", "نوفمبر",
    "دجنبر", "ديسمبر",
)
_GREG_MONTHS_AR_SORTED = sorted(set(_GREG_MONTHS_AR), key=len, reverse=True)


def _alef_insensitive(word: str) -> str:
    """Remplace tout alef initial (ا/أ/إ/آ/ٱ) par une classe de caractères
    couvrant toutes ses variantes, pour matcher que la normalisation
    d'alef (normalize_alef) ait déjà tourné ou non sur le texte."""
    if word and word[0] in "اأإآٱ":
        return "[اأإآٱ]" + re.escape(word[1:])
    return re.escape(word)


_GREG_MONTHS_PATTERN = "|".join(_alef_insensitive(m) for m in _GREG_MONTHS_AR_SORTED)

# Quirk résiduel connu de pdf_extractor._fix_bidi_line() : cette fonction
# corrige l'ordre des mots et l'intégrité des nombres pour les lignes RTL,
# mais ne rattache pas toujours les parenthèses au bon groupe voisin dans
# le motif très fréquent "<jour hégirien> <mois hégirien> <année
# hégirienne> (<jour grégorien> <mois grégorien> <année grégorienne>)" —
# le résultat brut ressort par exemple "...ذي القعدة 22) 1447 أبريل
# (2026" au lieu de "...ذي القعدة 1447 (22 أبريل 2026)". Le motif est
# suffisamment régulier et prévisible dans les textes juridiques marocains
# pour être corrigé par une regex ciblée, plutôt que d'implémenter la
# résolution complète des caractères neutres de l'algorithme BiDi (UAX #9,
# règles N1/N2) pour un seul cas d'usage.
HIJRI_GREGORIAN_PAREN_SWAP = re.compile(
    r"(?P<greg_day>[\d٠-٩]{1,2})\)\s*"
    r"(?P<mid>[\d٠-٩]{3,4}\s+)?"
    rf"(?P<greg_month>{_GREG_MONTHS_PATTERN})\s*[:.،\-]?\s*"
    r"\(\s*(?P<greg_year>[\d٠-٩]{4})"
)


def fix_hijri_gregorian_paren_placement(text: str) -> str:
    """Remet les parenthèses à leur place dans le motif date hégirienne +
    date grégorienne entre parenthèses (voir HIJRI_GREGORIAN_PAREN_SWAP).

    Deux variantes observées selon les documents : l'année hégirienne
    apparaît soit collée juste après la parenthèse mal placée (\"22)
    1447 أبريل (2026\"), soit déjà correctement située avant tout le
    groupe (\"1429 23)اكتوبر (2008\") — le groupe optionnel \"mid\" capture
    la première variante sans consommer la seconde.
    """
    def _repl(m):
        mid = m["mid"] or ""
        return f"{mid}({m['greg_day']} {m['greg_month']} {m['greg_year']})"

    return HIJRI_GREGORIAN_PAREN_SWAP.sub(_repl, text)


def clean_arabic_text(
    text: str,
    remove_headers: bool = True,
    fix_lam_meem_bug: bool = True,
    fix_hijri_gregorian_parens: bool = True,
    fix_laaiha_artifact: bool = True,
    remove_dot_leaders: bool = True,
    apply_tashkeel_removal: bool = True,
    apply_tatweel_removal: bool = True,
    apply_alef_normalization: bool = True,
    apply_digit_normalization: bool = True,
) -> str:
    """
    Pipeline complet de nettoyage du texte arabe.

    Args:
        text: texte brut extrait (depuis data/interim/ar/*.txt).
        remove_headers: si True, retire les en-têtes/pieds de page répétés.
        fix_lam_meem_bug: si True (par défaut), corrige la transposition
            ل/م en début de mot (المادة, المشار, المدير...) causée par une
            table ToUnicode corrompue dans certains PDF du Bulletin
            Officiel — voir LAM_MEEM_SWAP_PATTERN ci-dessus. Appliqué tôt,
            avant la suppression des en-têtes : les regex d'en-têtes
            (remove_page_headers_footers_ar) et la segmentation en
            articles en aval s'attendent à l'orthographe correcte.
        fix_hijri_gregorian_parens: si True (par défaut), corrige le
            placement des parenthèses dans le motif "date hégirienne
            (date grégorienne)" — voir HIJRI_GREGORIAN_PAREN_SWAP ci-dessus.
        fix_laaiha_artifact: si True (par défaut), corrige "الئحة" →
            "اللائحة" causé par le shadda sur le ل que Tesseract omet.
        remove_dot_leaders: si True, remplace les séquences de points (4+)
            par un espace.
        apply_*: voir arabic_utils.normalize_arabic_text pour le détail de
            chaque option de normalisation.

    Returns:
        Texte nettoyé, prêt pour la segmentation en articles.
    """
    text = normalize_line_endings(text)
    text = normalize_unicode_ar(text)

    if fix_lam_meem_bug:
        text = fix_lam_meem_transposition(text)

    if remove_headers:
        text = remove_page_headers_footers_ar(text)

    if remove_dot_leaders:
        text = remove_dot_leader_lines_ar(text)

    text = normalize_arabic_text(
        text,
        apply_tashkeel_removal=apply_tashkeel_removal,
        apply_tatweel_removal=apply_tatweel_removal,
        apply_alef_normalization=apply_alef_normalization,
        apply_digit_normalization=apply_digit_normalization,
    )

    # Après la normalisation de l'alef (أ/إ/آ/ٱ -> ا) : les mois grégoriens
    # de HIJRI_GREGORIAN_PAREN_SWAP sont écrits avec alef simple, donc cette
    # correction doit tourner sur du texte déjà normalisé pour matcher de
    # façon fiable "اكتوبر" aussi bien que la forme avec hamza d'origine.
    if fix_hijri_gregorian_parens:
        text = fix_hijri_gregorian_paren_placement(text)

    if fix_laaiha_artifact:
        text = fix_laaiha_ocr_artifact(text)

    text = collapse_blank_lines(text)

    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line.strip() != ""]
    text = "\n".join(lines)

    return text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python cleaner_ar.py <chemin_vers_fichier.txt>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        raw_text = f.read()
    cleaned = clean_arabic_text(raw_text)

    print(f"Longueur avant nettoyage : {len(raw_text)} caractères")
    print(f"Longueur après nettoyage : {len(cleaned)} caractères")
    print("\n--- Aperçu du texte nettoyé ---")
    print(cleaned[:1500])