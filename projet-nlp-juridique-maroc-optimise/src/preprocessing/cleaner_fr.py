"""
cleaner_fr.py
--------------
Nettoyage du texte français extrait des PDF juridiques marocains (Bulletin
Officiel notamment).

Problèmes concrets identifiés sur un vrai document (BO n° 7500) :
  - Retours à la ligne Windows (\\r\\n) et caractères de contrôle parasites
    (ex: \\x08) issus de l'extraction PDF
  - Espaces insécables (\\xa0) au lieu d'espaces normaux
  - En-têtes/pieds de page répétés sur CHAQUE page :
        "788"                                    (numéro de page seul)
        "BULLETIN OFFICIEL"
        "Nº 7500 – 28 chaoual 1447 (16-4-2026)"
  - Lignes de sommaire avec des points de suite ("......................790")
  - Espacement incohérent autour de la ponctuation (apostrophes typographiques,
    espaces avant : ; ! ? qui sont des conventions typographiques françaises
    mais peuvent gêner le NLP si mal normalisées)
"""

import re
import unicodedata


# --- Patterns d'en-tête/pied de page combinés en un seul passage ---
# Inclut : référence Nº/date, "BULLETIN OFFICIEL", fragments d'en-tête,
# et numéros de page isolés.
_HEADER_COMBINED_FR = re.compile(
    r"^\s*(?:"
    r"N[ºo°]\s*\d+\s*[–\-—]\s*\d+\s+\w+\s+\d{4}\s*\(.*?\)"  # "Nº 7500 – 28 chaoual 1447 (...)"
    r"|"
    r"BULLETIN OFFICIEL"                                         # "BULLETIN OFFICIEL" seul
    r"|"
    r".*?FFICIEL\s+N[ºo°]\s*\d+\s*[–\-—]"                      # fragments d'en-tête coupés
    r"|"
    r"\d{1,4}"                                                   # numéro de page seul
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# Lignes de sommaire avec points de suite : "Edition générale...................790"
DOT_LEADER_LINE = re.compile(r"^.*?\.{4,}\s*\d*\s*$", re.MULTILINE)
# Séquences de points au milieu d'une ligne (4+ points consécutifs) provenant
# de la fusion d'une ligne de sommaire avec du texte adjacent lors de
# l'extraction PDF. Ex : "L'opération ................................ recours"
INLINE_DOT_SEQUENCE = re.compile(r"\.{4,}")

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200E\u200F]")

# Caractères arabes isolés qui débordent de la colonne arabe voisine lors
# de l'extraction PDF. On supprime tout caractère dans les plages Unicode
# arabes (lettres, signes, formes de présentation) car un texte français
# n'en contient légitimement aucun — les rares mots arabes qu'on voudrait
# conserver (ex: noms de mois hégiriens) sont déjà capturés par les regex
# de dates AVANT ce nettoyage.
ARABIC_CHARS = re.compile(
    "[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    "\uFB50-\uFDFF\uFE70-\uFEFF"
    "]+"
)


def normalize_unicode(text: str) -> str:
    """
    Normalise le texte en forme NFC (caractères composés), remplace les
    espaces insécables par des espaces normaux, et retire les caractères de
    contrôle non imprimables issus de l'extraction PDF.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")  # espace insécable -> espace normal
    text = CONTROL_CHARS.sub("", text)
    return text

def normalize_apostrophes(text: str) -> str:
    """
    Normalise les apostrophes typographiques en apostrophe simple.
    """
    return text.replace('’', "'").replace('‘', "'")

def normalize_line_endings(text: str) -> str:
    """Convertit tous les retours à la ligne (\\r\\n, \\r) en \\n uniquement."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_page_headers_footers(text: str) -> str:
    """
    Retire les en-têtes/pieds de page répétés sur chaque page du Bulletin
    Officiel : référence du numéro/date, mention "BULLETIN OFFICIEL",
    fragments d'en-tête, numéros de page isolés, et mots en majuscules
    résiduels — le tout en un seul passage regex principal.

    Attention : le pattern des numéros de page ne matche QUE des lignes
    ne contenant que des chiffres, pour éviter de supprimer des numéros
    d'articles ou de décrets.
    """
    text = _HEADER_COMBINED_FR.sub("", text)
    # Mots isolés tout en majuscules (>=4 lettres) qui sont des fragments de
    # pied de page (ex: "NERAUX" issu de "GÉNÉRAUX") — exclus les vrais
    # marqueurs juridiques et titres légitimes.
    text = re.sub(
        r"^\s*(?!(?:ARTICLE|CHAPITRE|TITRE|LOI|DECRET|DAHIR|ARRETE|ANNEXE"
        r"|SOMMAIRE|MINISTERE|MINISTRE|BULLETIN|OFFICIEL|PREMIER|PREAMBULE"
        r"|SECTION|TABLE|CONSIDERANT|CONSIDÉRANT|CONSIDERE|VU[ES]?|ATTENDU"
        r"|APPROUVE|APPROUVÉ|FAIT|RABAT|SIGNÉ|SIGNE)\b)"
        r"[A-ZÉÈÊËÀÂÎÏÔÛÙÇ]{4,}\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def remove_dot_leader_lines(text: str) -> str:
    """
    Retire les lignes de sommaire/table des matières avec points de suite
    (ex : "Edition générale...................790") et les séquences de
    points résiduelles au milieu des lignes (ex : "L'opération ........
    recours") — artefacts de l'extraction PDF qui n'apportent pas
    d'information juridique exploitable pour l'extraction d'entités.
    """
    text = DOT_LEADER_LINE.sub("", text)
    text = INLINE_DOT_SEQUENCE.sub(" ", text)
    return text


def remove_arabic_artifacts(text: str) -> str:
    """
    Supprime les caractères arabes isolés qui ont débordé de la colonne
    arabe lors de l'extraction PDF. Nettoie aussi les marqueurs RTL/LTR
    (U+200E, U+200F) qui polluent le texte français.
    """
    return ARABIC_CHARS.sub("", text)


def collapse_blank_lines(text: str) -> str:
    """Réduit les suites de lignes vides à une seule ligne vide."""
    return re.sub(r"\n{3,}", "\n\n", text)


def clean_french_text(
    text: str,
    remove_headers: bool = True,
    remove_dot_leaders: bool = True,
    normalize_apos: bool = True,
    remove_arabic: bool = True,
) -> str:
    """
    Pipeline complet de nettoyage du texte français.

    Args:
        text: texte brut extrait (depuis data/interim/fr/*.txt).
        remove_headers: si True, retire les en-têtes/pieds de page répétés.
        remove_dot_leaders: si True, retire les lignes de sommaire à points
            de suite (utile pour le corps du texte, mais peut être désactivé
            si on veut traiter spécifiquement la page de sommaire).
        normalize_apos: si True, normalise les apostrophes typographiques.
        remove_arabic: si True, supprime les artefacts arabes résiduels.

    Returns:
        Texte nettoyé, prêt pour la segmentation en articles.
    """
    text = normalize_line_endings(text)
    text = normalize_unicode(text)

    if remove_headers:
        text = remove_page_headers_footers(text)
    if remove_dot_leaders:
        text = remove_dot_leader_lines(text)
    if normalize_apos:
        text = normalize_apostrophes(text)
    if remove_arabic:
        text = remove_arabic_artifacts(text)

    text = collapse_blank_lines(text)

    # Nettoyage final : retirer les espaces en fin de ligne, et les lignes
    # devenues vides suite aux suppressions ci-dessus
    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line.strip() != ""]
    text = "\n".join(lines)

    return text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python cleaner_fr.py <chemin_vers_fichier.txt>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        raw_text = f.read()
    cleaned = clean_french_text(raw_text)

    print(f"Longueur avant nettoyage : {len(raw_text)} caractères")
    print(f"Longueur après nettoyage : {len(cleaned)} caractères")
    print("\n--- Aperçu du texte nettoyé ---")
    print(cleaned[:1500])
