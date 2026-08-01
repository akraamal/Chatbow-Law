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
# Séquences de points au milieu d'une ligne (4+ points consécutifs).  Deux
# sources possibles :
#   - artefact d'extraction PDF (ligne de sommaire fusionnée avec du texte
#     adjacent, ex. "L'opération ................................ recours") ;
#   - convention de rédaction législative : le BO publie les amendements en
#     remplaçant les passages non modifiés par une ligne de points
#     ("L'opération ................................ recours") pour signifier
#     « texte inchangé, omis ici ».
# On ne peut pas les distinguer : au lieu de les réduire à un espace (ce qui
# fait perdre le signal « contenu volontairement élidé »), on les remplace
# par un marqueur explicite conservé dans le texte.
INLINE_DOT_SEQUENCE = re.compile(r"\.{4,}")
ELISION_MARKER = " […texte non modifié…] "

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200E\u200F]")

# Caractères arabes isolés qui débordent de la colonne arabe voisine lors
# de l'extraction PDF. Le BO français cite CEPENDANT légitimement des
# passages en arabe (titres d'émissions, clauses du cahier des charges
# reprises verbatim) : ces tronçons sont donc collectés (paramètre
# ``collector`` de remove_arabic_artifacts) au lieu d'être perdus, et
# exposés à part (champ ``possible_embedded_arabic`` du JSON annoté).
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
    (ex : "Edition générale...................790") et remplace les
    séquences de points résiduelles au milieu des lignes par un marqueur
    explicite : dans le BO, ces points sont la convention d'élision des
    amendements (« texte inchangé, omis ici ») — les réduire à un espace
    rendrait un texte partiel indistinguable d'un texte tronqué par erreur.
    """
    text = DOT_LEADER_LINE.sub("", text)
    text = INLINE_DOT_SEQUENCE.sub(ELISION_MARKER, text)
    return text


def remove_arabic_artifacts(text: str, collector: list | None = None) -> str:
    """
    Nettoie les caractères arabes résiduels du texte français.

    Distingue deux cas :
      - Tronçons arabes LÉGITIMES cités entre guillemets français
        («الحقيقة في 90 دقيقة», titre d'émission dans la décision du CSN ;
        «)...( تمتنع الشركة...», clause du cahier des charges SNRT reprise
        verbatim) : CONSERVÉS dans le texte.  Ils font partie du contenu
        juridique ; les retirer laissait des guillemets vides («  ») et
        rendait le RAG incapable de répondre sur ces clauses.  L'extracteur
        PDF a déjà rétabli leur ordre logique (voir _fix_bidi_line).
      - Artefacts (débordements isolés de la colonne arabe voisine, sans
        guillemets) : SUPPRIMÉS et, si *collector* est fourni, ajoutés
        pour relecture (champ ``possible_embedded_arabic`` du JSON).

    Nettoie aussi les marqueurs RTL/LTR (U+200E, U+200F).
    """

    def _drop(match: re.Match) -> str:
        if _is_quoted_arabic(text, match.start(), match.end()):
            return match.group(0)
        if collector is not None:
            collector.append(match.group(0).strip())
        return ""

    return ARABIC_CHARS.sub(_drop, text)


def _is_arabic_char(ch: str) -> bool:
    return ARABIC_CHARS.fullmatch(ch) is not None


def _is_quoted_arabic(text: str, start: int, end: int) -> bool:
    """Un tronçon arabe est une citation légitime s'il est précédé d'un
    guillemet ouvrant « ou suivi d'un guillemet fermant » (en sautant les
    espaces, les autres caractères arabes et les marqueurs d'élision
    ``)(.`` — ex. clause du cahier des charges SNRT citée sous la forme
    «)...( تمتنع الشركة ... )...(», et titres d'émissions scindés par des
    chiffres latins : «الحقيقة في 90 دقيقة» → 3 runs arabes)."""
    i = start - 1
    while i >= 0 and (text[i].isspace() or _is_arabic_char(text[i])
                      or text[i] in ")(."):
        i -= 1
    if i >= 0 and text[i] == "«":
        return True
    j = end
    while j < len(text) and (text[j].isspace() or _is_arabic_char(text[j])
                             or text[j] in ")(."):
        j += 1
    return j < len(text) and text[j] == "»"


def collapse_blank_lines(text: str) -> str:
    """Réduit les suites de lignes vides à une seule ligne vide."""
    return re.sub(r"\n{3,}", "\n\n", text)


def clean_french_text(
    text: str,
    remove_headers: bool = True,
    remove_dot_leaders: bool = True,
    normalize_apos: bool = True,
    remove_arabic: bool = True,
    arabic_runs: list | None = None,
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
        arabic_runs: liste optionnelle recevant chaque tronçon arabe retiré
            (voir remove_arabic_artifacts) — les citations arabes légitimes
            entre guillemets sont conservées dans le texte ; seuls les
            artefacts sont collectés ici.

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
        text = remove_arabic_artifacts(text, collector=arabic_runs)

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
