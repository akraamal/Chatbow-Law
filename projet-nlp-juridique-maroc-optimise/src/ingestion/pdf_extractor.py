"""
pdf_extractor.py
----------------
Extraction du texte natif depuis les PDF.

Ce module NE décide PAS si une page est scannée.
Il extrait simplement toutes les informations disponibles.

La décision d'appliquer l'OCR est prise plus tard dans pipeline.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
import statistics
import fitz


# ======================================================================
# Dataclasses
# ======================================================================

@dataclass
class TextBlock:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_number: int


@dataclass
class ExtractedPage:
    page_number: int

    text: str

    blocks: list = field(default_factory=list)

    extraction_method: str = "pdf"

    # Nombre de caractères extraits nativement
    char_count: int = 0

    # Calculé plus tard dans pipeline.py
    needs_ocr: bool = False

    # Ajouté plus tard par language_detector.py
    language: str = "unknown"
    
    has_text: bool = False  # À calculer après extraction

    


@dataclass
class ExtractedDocument:
    source_path: str

    full_text: str

    pages: list = field(default_factory=list)

    blocks: list = field(default_factory=list)

    n_pages: int = 0


# ======================================================================
# Extraction
# ======================================================================

def _is_rtl_text(text: str) -> bool:
    """
    Détecte si un texte est majoritairement en écriture arabe (RTL), en
    comptant les caractères dans les plages Unicode arabes (lettres de
    base + présentation). Un simple ratio > 30% suffit car même un texte
    arabe contient des chiffres/ponctuation latins mélangés.
    """
    if not text:
        return False
    arabic_ranges = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                      (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
    arabic_count = sum(
        1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in arabic_ranges)
    )
    return arabic_count > len(text) * 0.3


def _group_into_columns(blocks, min_ratio=3.0, min_gap_floor=6.0):
    """
    Détecte une éventuelle séparation en colonnes par un algorithme
    adaptatif basé sur la médiane des écarts horizontaux.

    Principe :
      1. Trie les blocs par x0 (projection sur l'axe X).
      2. Calcule tous les écarts x0(n) - x1(n-1) entre blocs consécutifs.
      3. Identifie l'écart MAXIMAL (le plus grand gap).
      4. Calcule la MÉDIANE des AUTRES écarts (valeur typique).
      5. Si le plus grand gap est significativement plus grand que les autres
         (ratio >= min_ratio) ET dépasse un seuil plancher (min_gap_floor),
         on considère qu'il s'agit d'une séparation entre colonnes.
      6. Dans ce cas, on assigne chaque bloc à la colonne de gauche ou de
         droite selon la position de son centre (x0+x1)/2 par rapport au
         milieu du gap.
      7. Les blocs qui CHEVAUCHENT la frontière entre les deux colonnes
         (titres de section centrés comme « TEXTES GENERAUX » qui
         traversent l'espace inter-colonnes) sont isolés dans ``spanning`` :
         ils n'appartiennent à aucune colonne et doivent être émis à leur
         position verticale, comme un bloc pleine largeur.

    Cette approche est plus robuste que les seuils absolus (pourcentage
    de l'emprise ou médiane des largeurs) car elle compare les écarts
    entre eux au sein d'une même zone, ce qui s'adapte naturellement
    à des mises en page très différentes (sommaires denses ~100 pt,
    corps d'arrêté ~200 pt).

    Returns:
        (columns, spanning) : liste de listes de blocs (une par colonne
        détectée) et liste des blocs chevauchant la frontière.  Si aucune
        séparation nette n'est trouvée, retourne une seule liste
        contenant tous les blocs (spanning vide).
    """
    if not blocks:
        return [], []

    if len(blocks) < 2:
        return [list(blocks)], []

    sorted_by_x = sorted(blocks, key=lambda b: b[0])

    # Calculer tous les écarts entre blocs consécutifs
    gaps = []
    for a, b in zip(sorted_by_x, sorted_by_x[1:]):
        gap = b[0] - a[2]
        gaps.append(gap)

    # Identifier le plus grand écart
    max_gap = max(gaps)
    max_idx = gaps.index(max_gap)

    if len(gaps) > 1:
        other_gaps = gaps[:max_idx] + gaps[max_idx + 1:]
        typical_gap = statistics.median(other_gaps)
    else:
        typical_gap = 0.0

    # Décider si le plus grand écart est une séparation de colonne
    if max_gap < max(min_gap_floor, typical_gap * min_ratio):
        return [sorted(blocks, key=lambda b: b[1])], []

    # Position de la frontière (milieu du gap)
    split_x = (sorted_by_x[max_idx][2] + sorted_by_x[max_idx + 1][0]) / 2

    left_col = []
    right_col = []
    for b in blocks:
        x_center = (b[0] + b[2]) / 2
        if x_center < split_x:
            left_col.append(b)
        else:
            right_col.append(b)

    # Blocs chevauchant la frontière entre colonnes : leur emprise X
    # s'étend de part et d'autre du milieu de l'espace inter-colonnes.
    # Ex. page 3 du BO_7522, « TEXTES GENERAUX » (x 236-359) centré au
    #-dessus du dahir n° 1-26-08 : sans cette étape, il était rattaché à
    # la colonne de droite et se retrouvait émis après toute la colonne
    # gauche, en plein milieu de la loi organique n° 36-24.
    spanning = []
    if left_col and right_col:
        gap_left = max(b[2] for b in left_col)
        gap_right = min((b[0] for b in right_col if b[0] >= gap_left), default=gap_left)
        mid_gap = (gap_left + gap_right) / 2
        spanning = [b for b in blocks if b[0] < mid_gap < b[2]]
        if spanning:
            span_ids = {id(b) for b in spanning}
            left_col = [b for b in left_col if id(b) not in span_ids]
            right_col = [b for b in right_col if id(b) not in span_ids]

    # Ne garder que les colonnes non vides
    columns = [col for col in (left_col, right_col) if col]
    for col in columns:
        col.sort(key=lambda b: b[1])

    return columns, spanning


FULL_WIDTH_RATIO = 0.6
BAND_Y_GAP_THRESHOLD = 15.0


def _split_full_width_blocks(blocks, page_width):
    full_width_blocks, column_blocks = [], []
    threshold = page_width * FULL_WIDTH_RATIO
    for b in blocks:
        width = b[2] - b[0]
        if width > threshold:
            full_width_blocks.append(b)
        else:
            column_blocks.append(b)
    return full_width_blocks, column_blocks


def _group_into_bands(blocks):
    if not blocks:
        return []
    sorted_by_y = sorted(blocks, key=lambda b: b[1])
    bands = []
    current_band = [sorted_by_y[0]]

    for b in sorted_by_y[1:]:
        gap = b[1] - current_band[-1][3]
        if gap > BAND_Y_GAP_THRESHOLD:
            bands.append(current_band)
            current_band = [b]
        else:
            current_band.append(b)

    if current_band:
        bands.append(current_band)

    return bands


def _order_blocks_for_reading(raw_blocks):
    if not raw_blocks:
        return []

    full_text = " ".join(b[4] for b in raw_blocks)
    page_is_rtl = _is_rtl_text(full_text)

    page_width = max(b[2] for b in raw_blocks) - min(b[0] for b in raw_blocks)
    full_width_blocks, column_candidates = _split_full_width_blocks(raw_blocks, page_width)

    if not column_candidates:
        return sorted(full_width_blocks, key=lambda b: (b[1], b[0]))

    columns, spanning = _group_into_columns(column_candidates)

    if len(columns) <= 1:
        # Une seule colonne : ordre de lecture vertical simple.
        all_blocks = full_width_blocks + spanning + (columns[0] if columns else [])
        return sorted(all_blocks, key=lambda b: (b[1], b[0]))

    # Deux colonnes (mise en page type BO) : l'ordre de lecture est
    # COLONNE PAR COLONNE (toute la colonne de gauche, puis celle de
    # droite) et non ligne par ligne.  L'ancien découpage en bandes
    # horizontales intercalait les deux colonnes rangée par rangée, ce
    # qui mélangeait des textes distincts côte à côte — ex. BO_7522,
    # page 15 : la fin de l'arrêté antidumping (annexes 1 et 2) en
    # colonne gauche se retrouvait entrelacée avec l'arrêté
    # "laboratoires habilités" de la colonne droite, contaminant les
    # articles des deux instruments.
    if page_is_rtl:
        columns = sorted(columns, key=lambda col: -max(b[0] for b in col))
    else:
        columns = sorted(columns, key=lambda col: min(b[0] for b in col))

    ordered = []
    fw_sorted = sorted(full_width_blocks + spanning, key=lambda b: b[1])

    def _emit_pending_full_width(before_y: float) -> None:
        while fw_sorted and fw_sorted[0][1] < before_y:
            ordered.append(fw_sorted.pop(0))

    for col in columns:
        col_blocks = sorted(col, key=lambda b: b[1])
        if col_blocks:
            _emit_pending_full_width(col_blocks[0][1])
        for b in col_blocks:
            _emit_pending_full_width(b[1])
            ordered.append(b)
    ordered.extend(fw_sorted)

    return ordered


_BIDI_MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}

# Plages Unicode arabes (arabe, arabe étendu-A/B, formes de présentation)
_RTL_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def _is_rtl_char(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in _RTL_RANGES)


def _is_ltr_char(ch: str) -> bool:
    # Lettre hors plages arabes (latin, etc.) → caractère fort LTR.
    return ch.isalpha() and not _is_rtl_char(ch)


def _fix_bidi_line(chars: list) -> str:
    """
    Reconstruit une ligne à partir de ses caractères (voir page.get_text
    ("rawdict")), en corrigeant un bug distinct de celui traité par
    _order_blocks_for_reading() : même une fois les blocs/colonnes dans le
    bon ordre, `page.get_text("blocks")` assemble parfois le TEXTE d'une
    ligne RTL dans l'ordre visuel (gauche → droite) au lieu de l'ordre de
    lecture logique — ex. une ligne de référence de décret ressort en
    "من23 صادر في943.26 والتجارة رقمةقرار لوزير الصناع" au lieu de
    "قرار لوزير الصناعة والتجارة رقم 943.26 صادر في 23 من" (confirmé sur
    data/raw/ar/BO_7515_Ar.pdf, page 9). Les positions des caractères
    individuels (via "rawdict") restent fiables même quand le texte déjà
    assemblé par "blocks" ne l'est pas plus.

    Principe (mini algorithme BiDi, cf. Unicode UAX #9) : on trie les
    caractères par position physique gauche→droite, on découpe en
    tronçons homogènes (arabe vs "autre" : chiffres/latin/ponctuation),
    on inverse l'ORDRE des tronçons (paragraphe RTL) et le contenu de
    CHAQUE tronçon (arabes comme "autre" : dans un paragraphe RTL, tout
    est disposé physiquement de droite à gauche, donc même les chiffres
    apparaissent dans l'ordre inverse de leur lecture logique — "943.26"
    est placé physiquement comme "6,2,.,3,4,9" de gauche à droite), et on
    permute les parenthèses/crochets (miroir BiDi standard : "(" et ")"
    échangent de rôle quand le sens de lecture s'inverse).

    Limite connue : les caractères neutres (espaces, parenthèses) ne sont
    pas toujours rattachés au bon tronçon voisin dans les dates
    hégiriennes suivies d'une date grégorienne entre parenthèses — la
    résolution complète des caractères neutres nécessiterait
    l'algorithme BiDi complet (règles N1/N2 d'UAX #9). Avec ce correctif,
    les nombres (décrets, dates, prix) sont correctement rétablis dans
    l'immense majorité des cas.
    """
    chars = sorted(chars, key=lambda c: c["bbox"][0])

    # Direction du paragraphe (UAX #9, règle P2) : le PREMIER caractère
    # fort (lettre) de la ligne décide du sens.  Une ligne française qui
    # contient un caractère arabe isolé (chiffre arabo-indien, glyphe
    # parasite, date hégirienne) reste LTR — l'ancien comportement
    # inversait alors TOUTE la ligne dès qu'un caractère RTL était
    # présent (ex. « relative aux émissions « 09 » « et » » ressortait
    # « » « te » 09 « snoissimé xua evitaler »), même si la ligne était
    # du français pur.  La correction BiDi n'est appliquée que si le
    # premier caractère fort est RTL (ligne arabe, éventuellement
    # ponctuée de nombres latins).
    paragraph_is_rtl = None
    for c in chars:
        ch = c["c"]
        if _is_rtl_char(ch):
            paragraph_is_rtl = True
            break
        if _is_ltr_char(ch):
            paragraph_is_rtl = False
            break
    if paragraph_is_rtl is False:
        return "".join(c["c"] for c in chars)  # ligne LTR : rien à corriger

    runs, cur_type, cur = [], None, []
    for c in chars:
        ch = c["c"]
        t = "rtl" if _is_rtl_char(ch) else "other"
        if t != cur_type and cur:
            runs.append((cur_type, cur))
            cur = []
        cur_type = t
        cur.append(ch)
    if cur:
        runs.append((cur_type, cur))

    if not any(t == "rtl" for t, _ in runs):
        return "".join(c["c"] for c in chars)  # ligne LTR : rien à corriger

    runs.reverse()
    out = []
    for t, chs in runs:
        rev = list(reversed(chs))
        if t == "rtl":
            out.append("".join(rev))
        else:
            out.append("".join(_BIDI_MIRROR.get(ch, ch) for ch in rev))
    return "".join(out)


def _extract_blocks_via_rawdict(page) -> list:
    """
    Reconstruit les blocs d'une page ligne par ligne à partir de
    page.get_text("rawdict"), en appliquant _fix_bidi_line() à chaque
    ligne. Renvoie une liste de tuples (x0, y0, x1, y1, text) au même
    format que page.get_text("blocks"), pour rester compatible avec
    _order_blocks_for_reading(raw_blocks).
    """
    raw = page.get_text("rawdict")
    blocks = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # ignore les blocs image
            continue
        lines_text = []
        bx0, by0, bx1, by1 = block["bbox"]
        for line in block.get("lines", []):
            chars = [c for span in line["spans"] for c in span.get("chars", [])]
            if not chars:
                continue
            lines_text.append(_fix_bidi_line(chars))
        if lines_text:
            blocks.append((bx0, by0, bx1, by1, "\n".join(lines_text)))
    return blocks


# Fraction du haut de page à exclure (bandeau d'en-tête répété : "BULLETIN
# OFFICIEL N° XXXX — date", numéro de page, etc.) — 8% de la hauteur.
# Calibré sur les BO marocains réels (confirmé sur BO_6788_Fr).
HEADER_BAND_FRACTION = 0.08
# Fraction du bas de page à exclure (pied de page : numéro de page, suite
# de section, etc.) — 6% de la hauteur.
FOOTER_BAND_FRACTION = 0.06


def extract_text_from_pdf(pdf_path: str) -> ExtractedDocument:

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

    with fitz.open(pdf_path) as doc:
        pages = []
        all_blocks = []
        full_text_parts = []

        for page_index, page in enumerate(doc):

            page_height = page.rect.height
            header_cutoff = page_height * HEADER_BAND_FRACTION
            footer_cutoff = page_height * (1.0 - FOOTER_BAND_FRACTION)

            raw_blocks = _extract_blocks_via_rawdict(page)

            raw_blocks = _order_blocks_for_reading(raw_blocks)

            page_blocks = []
            page_text_parts = []

            for block in raw_blocks:

                x0, y0, x1, y1, text, *_ = block

                # Exclure les blocs dans la bande d'en-tête (haut de page)
                if y1 < header_cutoff:
                    continue
                # Exclure les blocs dans la bande de pied de page (bas de page)
                if y0 > footer_cutoff:
                    continue

                text = text.strip()

                if not text:
                    continue

                tb = TextBlock(
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page_number=page_index + 1
                )

                page_blocks.append(tb)
                all_blocks.append(tb)

                page_text_parts.append(text)

            page_text = "\n".join(page_text_parts)
            
            has_text = len(page_text.strip()) > 50  # Seuil empirique

            

            pages.append(
                ExtractedPage(
                    page_number=page_index + 1,
                    text=page_text,
                    blocks=page_blocks,
                    extraction_method="pdf",
                    char_count=len(page_text.strip()),
                    has_text=has_text
                )
            )

            full_text_parts.append(page_text)

    return ExtractedDocument(
        source_path=str(path),
        full_text="\n".join(full_text_parts),
        pages=pages,
        blocks=all_blocks,
        n_pages=len(pages),
    )


# ======================================================================
# Debug
# ======================================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage:")
        print("python pdf_extractor.py <pdf>")
        sys.exit(1)

    document = extract_text_from_pdf(sys.argv[1])

    print("=" * 70)

    print(f"Pages : {document.n_pages}")

    print("=" * 70)

    for page in document.pages:

        print(
            f"Page {page.page_number:3d} | "
            f"chars={page.char_count:5d}"
        )