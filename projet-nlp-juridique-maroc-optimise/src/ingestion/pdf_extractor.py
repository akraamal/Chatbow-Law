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


# Version de la logique d'extraction/ordonnancement.  À incrémenter à chaque
# modification qui peut changer le texte produit (ordre des blocs, colonnes,
# BiDi, bandes) : les fichiers data/interim/*.txt antérieurs sont alors
# détectés comme stalés via leur sidecar .meta.json
# (pipeline.stamp_interim_provenance) et la régénération des JSON est
# refusée tant que l'ingestion n'a pas été relancée.
EXTRACTOR_VERSION = "5"


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


# Nombre minimal de blocs ENTIERS requis de part et d'autre d'une gouttière
# candidate pour la retenir comme séparation de colonnes (cf.
# _is_valid_column_gap).
MIN_BLOCKS_PER_COLUMN = 2


def _is_valid_column_gap(sorted_by_x, idx, min_blocks=MIN_BLOCKS_PER_COLUMN):
    """Un écart entre deux blocs consécutifs (en x) n'est une gouttière de
    colonnes que si des blocs ENTIERS se trouvent de part et d'autre de la
    frontière : aucun bloc ne doit chevaucher la séparation (les titres
    centrés qui traversent l'inter-colonnes sont retirés en ``spanning``
    après le split).  Un simple max-gap se trompe sinon : ex. un bloc
    centré « ARRÊTE : » (x0 420,8) précédé en x d'une fin de ligne courte
    (x1 387,3) produit un écart supérieur à la vraie gouttière."""
    split_x = (sorted_by_x[idx][2] + sorted_by_x[idx + 1][0]) / 2
    left = sum(1 for b in sorted_by_x if b[2] <= split_x)
    right = sum(1 for b in sorted_by_x if b[0] >= split_x)
    return left >= min_blocks and right >= min_blocks


def _find_empty_band(blocks, min_blocks=MIN_BLOCKS_PER_COLUMN):
    """Cherche la bande verticale vide la plus large entre les blocs, avec
    au moins ``min_blocks`` blocs ENTIERS de part et d'autre.

    On balaie la COUVERTURE de l'axe X (union des intervalles x des blocs),
    pas les écarts entre blocs consécutifs : un bloc centré DANS une
    colonne (ex. « ARRÊTE : » de la page 98 du BO_7492, x 134-174) interrompt
    la séquence des blocs triés par x0 et masque la vraie gouttière
    (289→309) — aucun couple consécutif ne la traverse, alors que le
    balayage de couverture la voit comme une bande sans aucun bloc.

    Returns:
        (start, end) du milieu de la bande la plus large, ou None si aucune
        bande valide (blocs entiers des deux côtés).
    """
    if not blocks:
        return None

    events = []
    for b in blocks:
        events.append((b[0], +1))
        events.append((b[2], -1))
    events.sort()

    coverage = 0
    bands = []
    x = events[0][0]
    for pos, delta in events:
        if pos > x and coverage == 0:
            bands.append((x, pos))
        coverage += delta
        x = max(x, pos)

    best = None
    for start, end in bands:
        left = sum(1 for b in blocks if b[2] <= start)
        right = sum(1 for b in blocks if b[0] >= end)
        if left < min_blocks or right < min_blocks:
            continue
        if best is None or (end - start) > (best[1] - best[0]):
            best = (start, end)
    return best


def _group_into_columns(blocks, min_ratio=3.0, min_gap_floor=6.0):
    """
    Détecte une éventuelle séparation en colonnes par un algorithme
    adaptatif basé sur la médiane des écarts horizontaux.

    Principe :
      1. Trie les blocs par x0 (projection sur l'axe X).
      2. Calcule tous les écarts x0(n) - x1(n-1) entre blocs consécutifs.
      3. Parmi les écarts qui dépassent le seuil adaptatif
         (max(min_gap_floor, médiane des écarts * min_ratio)), retient
         les gouttières PLAUSIBLES : un écart ne sépare deux colonnes que
         si des blocs ENTIERS se trouvent de part et d'autre (aucun bloc
         ne chevauche la séparation — les titres centrés sont traités en
         spanning plus tard).
      4. Choisit la gouttière au plus grand écart valide.
      5. Dans ce cas, on assigne chaque bloc à la colonne de gauche ou de
         droite selon la position de son centre (x0+x1)/2 par rapport au
         milieu de la gouttière.
      6. Les blocs qui CHEVAUCHENT la frontière entre les deux colonnes
         (titres de section centrés comme « TEXTES GENERAUX » qui
         traversent l'espace inter-colonnes) sont isolés dans ``spanning`` :
         ils n'appartiennent à aucune colonne et doivent être émis à leur
         position verticale, comme un bloc pleine largeur.

    Le filtre « blocs entiers de part et d'autre » (étape 3) corrige un
    faux positif du simple max-gap : BO_7492 page 103, le bloc centré
    « ARRÊTE : » (x0 420,8) précédé en x de la fin courte « de fgiuier. »
    (x1 387,3) créait un écart de 33,5 pt supérieur à la vraie gouttière
    (21,4 pt) — la page était alors lue en bandes horizontales et
    « ARRÊTE : » émis seul en fin de page (fausse frontière d'instrument
    dans l'arrêté n° 197-26).

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

    typical_gap = statistics.median(gaps) if gaps else 0.0
    threshold = max(min_gap_floor, typical_gap * min_ratio)

    # 1) Gouttière par balayage de couverture : robuste aux blocs centrés
    # dans une colonne (ex. « ARRÊTE : » de la page 98 du BO_7492) qui
    # masquent la gouttière dans la projection des écarts adjacents.
    split_x = None
    split_from_band = False
    band = _find_empty_band(blocks)
    if band is not None and (band[1] - band[0]) >= threshold:
        split_x = (band[0] + band[1]) / 2
        split_from_band = True
    else:
        # 1bis) La gouttière peut être masquée par des blocs de FOND de page
        # qui la traversent — signatures et séparateurs « * * ».  Ex.
        # BO_7510 page 9 : « ZAKIA DRIOUICH » (x 294-548, y 723) et
        # « * * » (x 281-313, y 765) couvrent l'espace inter-colonnes
        # (289→309) en bas de page, si bien que le balayage ne trouve
        # aucune bande vide.  On relance le balayage sur la zone haute de
        # la page (hors marge de pied) : la gouttière réelle réapparaît
        # alors, non contaminée par les signatures.  Ce repli n'est retenu
        # que si des blocs de pied traversent réellement la bande trouvée
        # (sinon, une bande haute d'une page tabulaire pourrait être
        # confondue avec une gouttière de colonnes, ex. BO_7500 page 5).
        bottom_limit = max(b[3] for b in blocks) - FOOTER_BLOCK_MARGIN
        body_blocks = [b for b in blocks if b[1] < bottom_limit]
        footer_blocks = [b for b in blocks if b[1] >= bottom_limit]
        band = _find_empty_band(body_blocks)
        if (
            band is not None
            and (band[1] - band[0]) >= threshold
            and any(
                fb[0] < (band[0] + band[1]) / 2 < fb[2] and _is_decorative_straddler(fb)
                for fb in footer_blocks
            )
        ):
            split_x = (band[0] + band[1]) / 2
            split_from_band = True

    if split_x is None:
        # 2) Repli : écarts entre blocs consécutifs (nécessaire quand un
        # titre centré traverse l'inter-colonnes, ex. « TEXTES GENERAUX »
        # du BO_7522 : la bande n'est alors pas vide et le balayage ne
        # trouve rien — le titre est isolé en ``spanning`` après le split).
        candidates = [
            (gap, idx)
            for idx, gap in enumerate(gaps)
            if gap >= threshold and _is_valid_column_gap(sorted_by_x, idx)
        ]
        if candidates:
            _, max_idx = max(candidates)
            split_x = (sorted_by_x[max_idx][2] + sorted_by_x[max_idx + 1][0]) / 2

    if split_x is None:
        return [sorted(blocks, key=lambda b: b[1])], []

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
    # gauche, en plein milieu de la loi organique n° 36-24.  C'est aussi le
    # sort des signatures et séparateurs de pied de page qui traversent la
    # gouttière (BO_7510 page 9 : « ZAKIA DRIOUICH » x 294-548 et « * * »
    # x 281-313) : ils sont émis à leur position verticale, en fin de page,
    # et non dans une colonne.
    if split_from_band:
        # Gouttière fiable (bande vide) : la frontière est le centre exact de
        # la masse inter-colonnes, tout bloc qui la traverse est spannting —
        # y compris les signatures et séparateurs de pied de page (BO_7510
        # page 9 : « ZAKIA DRIOUICH » x 294-548 et « * * » x 281-313).
        spanning = [b for b in blocks if b[0] < split_x < b[2]]
    else:
        # Repli : la frontière entre les deux colonnes peut ne pas coïncider
        # avec le centre du split par écart de blocs (ex. BO_7500 page 3 :
        # « DÉCRÈTE : » x 132-177 | « TEXTES GENERAUX » x 236-359, split à
        # x 206 mais gouttière réelle à ~300) — on repère alors les
        # traversées par la gouttière réelle, déduite des emprises des
        # colonnes nettes.
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
# Marge de pied de page : les blocs sous cette ligne (signatures, séparateurs
# « * * », numéros de page) peuvent traverser la gouttière et masquer les
# colonnes au balayage de couverture (ex. BO_7510 page 9, « ZAKIA DRIOUICH »
# x 294-548).  Le balayage est alors relancé sur la zone haute uniquement.
FOOTER_BLOCK_MARGIN = 120.0


def _is_decorative_straddler(block):
    """
    Un bloc de pied de page qui chevauche la gouttière ne doit déclencher le
    repli « bande sur le corps » que s'il s'agit d'une DÉCORATION de fin de
    page — séparateur d'astérisques (« * * ») ou signature attestant
    l'arrêté (« ZAKIA DRIOUICH. ») — et non d'une cellule de tableau qui
    traverse par hasard la bande trouvée par le balayage (ex. les en-têtes
    tabulaires « عونلا »/« يملعلا مسلاا » de la page des variétés protégées
    du BO_7500, qui couvrent 159→181 sans que la page soit réellement
    bicolonne).

    Détection sans expression régulière (module backend léger) :
      - la présence d'une astérisque (séparateurs « * * », signatures
        suivies du séparateur « ZAKIA DRIOUICH.\n* ») ;
      - sinon une signature : texte court, en MAJUSCULES, se terminant
        par « . » (ex. « FOUZI LEKJAA. »).
    """
    text = block[4].strip()
    if not text:
        return False
    if "*" in text:
        return True
    upper = "".join(ch for ch in text if not ch.isspace())
    if not upper.endswith("."):
        return False
    if len(upper) > 40:
        return False
    return all(ch.isupper() or ch in ".-'/," for ch in upper)


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

    # Page dominée par des blocs pleine largeur (paragraphes pleins, titre +
    # visas) : les « candidats colonnes » ne sont alors que des fragments de
    # lignes (ex. le « ARRÊTE : » centré de la page 83 du BO_7492, x 273-322,
    # qui crée un faux écart de 78 pt accepté comme gouttière) — la lecture
    # est un simple tri vertical.  Sur les vraies pages bicolonnes, le
    # pleine largeur se réduit au bandeau d'en-tête (~60 caractères) et les
    # colonnes portent le reste du texte.
    fw_chars = sum(len(b[4]) for b in full_width_blocks)
    cc_chars = sum(len(b[4]) for b in column_candidates)
    if fw_chars >= cc_chars:
        return sorted(raw_blocks, key=lambda b: (b[1], b[0]))

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


# Marge verticale au-dessus du bloc « SOMMAIRE » pour rattacher les blocs
# du sommaire (ex. « Pages ») à la zone sommaire et non au bandeau haut.
_SOMMAIRE_MARGIN_Y = 12.0


def _order_sommaire_page(raw_blocks):
    """
    Page de garde des éditions FR (couverture + sommaire) : le bandeau haut
    (tarifs d'abonnement, table de prix sur 4 colonnes) et le sommaire
    (2 colonnes en bas) ont des structures de colonnes DIFFÉRENTES — aucun
    split global ne peut les traiter ensemble, car les blocs des deux zones
    s'intercalent en x (ex. BO_7522 page 1 : « A destination de l'étranger »
    du tableau d'abonnement chevauche la colonne droite du sommaire, et
    l'écart maximal est celui du tableau, pas celui du sommaire → les
    entrées « Arrêté conjoint... » étaient étiquetées "spanning" et se
    retrouvaient intercalées ligne par ligne avec la colonne gauche :
    "Arrêté conjoint du ministre de l'industrie et du Cour
    constitutionnelle. commerce et de la ministre...").

    On découpe donc la page en deux zones :
      - zone sommaire (à partir du bloc « SOMMAIRE ») : ordonnée
        colonne-par-colonne (_order_blocks_for_reading, fiable ici car la
        zone est proprement bicolonne) ;
      - bandeau haut (abonnements) : simple tri par y.

    Retourne None si la page ne contient pas de sommaire détectable.
    """
    sommaire_y = None
    for b in raw_blocks:
        t = b[4].strip()
        if t == "SOMMAIRE" or (t.startswith("SOMMAIRE") and len(t) < 20):
            sommaire_y = b[1]
            break
    if sommaire_y is None:
        return None

    toc = [b for b in raw_blocks if b[1] >= sommaire_y - _SOMMAIRE_MARGIN_Y]
    toc_ids = {id(b) for b in toc}
    top = [b for b in raw_blocks if id(b) not in toc_ids]

    ordered_toc = _order_blocks_for_reading(toc)
    return sorted(top, key=lambda b: (b[1], b[0])) + ordered_toc


_BIDI_MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}

# Plages Unicode arabes (arabe, arabe étendu-A/B, formes de présentation)
_RTL_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def _is_rtl_char(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in _RTL_RANGES)


def _is_ltr_char(ch: str) -> bool:
    # Lettre hors plages arabes (latin, etc.) → caractère fort LTR.
    return ch.isalpha() and not _is_rtl_char(ch)


def _rtl_segment_to_logical(visual_chars: list, mirror_parens: bool = True) -> str:
    """Rétablit l'ordre logique d'un tronçon arabe disposé en ordre VISUEL.

    Le rendu RTL affiche les caractères de droite à gauche : on inverse la
    séquence puis on rétablit les nombres (forts LTR, affichés à l'endroit
    même dans un contexte RTL) et on permute les parenthèses/crochets
    (miroir BiDi, désactivable via *mirror_parens* : dans une ligne RTL
    entière, les parenthèses physiques sont déjà les caractères logiques,
    simplement déplacés — l'inversion suffit, le miroir les inverserait).

    La restauration des nombres porte sur des TOKENS complets
    (``r"[0-9][0-9.,]*"``, ex. "943.26", "1.080.246,00") et non sur des
    groupes de chiffres contigus : "943.26" inversé donne "62.349" dont
    une restauration groupe-par-groupe produirait "26.943" à tort.
    """
    rev = [_BIDI_MIRROR.get(ch, ch) for ch in reversed(visual_chars)] \
        if mirror_parens else list(reversed(visual_chars))
    out = []
    i = 0
    n = len(rev)
    while i < n:
        if rev[i].isdigit():
            j = i
            while j < n and (rev[j].isdigit() or rev[j] in ".,"):
                j += 1
            out.extend(reversed(rev[i:j]))
            i = j
        else:
            out.append(rev[i])
            i += 1
    return "".join(out)


def _fix_ltr_line_with_rtl_segments(chars: list) -> str:
    """Ligne LTR (français) contenant des tronçons arabes cités.

    Ex. « relative aux émissions «الحقيقة في 90 دقيقة» et «أسد إفريقيا»
    diffusées par ... » (décision du CSN, BO_7522) : le français est déjà
    dans l'ordre de lecture, mais chaque tronçon arabe est disposé en
    ordre VISUEL (inversé) — «ةقيقد 90 يف ةقيقحلا».  On isole chaque
    segment RTL (run arabe + neutres/chiffres intercalés) et on rétablit
    son ordre logique sans toucher au reste de la ligne.
    """
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
        return "".join(c["c"] for c in chars)  # ligne purement LTR

    out = []
    i = 0
    n = len(runs)
    while i < n:
        t, chs = runs[i]
        if t != "rtl":
            out.append("".join(chs))
            i += 1
            continue
        # Segment RTL : runs arabes + éventuels runs neutres INTERCALÉS
        # (espaces, chiffres — jamais de lettres latines, qui signaleraient
        # du français entre deux citations, ex. "» et «").
        j = i
        while j + 1 < n:
            nt, nchs = runs[j + 1]
            if nt == "rtl":
                j += 1
            elif (j + 2 < n and runs[j + 2][0] == "rtl"
                  and not any(c.isalpha() and not _is_rtl_char(c) for c in nchs)):
                j += 1
            else:
                break
        visual = [c for _, chs2 in runs[i:j + 1] for c in chs2]
        out.append(_rtl_segment_to_logical(visual))
        i = j + 1
    return "".join(out)


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
    on inverse l'ORDRE des tronçons (paragraphe RTL) et on inverse le
    contenu des tronçons ARABES.  Les tronçons «autre» sont gérés selon
    leur contenu (cf. le code) :
      - latin (sigle, unité, ex. "NM 01.4.510") : laissés tels quels ;
      - chiffres/ponctuation purs : inversés, puis leurs tokens numériques
        rétablis dans l'ordre de lecture ("943.26" est disposé
        physiquement en ordre LTR ; l'inversion le retournerait sinon).

    Limite connue : les caractères neutres (espaces, parenthèses) ne sont
    pas toujours rattachés au bon tronçon voisin dans les dates
    hégiriennes suivies d'une date grégorienne entre parenthèses — la
    résolution complète des caractères neutres nécessiterait
    l'algorithme BiDi complet (règles N1/N2 d'UAX #9). Avec ce correctif,
    les nombres (décrets, dates, prix) sont correctement rétablis dans
    l'immense majorité des cas.
    """
    chars = sorted(chars, key=lambda c: c["bbox"][0])

    # Direction du paragraphe : le PREMIER caractère fort (UAX #9, P2)
    # DOIT CONVERGER avec la majorité des lettres fortes.  Seule la règle
    # P2 se trompe sur les lignes françaises dont le saut de ligne place
    # une citation arabe EN TÊTE (ex. «الحقيقة في 90 دقيقة» et de
    # l'édition du 18 janvier 2026...) : le premier caractère fort y est
    # arabe, mais la ligne est majoritairement française — l'ancien code
    # inversait alors toute la ligne (sortie : "ed 6202 reivnaj 81 ud
    # noitidé'l ed te »الحقيقة...«").  Inversement, une majorité seule se
    # trompe sur les lignes ~50/50 (ex. «relative aux émissions
    # «الحقيقة...» et «أسد إفريقيا»», 24 lettres arabes vs 22 latines).
    # Les deux signaux doivent donc être RTL pour appliquer l'inversion
    # RTL complète ; sinon on traite la ligne comme LTR avec tronçons RTL.
    first_strong_rtl = None
    for c in chars:
        ch = c["c"]
        if _is_rtl_char(ch):
            first_strong_rtl = True
            break
        if _is_ltr_char(ch):
            first_strong_rtl = False
            break
    rtl_letters = sum(1 for c in chars if _is_rtl_char(c["c"]))
    ltr_letters = sum(1 for c in chars if _is_ltr_char(c["c"]))
    paragraph_is_rtl = bool(first_strong_rtl) and rtl_letters >= ltr_letters

    if not paragraph_is_rtl:
        if ltr_letters == 0:
            return "".join(c["c"] for c in chars)  # pas de lettres du tout
        # Ligne LTR (français) : les éventuels tronçons arabes cités
        # («الحقيقة في 90 دقيقة») sont inversés UNIQUEMENT, pas toute la
        # ligne — l'ancien comportement laissait les tronçons arabes dans
        # leur ordre visuel (supprimés ensuite par le nettoyage) ou
        # inversait toute la ligne quand un caractère RTL était présent.
        return _fix_ltr_line_with_rtl_segments(chars)

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
        if t == "rtl":
            out.append("".join(reversed(chs)))
        elif any(_is_ltr_char(c) for c in chs):
            # Run «autre» contenant des lettres latines (sigle, unité —
            # ex. "NM 01.4.510") : segment LTR déjà dans l'ordre logique ;
            # l'inverser casserait le sigle et son nombre.
            out.append("".join(chs))
        else:
            # Run «autre» pur (chiffres/ponctuation/espaces) : l'inversion
            # + restauration des tokens numériques, SANS miroir des
            # parenthèses (elles sont déjà des caractères logiques).
            out.append(_rtl_segment_to_logical(chs, mirror_parens=False))
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

            sommaire_ordered = _order_sommaire_page(raw_blocks)
            if sommaire_ordered is not None:
                raw_blocks = sommaire_ordered
            else:
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