"""
layout_splitter.py
--------------------
Le Bulletin Officiel marocain publie souvent le texte français et sa
traduction arabe sur la même page, en deux colonnes (généralement : arabe à
droite, français à gauche). Si on extrait le texte sans tenir compte de la
position des blocs, les deux langues se retrouvent mélangées ligne par ligne.

Ce module utilise les coordonnées des blocs (fournies par pdf_extractor.py)
pour séparer les deux colonnes avant toute autre étape de traitement.
"""

import re
from dataclasses import dataclass
from src.ingestion.pdf_extractor import ExtractedDocument, TextBlock

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


@dataclass
class BilingualSplit:
    text_left: str   # généralement français
    text_right: str  # généralement arabe
    source_path: str


def _get_page_width_midpoint(blocks: list) -> float:
    """Calcule le point médian horizontal de la page à partir des blocs."""
    if not blocks:
        return 0.0
    min_x = min(b.x0 for b in blocks)
    max_x = max(b.x1 for b in blocks)
    return (min_x + max_x) / 2


def _arabic_ratio(text: str) -> float:
    """Proportion de caractères arabes dans le texte."""
    if not text.strip():
        return 0.0
    total = sum(1 for c in text if c.isalpha())
    if total == 0:
        return 0.0
    arabic = len(_ARABIC_RE.findall(text))
    return arabic / total


def split_bilingual_columns(document: ExtractedDocument) -> BilingualSplit:
    """
    Sépare les blocs de texte en deux colonnes (gauche/droite) selon leur
    position horizontale par rapport au milieu de la page.

    Hypothèse : mise en page à 2 colonnes, une par langue. Cette hypothèse
    doit être validée visuellement sur un échantillon de documents réels
    avant une utilisation à grande échelle (certains BO sont en pleine page
    par langue plutôt qu'en colonnes — dans ce cas, utiliser plutôt
    language_detector.py par page complète).

    Args:
        document: l'ExtractedDocument complet (pas une page isolée) — le
            découpage se fait sur l'ensemble des blocs du document.

    Returns:
        BilingualSplit avec le texte de chaque colonne, trié par page puis
        par position verticale pour respecter l'ordre de lecture.
    """
    if not document.blocks:
        return BilingualSplit(text_left="", text_right="", source_path=document.source_path)

    midpoint = _get_page_width_midpoint(document.blocks)

    left_blocks = []
    right_blocks = []

    for b in document.blocks:
        block_center = (b.x0 + b.x1) / 2
        if block_center < midpoint:
            left_blocks.append(b)
        else:
            right_blocks.append(b)

    # Tri par page puis par position verticale pour respecter l'ordre de lecture
    def sort_key(b: TextBlock):
        return (b.page_number, b.y0)

    left_blocks.sort(key=sort_key)
    right_blocks.sort(key=sort_key)

    text_left = "\n".join(b.text for b in left_blocks)
    text_right = "\n".join(b.text for b in right_blocks)

    return BilingualSplit(
        text_left=text_left,
        text_right=text_right,
        source_path=document.source_path,
    )


def detect_layout_type(doc: ExtractedDocument, threshold_ratio: float = 0.35) -> str:
    """
    Détecte heuristiquement si le document est probablement bilingue en
    colonnes ou en pleine page, en observant la dispersion horizontale des
    blocs ET la répartition linguistique.

    Résout le problème des BO arabes dont la page d'en-tête (mono-colonne,
    purement arabe) était classifiée "colonnes" parce que ses blocs se
    répartissent accidentellement des deux côtés de l'axe médian. La
    vérification linguistique supplémentaire détecte ce cas : si les deux
    "colonnes" sont majoritairement dans la même langue, ce n'est pas une
    vraie mise en page bilingue.

    Returns:
        "colonnes" si les blocs semblent répartis en deux groupes distincts
        ET que les deux groupes sont dans des langues différentes,
        "pleine_page" sinon.
    """
    if not doc.blocks:
        return "pleine_page"

    midpoint = _get_page_width_midpoint(doc.blocks)
    left_count = sum(1 for b in doc.blocks if (b.x0 + b.x1) / 2 < midpoint)
    right_count = len(doc.blocks) - left_count

    ratio = min(left_count, right_count) / max(left_count, right_count, 1)
    if ratio <= threshold_ratio:
        return "pleine_page"

    # Vérification linguistique : si les deux côtés sont dans la même langue,
    # ce n'est pas une vraie mise en page bilingue.
    left_text = " ".join(b.text for b in doc.blocks if (b.x0 + b.x1) / 2 < midpoint)
    right_text = " ".join(b.text for b in doc.blocks if (b.x0 + b.x1) / 2 >= midpoint)

    left_ar_ratio = _arabic_ratio(left_text)
    right_ar_ratio = _arabic_ratio(right_text)

    # Les deux côtés sont majoritairement arabes → mono-langue arabe
    if left_ar_ratio > 0.5 and right_ar_ratio > 0.5:
        return "pleine_page"
    # Les deux côtés sont majoritairement non-arabes → mono-langue français
    if left_ar_ratio < 0.5 and right_ar_ratio < 0.5:
        return "pleine_page"

    return "colonnes"


if __name__ == "__main__":
    import sys
    from src.ingestion.pdf_extractor import extract_text_from_pdf

    if len(sys.argv) < 2:
        print("Usage : python layout_splitter.py <chemin_vers_pdf>")
        sys.exit(1)

    extracted = extract_text_from_pdf(sys.argv[1])
    layout = detect_layout_type(extracted)
    print(f"Type de mise en page détecté : {layout}")

    if layout == "colonnes":
        split = split_bilingual_columns(extracted)
        print("--- Colonne gauche (probablement FR) ---")
        print(split.text_left[:500])
        print("\n--- Colonne droite (probablement AR) ---")
        print(split.text_right[:500])
