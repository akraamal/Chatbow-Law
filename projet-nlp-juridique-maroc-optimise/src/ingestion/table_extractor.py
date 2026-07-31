"""
table_extractor.py
--------------------
Extraction dédiée des tableaux dans les PDF juridiques (ex : listes de
variétés protégées, barèmes tarifaires, annexes structurées).

Problème résolu : pdf_extractor.py extrait le texte "bloc par bloc" trié par
position, ce qui fonctionne bien pour du texte narratif linéaire mais mélange
les cellules d'un tableau (surtout les tableaux bilingues FR/AR, où les
colonnes de langues différentes se retrouvent entrelacées de façon
incohérente).

Ce module utilise pdfplumber, qui a une détection de structure de tableau
(lignes/colonnes, avec ou sans bordures visibles) bien plus fiable que le
simple tri de blocs de PyMuPDF.

Installation :
    pip install pdfplumber
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

import pdfplumber


@dataclass
class ExtractedTable:
    page_number: int
    bbox: tuple            # (x0, top, x1, bottom) — coordonnées pdfplumber
    rows: list              # liste de listes (une liste de cellules par ligne)
    n_rows: int
    n_cols: int


@dataclass
class TableExtractionResult:
    source_path: str
    tables: list = field(default_factory=list)


def extract_tables_from_pdf(pdf_path: str) -> TableExtractionResult:
    """
    Parcourt toutes les pages du PDF et extrait les tableaux détectés, avec
    leur position (bbox) pour permettre ensuite d'exclure cette zone du texte
    narratif extrait par pdf_extractor.py.

    Returns:
        TableExtractionResult avec la liste des tableaux trouvés, chacun sous
        forme de liste de lignes (chaque ligne = liste de cellules texte).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

    tables_found = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            # find_tables() détecte les tableaux avec leur position exacte,
            # contrairement à extract_tables() qui ne retourne que le contenu
            found = page.find_tables()

            for table in found:
                rows = table.extract()
                if not rows:
                    continue

                # Nettoyage : remplacer les cellules None (fusion de cellules
                # ou cellule vide détectée) par une chaîne vide
                clean_rows = [
                    [cell.strip() if cell else "" for cell in row]
                    for row in rows
                ]

                n_rows = len(clean_rows)
                n_cols = max((len(r) for r in clean_rows), default=0)

                tables_found.append(
                    ExtractedTable(
                        page_number=page_index + 1,
                        bbox=table.bbox,  # (x0, top, x1, bottom)
                        rows=clean_rows,
                        n_rows=n_rows,
                        n_cols=n_cols,
                    )
                )

    return TableExtractionResult(source_path=str(pdf_path), tables=tables_found)


def tables_to_serializable(result: TableExtractionResult) -> list:
    """Convertit le résultat en structure JSON-sérialisable pour sauvegarde disque."""
    return [
        {
            "page_number": t.page_number,
            "bbox": list(t.bbox),
            "n_rows": t.n_rows,
            "n_cols": t.n_cols,
            "rows": t.rows,
        }
        for t in result.tables
    ]


def save_tables_as_json(result: TableExtractionResult, output_path: str) -> None:
    """Sauvegarde les tableaux extraits dans un fichier JSON structuré."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(tables_to_serializable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_table_bboxes_by_page(result: TableExtractionResult) -> dict:
    """
    Regroupe les bounding boxes des tableaux par numéro de page, sous forme
    de dict {page_number: [(x0, top, x1, bottom), ...]}.

    Utile pour filtrer les blocs de texte de pdf_extractor.py qui tombent
    dans une zone de tableau (voir filter_blocks_outside_tables ci-dessous).
    """
    bboxes_by_page = {}
    for t in result.tables:
        bboxes_by_page.setdefault(t.page_number, []).append(t.bbox)
    return bboxes_by_page


def _block_overlaps_bbox(block_bbox: tuple, table_bbox: tuple, overlap_threshold: float = 0.5) -> bool:
    """
    Vérifie si un bloc de texte (x0, y0, x1, y1) chevauche significativement
    une zone de tableau (x0, top, x1, bottom). Le seuil évite d'exclure un
    bloc qui ne fait que légèrement toucher le bord du tableau.
    """
    bx0, by0, bx1, by1 = block_bbox
    tx0, ttop, tx1, tbottom = table_bbox

    inter_x0 = max(bx0, tx0)
    inter_y0 = max(by0, ttop)
    inter_x1 = min(bx1, tx1)
    inter_y1 = min(by1, tbottom)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return False

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    block_area = max((bx1 - bx0) * (by1 - by0), 1e-6)

    return (inter_area / block_area) >= overlap_threshold


def filter_blocks_outside_tables(blocks: list, bboxes_by_page: dict) -> list:
    """
    Filtre une liste de TextBlock (de pdf_extractor.py) pour ne garder que
    ceux qui ne chevauchent PAS une zone de tableau déjà extraite séparément.

    Args:
        blocks: liste de TextBlock (voir pdf_extractor.py).
        bboxes_by_page: résultat de get_table_bboxes_by_page().

    Returns:
        Liste filtrée de TextBlock, sans les cellules de tableaux.
    """
    filtered = []
    for b in blocks:
        table_bboxes_this_page = bboxes_by_page.get(b.page_number, [])
        block_bbox = (b.x0, b.y0, b.x1, b.y1)

        if any(_block_overlaps_bbox(block_bbox, tb) for tb in table_bboxes_this_page):
            continue  # bloc dans un tableau -> exclu du texte narratif

        filtered.append(b)

    return filtered


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python table_extractor.py <chemin_vers_pdf>")
        sys.exit(1)

    result = extract_tables_from_pdf(sys.argv[1])
    print(f"{len(result.tables)} tableau(x) détecté(s).")
    for t in result.tables:
        print(f"  Page {t.page_number} — {t.n_rows} lignes x {t.n_cols} colonnes — bbox={t.bbox}")
        for row in t.rows[:3]:
            print(f"    {row}")
