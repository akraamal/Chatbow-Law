"""
pipeline.py
-----------

Pipeline principal.

Étapes :

1. Extraction du texte natif
2. Analyse du document
3. Détermination automatique des pages nécessitant un OCR
4. OCR
5. Détection de langue
6. (run_ingestion_pipeline) Détection de mise en page bilingue + tableaux
"""

from dataclasses import dataclass, field
from statistics import median

from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.ocr_extractor import ocr_missing_pages
from src.ingestion.language_detector import (
    detect_document_languages,
    split_document_by_language,
)
from src.ingestion.layout_splitter import detect_layout_type, split_bilingual_columns
from src.ingestion.table_extractor import (
    extract_tables_from_pdf,
    tables_to_serializable,
    get_table_bboxes_by_page,
    filter_blocks_outside_tables,
)


# ----------------------------------------------------------------------
# Analyse du document
# ----------------------------------------------------------------------

def analyze_document(document):
    """
    Analyse les pages et détermine automatiquement
    lesquelles nécessitent un OCR.
    """

    counts = [
        page.char_count
        for page in document.pages
    ]

    if not counts:
        return document

    median_chars = median(counts)

    # seuil adaptatif
    threshold = max(
        300,
        int(median_chars * 0.30)
    )

    print("\n========== PAGE ANALYSIS ==========\n")
    print(f"Median chars : {median_chars}")
    print(f"OCR threshold: {threshold}\n")

    for page in document.pages:

        page.needs_ocr = page.char_count < threshold

        print(
            f"Page {page.page_number:3d} | "
            f"{page.char_count:5d} chars | "
            f"needs_ocr={page.needs_ocr}"
        )

    return document


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------

def _strip_table_blocks(document, table_result):
    """
    Retire du document les blocs de texte qui chevauchent une zone de
    tableau déjà extraite séparément par table_extractor.py, puis
    régénère page.text et document.full_text à partir des blocs filtrés.

    Sans cette étape, une cellule de tableau (ex : barème tarifaire)
    se retrouve DEUX FOIS dans les données : une fois proprement dans
    tables[] (lignes/colonnes), et une fois aplatie en texte narratif
    dans document.pages[].text / full_text, où elle finit noyée dans
    l'article qui l'entoure. C'est la cause directe du symptôme
    "Table Destruction" observé sur BO_7510_Fr (tarifs butane).

    Doit être appelée APRÈS l'OCR (les pages OCRisées ont aussi des
    blocs, via page.blocks) mais AVANT toute segmentation en articles.
    """
    bboxes_by_page = get_table_bboxes_by_page(table_result)

    if not bboxes_by_page:
        return document  # aucun tableau détecté, rien à filtrer

    for page in document.pages:
        if not page.blocks:
            continue  # page OCR pure sans blocs positionnés : rien à filtrer ici
        page.blocks = filter_blocks_outside_tables(page.blocks, bboxes_by_page)
        page.text = "\n".join(b.text for b in page.blocks)

    if document.blocks:
        document.blocks = filter_blocks_outside_tables(document.blocks, bboxes_by_page)

    document.full_text = "\n".join(page.text for page in document.pages)

    return document


def process_pdf(pdf_path: str):

    # ----------------------------------------------------------
    # 1. Extraction native
    # ----------------------------------------------------------

    document = extract_text_from_pdf(pdf_path)

    # ----------------------------------------------------------
    # 2. Analyse
    # ----------------------------------------------------------

    document = analyze_document(document)

    # ----------------------------------------------------------
    # 3. OCR
    # ----------------------------------------------------------

    document = ocr_missing_pages(
        pdf_path,
        document
    )

    # ----------------------------------------------------------
    # 3bis. Extraction des tableaux + retrait de leurs blocs du texte
    #        narratif (AVANT toute finalisation du texte utilisée plus
    #        loin par la segmentation en articles).
    # ----------------------------------------------------------

    try:
        table_result = extract_tables_from_pdf(pdf_path)
    except Exception as e:
        print(f"    --> Extraction des tableaux échouée : {e}")
        table_result = None

    if table_result is not None and table_result.tables:
        document = _strip_table_blocks(document, table_result)

    # Attaché au document pour que run_ingestion_pipeline() n'ait pas à
    # relancer extract_tables_from_pdf() une seconde fois.
    document.table_result = table_result

    # ----------------------------------------------------------
    # 4. Détection de langue
    # ----------------------------------------------------------

    document = detect_document_languages(
        document
    )

    return document


# ----------------------------------------------------------------------
# Pipeline d'ingestion "haut niveau" (utilisé par
# scripts/run_ingestion_batch.py)
# ----------------------------------------------------------------------

@dataclass
class IngestionResult:
    """
    Résultat consolidé d'un document après ingestion complète : texte natif
    + OCR + détection de langue + séparation bilingue + tableaux.
    """
    source_path: str
    text_fr: str = ""
    text_ar: str = ""
    text_unknown: str = ""
    tables: list = field(default_factory=list)
    detected_layout: str = "pleine_page"  # "colonnes" ou "pleine_page"
    used_ocr: bool = False
    warnings: list = field(default_factory=list)


def run_ingestion_pipeline(pdf_path: str) -> IngestionResult:
    """
    Point d'entrée unique utilisé par run_ingestion_batch.py : orchestre
    l'extraction native + OCR + langue (process_pdf), puis ajoute la
    détection de mise en page bilingue (layout_splitter) et l'extraction
    des tableaux (table_extractor) pour produire un IngestionResult prêt à
    être sauvegardé sur disque.
    """
    warnings = []

    document = process_pdf(pdf_path)

    used_ocr = any(page.extraction_method == "ocr" for page in document.pages)

    layout = detect_layout_type(document)

    if layout == "colonnes":
        split = split_bilingual_columns(document)
        text_fr, text_ar, text_unknown = split.text_left, split.text_right, ""
    else:
        grouped = split_document_by_language(document)
        text_fr = grouped.get("fr", "")
        text_ar = grouped.get("ar", "")
        text_unknown = grouped.get("unknown", "")

    # table_result a déjà été calculé (et ses blocs retirés du texte
    # narratif) à l'intérieur de process_pdf() — on le réutilise au lieu
    # de relancer extract_tables_from_pdf() une seconde fois sur le PDF.
    table_result = getattr(document, "table_result", None)
    if table_result is not None:
        tables = tables_to_serializable(table_result)
    else:
        tables = []
        warnings.append("Extraction des tableaux échouée ou aucun tableau détecté.")

    if not text_fr.strip() and not text_ar.strip() and not text_unknown.strip():
        warnings.append("Aucun texte extrait (natif ni OCR) pour ce document.")

    return IngestionResult(
        source_path=str(pdf_path),
        text_fr=text_fr,
        text_ar=text_ar,
        text_unknown=text_unknown,
        tables=tables,
        detected_layout=layout,
        used_ocr=used_ocr,
        warnings=warnings,
    )


# ----------------------------------------------------------------------
# Debug
# ----------------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print("Usage:")
        print("python pipeline.py <pdf>")

        sys.exit(1)

    document = process_pdf(
        sys.argv[1]
    )

    print("\n")

    print("=" * 70)

    print("SUMMARY")

    print("=" * 70)

    for page in document.pages:

        print(
            f"Page {page.page_number:3d} | "
            f"{page.language:8s} | "
            f"{page.extraction_method:3s} | "
            f"{page.char_count:5d} chars"
        )