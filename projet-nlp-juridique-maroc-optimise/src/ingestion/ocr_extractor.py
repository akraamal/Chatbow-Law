"""
ocr_extractor.py
------------------
Extraction de texte par OCR pour les PDF scannés (images), en français et
en arabe.

Utilise :
  - PyMuPDF (fitz) pour rasteriser les pages du PDF en images
  - pytesseract pour l'OCR, avec les packs de langue "fra" et "ara"

Installation :
    pip install pymupdf pytesseract pillow

Le binaire Tesseract doit être installé séparément (pas via pip) :
    Ubuntu/Debian :
        sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara
    macOS (Homebrew) :
        brew install tesseract tesseract-lang
    Windows :
        https://github.com/UB-Mannheim/tesseract/wiki

Vérifier les langues installées :
    tesseract --list-langs

Chemin du binaire Tesseract :
    Sur Linux/macOS, pytesseract trouve automatiquement le binaire dans le
    PATH. Sur Windows (ou si l'installation est dans un emplacement non
    standard), définir la variable d'environnement TESSERACT_CMD, ex :
        set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""
import concurrent.futures
import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

_TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
elif sys.platform == "win32":
    _common_win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in _common_win_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break


@dataclass
class OCRPageResult:
    page_number: int
    text: str
    mean_confidence: float  # confiance moyenne OCR (0-100), utile pour évaluer la qualité

def _adaptive_dpi(native_chars: int, threshold: int) -> int:
    """
    Choisit un DPI adapté selon le nombre de caractères natifs extraits :
      - native_chars < 100     → DPI 400  (page quasi-vide, besoin max de détail)
      - native_chars < 500     → DPI 350
      - native_chars < threshold * 0.7 → DPI 300
      - sinon                  → DPI 200  (juste besoin de gagner min_gain chars)
    """
    if native_chars < 100:
        return 400
    if native_chars < 500:
        return 350
    if native_chars < int(threshold * 0.7):
        return 300
    return 200


def ocr_single_page(
    page,
    dpi=400,
    lang="fra+ara",
    psm: int | None = None,
    debug=False,
    debug_path="debug_page.png",
):

    rendered_image = _render_page_to_image(
        page,
        dpi
    )

    if debug:
        rendered_image.save(debug_path)

    text, conf = _ocr_image(
        rendered_image,
        lang,
        psm=psm,
    )

    return text, conf


def ocr_missing_pages(
    pdf_path: str,
    extracted_document,
    min_gain: int = 100,
    max_workers: int | None = None,
    lang: str | None = None,
):
    """
    OCR uniquement les pages marquées 'needs_ocr', en parallèle.

    Si l'OCR produit significativement plus de texte que
    l'extraction native, on remplace le texte de la page.

    Args:
        max_workers: nombre de workers parallèles (None = auto, 1 = séquentiel).
        lang: code de langue Tesseract (ex: "ara", "fra", "fra+ara").
              Si None, déduit du nom du fichier PDF.
    """

    if lang is None:
        lang = _infer_lang_from_path(pdf_path)

    psm = _lang_to_psm(lang.split("+")[0])

    with fitz.open(pdf_path) as doc:

        print("\n========== OCR ==========\n")

        # 1) Calculer le seuil adaptatif (celui utilisé dans analyze_document)
        counts = [p.char_count for p in extracted_document.pages if p.needs_ocr]
        if counts:
            from statistics import median
            threshold = max(300, int(median(counts) * 0.30))
        else:
            threshold = 300

        # 2) Pré-rendre les images dans le thread principal (fitz n'est pas
        #    thread-safe pour l'accès simultané au document).
        tasks = []
        for page in extracted_document.pages:
            if not page.needs_ocr:
                continue
            pdf_page = doc[page.page_number - 1]
            dpi = _adaptive_dpi(page.char_count, threshold)
            image = _render_page_to_image(pdf_page, dpi)
            tasks.append((page, image, dpi))

        if not tasks:
            extracted_document.full_text = "\n".join(
                p.text for p in extracted_document.pages
            )
            return extracted_document

        # 3) OCR en parallèle
        def _ocr_task(img, ocr_lang, ocr_psm):
            return _ocr_image(img, ocr_lang, psm=ocr_psm)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_map = {}
            for page, image, dpi in tasks:
                print(f"OCR page {page.page_number} (DPI={dpi}, PSM={psm})")
                fut = executor.submit(_ocr_task, image, lang, psm)
                fut_map[fut] = (page, dpi)

            for fut in concurrent.futures.as_completed(fut_map):
                page, dpi = fut_map[fut]
                ocr_text, confidence = fut.result()
                native_chars = page.char_count
                ocr_chars = len(ocr_text.strip())

                print(f"    Native : {native_chars} chars")
                print(f"    OCR    : {ocr_chars} chars")
                print(f"    Conf   : {confidence:.1f}")

                need_ocr_gain = ocr_chars > native_chars + min_gain
                needs_ocr_and_decent = (
                    page.needs_ocr
                    and confidence > 0.7
                    and ocr_chars > native_chars * 0.5
                )
                use_ocr = need_ocr_gain or needs_ocr_and_decent

                if use_ocr:
                    print("    --> OCR selected\n")
                    page.text = ocr_text
                    page.char_count = ocr_chars
                    page.extraction_method = "ocr"
                else:
                    print("    --> Native text kept\n")

    extracted_document.full_text = "\n".join(
        page.text
        for page in extracted_document.pages
    )

    return extracted_document

@dataclass
class OCRDocument:
    source_path: str
    full_text: str
    pages: list = field(default_factory=list)
    n_pages: int = 0


def _render_page_to_image(
    page,
    dpi=400,
):

    zoom = dpi / 72

    matrix = fitz.Matrix(
        zoom,
        zoom
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    img = Image.open(
        io.BytesIO(
            pix.tobytes("png")
        )
    )

    return img.convert("RGB")

def _lang_to_psm(lang: str) -> int:
    """
    Choisit le mode de segmentation de page (PSM) selon la langue :
      - arabe ('ara')  → 6  (uniform block of text, meilleur pour RTL)
      - français ('fra') → 4 (single column of variable sizes, meilleur pour LTR)
      - mixte / inconnu → 6  (valeur par défaut, prudent pour l'arabe)
    """
    if lang == "fra":
        return 4
    if lang == "ara":
        return 6
    return 6


def _infer_lang_from_path(path: str) -> str:
    """
    Infère la langue du document depuis le nom du fichier PDF.
    Conventions : BO_XXXX_Fr.pdf → fra, BO_XXXX_Ar.pdf → ara
    """
    from pathlib import Path
    stem = Path(path).stem
    if stem.endswith("_Fr"):
        return "fra"
    if stem.endswith("_Ar"):
        return "ara"
    return "fra+ara"


# ── Détection de colonnes (portée depuis column_aware_ocr.py) ───────────
# Sans ceci, _ocr_image() OCRise la page entière en un seul appel
# pytesseract, à plat, de gauche à droite — ce qui entrelace deux
# colonnes sur une page scannée à 2 colonnes (confirmé : la page de
# sommaire/couverture de BO_7510_Fr reste incorrectement mélangée même
# après correction du découpage en colonnes de pdf_extractor.py, car
# cette page passe par l'OCR — donc par CE fichier — et non par
# l'extraction native).
GUTTER_MIN_CONF = 20              # confiance OCR minimale pour un mot pris en compte
GUTTER_MID_BAND = (0.30, 0.70)    # ne cherche le gutter que dans la bande centrale de la page
GUTTER_MIN_GAP_RATIO = 0.02       # le gap doit dépasser 2% de la largeur de page pour être un vrai gutter
HEADER_BAND_FRACTION = 0.08       # bandeau d'en-tête à traiter séparément (pas de logique colonnes)


def _detect_gutter(image: Image.Image, lang: str):
    """
    Renvoie la coordonnée x du gutter (espace entre 2 colonnes), ou None
    si la page semble être en une seule colonne (aucun gap fiable trouvé).
    """
    w, h = image.size
    probe_lang = lang.split("+")[0]

    data = pytesseract.image_to_data(image, lang=probe_lang, output_type=pytesseract.Output.DATAFRAME)
    data = data[data.conf > GUTTER_MIN_CONF].dropna(subset=["text"])
    data = data[data.text.str.strip() != ""]
    if len(data) < 20:
        return None

    data["center_x"] = data["left"] + data["width"] / 2
    centers = np.sort(data["center_x"].values)

    mid_band = centers[(centers > w * GUTTER_MID_BAND[0]) & (centers < w * GUTTER_MID_BAND[1])]
    if len(mid_band) < 2:
        return None

    mid_band = np.sort(mid_band)
    gaps = np.diff(mid_band)
    biggest_gap_idx = np.argmax(gaps)
    biggest_gap = gaps[biggest_gap_idx]

    if biggest_gap < w * GUTTER_MIN_GAP_RATIO:
        return None

    return (mid_band[biggest_gap_idx] + mid_band[biggest_gap_idx + 1]) / 2


def _mean_confidence_from_image(image: Image.Image, lang: str) -> float:
    """Calcule la confiance OCR moyenne d'une image (facteur commun,
    utilisé aussi bien pour l'OCR pleine page que pour chaque colonne)."""
    data = pytesseract.image_to_data(
        image, lang=lang, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
    )
    confidences = []
    for conf in data["conf"]:
        try:
            conf = float(conf)
            if conf >= 0:
                confidences.append(conf)
        except ValueError:
            pass
    return sum(confidences) / len(confidences) if confidences else 0.0


def _ocr_image(
    image: Image.Image,
    lang: str = "fra+ara",
    psm: int | None = None,
) -> tuple[str, float]:
    """
    OCR d'une image, avec prise en compte des colonnes.

    Paramètres :
        lang : code de langue Tesseract ("fra", "ara", "fra+ara"...)
        psm  : mode de segmentation de page (6 pour arabe, 4 pour français).
               Si None, il est déduit automatiquement de l'argument lang.

    Comportement :
        1. Le bandeau d'en-tête (haut de page, running header répété) est
           toujours OCRisé à part, en pleine largeur — un en-tête n'a
           jamais de mise en page en colonnes.
        2. Le reste de la page est analysé pour détecter un gutter de
           colonnes (voir _detect_gutter). S'il y en a un, chaque colonne
           est rognée et OCRisée séparément, puis concaténée dans l'ordre
           de lecture (colonne gauche entière, puis colonne droite) —
           au lieu d'un seul appel à plat qui entrelacerait les 2
           colonnes ligne par ligne.
        3. Si aucun gutter fiable n'est trouvé (page mono-colonne,
           tableau, page de titre), on OCRise le corps de page entier
           en un seul appel, comme avant.

    Retourne :
        texte reconnu
        confiance moyenne
    """
    image = image.convert("L")   # grayscale

    if psm is None:
        single = lang.split("+")[0]
        psm = _lang_to_psm(single)

    w, h = image.size
    header_cutoff = int(h * HEADER_BAND_FRACTION)
    header_img = image.crop((0, 0, w, header_cutoff))
    body_img = image.crop((0, header_cutoff, w, h))

    header_text = pytesseract.image_to_string(header_img, lang=lang, config=f"--oem 3 --psm 6")

    gutter_x = _detect_gutter(body_img, lang)

    if gutter_x is None:
        body_text = pytesseract.image_to_string(body_img, lang=lang, config=f"--oem 3 --psm {psm}")
        combined_text = "\n".join([header_text.strip(), body_text.strip()])
        mean_confidence = _mean_confidence_from_image(image, lang)
        return combined_text.strip(), mean_confidence

    left_col = body_img.crop((0, 0, int(gutter_x), body_img.height))
    right_col = body_img.crop((int(gutter_x), 0, w, body_img.height))

    left_text = pytesseract.image_to_string(left_col, lang=lang, config=f"--oem 3 --psm {psm}")
    right_text = pytesseract.image_to_string(right_col, lang=lang, config=f"--oem 3 --psm {psm}")

    combined_text = "\n".join([header_text.strip(), left_text.strip(), right_text.strip()])

    left_conf = _mean_confidence_from_image(left_col, lang)
    right_conf = _mean_confidence_from_image(right_col, lang)
    confs = [c for c in (left_conf, right_conf) if c > 0]
    mean_confidence = sum(confs) / len(confs) if confs else 0.0

    return combined_text.strip(), mean_confidence


def extract_text_with_ocr(
    pdf_path: str,
    lang: str = "fra+ara",
    dpi: int = 300,
) -> OCRDocument:
    """
    Extrait le texte d'un PDF scanné via OCR, page par page.

    Args:
        pdf_path: chemin vers le PDF scanné.
        lang: langue(s) Tesseract ("fra", "ara", ou "fra+ara" pour les deux).
        dpi: résolution de rasterisation (300 recommandé).

    Returns:
        OCRDocument avec le texte de chaque page et un score de confiance.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

    with fitz.open(pdf_path) as doc:
        pages_result = []
        full_text_parts = []

        for page_index, page in enumerate(doc):
            image = _render_page_to_image(page, dpi=dpi)
            text, confidence = _ocr_image(image, lang=lang)

            pages_result.append(
                OCRPageResult(page_number=page_index + 1, text=text, mean_confidence=confidence)
            )
            full_text_parts.append(text)

        n_pages = doc.page_count

    return OCRDocument(
        source_path=str(pdf_path),
        full_text="\n".join(full_text_parts),
        pages=pages_result,
        n_pages=n_pages,
    )


def extract_text_with_ocr_bilingual_columns(
    pdf_path: str,
    dpi: int = 300,
    split_ratio: float = 0.5,
) -> dict:
    """
    Variante pour PDF scannés bilingues en 2 colonnes (FR à gauche, AR à
    droite). Découpe chaque image de page en 2 moitiés avant l'OCR, et
    applique la langue adéquate à chaque moitié — plus précis que d'OCRiser
    la page entière avec "fra+ara", car Tesseract se trompe moins quand on
    lui indique la bonne langue par zone.

    Args:
        split_ratio: proportion horizontale du split (0.5 = milieu de page).
            À ajuster si les colonnes ne sont pas de largeur égale.

    Returns:
        dict {"fr": OCRDocument, "ar": OCRDocument}
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

    with fitz.open(pdf_path) as doc:
        fr_pages, ar_pages = [], []

        for page_index, page in enumerate(doc):
            image = _render_page_to_image(page, dpi=dpi)
            width, height = image.size
            split_x = int(width * split_ratio)

            left_image = image.crop((0, 0, split_x, height))
            right_image = image.crop((split_x, 0, width, height))

            fr_text, fr_conf = _ocr_image(left_image, lang="fra")
            ar_text, ar_conf = _ocr_image(right_image, lang="ara")

            fr_pages.append(OCRPageResult(page_number=page_index + 1, text=fr_text, mean_confidence=fr_conf))
            ar_pages.append(OCRPageResult(page_number=page_index + 1, text=ar_text, mean_confidence=ar_conf))

        n_pages = doc.page_count

    return {
        "fr": OCRDocument(
            source_path=str(pdf_path),
            full_text="\n".join(p.text for p in fr_pages),
            pages=fr_pages,
            n_pages=n_pages,
        ),
        "ar": OCRDocument(
            source_path=str(pdf_path),
            full_text="\n".join(p.text for p in ar_pages),
            pages=ar_pages,
            n_pages=n_pages,
        ),
    }


def check_tesseract_languages_available() -> list:
    """Vérifie quelles langues sont installées sur la machine (diagnostic)."""
    try:
        langs = pytesseract.get_languages(config="")
        return langs
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract n'est pas installé ou introuvable dans le PATH. "
            "Voir les instructions d'installation en haut de ce fichier."
        )

if __name__ == "__main__":
    import sys
    import fitz

    if len(sys.argv) != 3:
        print("Usage:")
        print("python src/ingestion/ocr_extractor.py <pdf> <page_number>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2])

    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        text, conf = ocr_single_page(
            page,
            dpi=400,
            lang="ara+fra"
        )

    print("=" * 80)
    print(f"Page {page_number}")
    print(f"Confidence: {conf:.2f}")
    print("=" * 80)
    print(text)