"""
ocr_extractor_paddle.py
------------------------
Extraction de texte par OCR pour les PDF scannés (images), en français et
en arabe — via PaddleOCR.

Fichier VOLONTAIREMENT séparé de ocr_extractor.py (Tesseract), pour
pouvoir comparer les deux moteurs sur les mêmes pages avant de décider
lequel garder. Voir scripts/compare_ocr_engines.py.

Tant que la comparaison n'a pas tranché, src/ingestion/pipeline.py continue
d'importer ocr_extractor.py (Tesseract) — ce fichier n'est pas branché au
pipeline principal.

Pourquoi PaddleOCR pourrait être un meilleur choix ici :
  - Détection de zones de texte plus robuste sur des mises en page denses
    comme le Bulletin Officiel.
  - ATTENTION : le modèle de RECONNAISSANCE arabe de PaddleOCR (PP-OCRv4)
    est médiocre sur les fonts historiques du BO marocain — il produit
    du texte inexploitable malgré une confiance par caractère élevée.
    Solution : utiliser le mode hybride _ocr_image_hybrid() qui combine
    la DÉTECTION PaddleOCR (robuste) avec la RECONNAISSANCE Tesseract
    (précise sur ces documents).
  - ATTENTION : ça ne résout pas à lui seul le bug d'ordre des colonnes
    documenté dans le changelog — on garde donc la même stratégie de
    découpage en 2 colonnes (FR à gauche, AR à droite) avant OCR, plutôt
    que de compter sur la mise en page automatique de PaddleOCR.

Installation :
    pip install paddlepaddle paddleocr pymupdf pillow
    # GPU (optionnel, accélère nettement) :
    #   pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

Premier lancement : PaddleOCR télécharge ses modèles (~100-200 Mo par
langue) depuis Hugging Face/BOS et les met en cache dans ~/.paddlex/ —
nécessite une connexion internet la première fois seulement. À tester
AVANT la soutenance, pas le jour même.
"""
import io
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Moteurs PaddleOCR mis en cache — l'instanciation charge les modèles en
# mémoire (coûteux), donc on ne crée un moteur qu'une seule fois par langue
# et on le réutilise, contrairement à pytesseract qui était stateless.
# ---------------------------------------------------------------------------
_ENGINES: dict = {}


def _get_engine(lang: str):
    """
    Renvoie un moteur PaddleOCR pour la langue demandée, en le créant et le
    mettant en cache au premier appel.

    Codes langue : "fr" pour le français, "ar" pour l'arabe. Si PaddleOCR
    lève une erreur sur le code langue (ça arrive selon les versions),
    essayer "arabic" à la place de "ar" — c'est le nom du groupe de
    caractères utilisé en interne pour l'arabe/persan/ourdou.
    """
    if lang not in _ENGINES:
        try:
            import torch  # noqa: F401; doit précéder paddleocr sur Windows
        except Exception:
            pass

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "paddleocr n'est pas installé. Exécutez :\n"
                "    pip install paddlepaddle paddleocr"
            )
        _ENGINES[lang] = PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            show_log=False,
        )
    return _ENGINES[lang]


@dataclass
class OCRPageResult:
    page_number: int
    text: str
    mean_confidence: float  # confiance moyenne OCR (0-100), comparable à ocr_extractor.py


@dataclass
class OCRDocument:
    source_path: str
    full_text: str
    pages: list = field(default_factory=list)
    n_pages: int = 0


def _render_page_to_image(page, dpi=400):
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img.convert("RGB")


def _lang_code_for_paddle(lang: str) -> str:
    """Traduit nos codes internes ('fra', 'ara') vers les codes PaddleOCR
    ('fr', 'ar')."""
    mapping = {"fra": "fr", "ara": "ar", "fr": "fr", "ar": "ar"}
    if lang in mapping:
        return mapping[lang]
    print(f"[ocr_extractor_paddle] Langue '{lang}' non reconnue, repli sur 'fr'.")
    return "fr"


def _ocr_image(image: Image.Image, lang: str = "fra") -> tuple[str, float]:
    """
    OCR d'une image avec PaddleOCR 2.x.

    Retourne :
        texte reconnu (lignes jointes dans l'ordre de détection),
        confiance moyenne ramenée sur 0-100 (comparable à ocr_extractor.py).
    """
    engine = _get_engine(_lang_code_for_paddle(lang))

    image_np = np.array(image.convert("RGB"))
    # API 2.x : engine.ocr() retourne list[list[bbox, (text, score)]]
    results = engine.ocr(image_np, cls=False)

    if not results or not results[0]:
        return "", 0.0

    lines = results[0]
    texts = []
    scores = []

    for line in lines:
        if not line or len(line) < 2:
            continue
        text_tuple = line[1]
        texts.append(text_tuple[0])
        scores.append(float(text_tuple[1]))

    text = "\n".join(texts)
    mean_confidence = (sum(scores) / len(scores) * 100) if scores else 0.0

    return text.strip(), mean_confidence


def _ocr_image_hybrid(
    image: Image.Image,
    lang: str = "fra",
    det_conf_threshold: float = 0.3,
) -> tuple[str, float]:
    """
    Hybride : PaddleOCR pour la DÉTECTION des zones de texte + Tesseract pour
    la RECONNAISSANCE sur chaque zone recadrée.

    PaddleOCR a un excellent détecteur de zones de texte (indépendant de la
    police), mais son modèle de reconnaissance arabe est médiocre sur les
    fonts historiques du BO marocain. Tesseract a l'inverse : meilleure
    reconnaissance arabe mais détection de zones parfois moins robuste.

    Cette fonction combine le meilleur des deux.

    Args:
        det_conf_threshold: seuil de confiance de détection PaddleOCR
            (0-1). Les zones sous ce seuil sont ignorées.

    Retourne :
        texte reconnu (lignes jointes ordonnées de haut en bas, puis
        gauche à droite),
        confiance moyenne Tesseract (0-100, comparable aux autres fonctions).
    """
    import pytesseract

    engine = _get_engine(_lang_code_for_paddle(lang))
    image_np = np.array(image.convert("RGB"))
    results = engine.ocr(image_np, cls=False)

    if not results or not results[0]:
        return "", 0.0

    tess_lang = "ara" if lang in ("ara", "ar") else "fra+ara"
    confidences = []

    # Fusionner les bbox proches horizontalement (même ligne)
    boxes = []
    for line in results[0]:
        if not line or len(line) < 2:
            continue
        bbox, (rec_text, rec_conf) = line
        if rec_conf < det_conf_threshold:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        cy = (y1 + y2) / 2
        boxes.append((cy, x1, y1, x2, y2))

    if not boxes:
        return "", 0.0

    # Trier par y, puis fusionner les boîtes sur la même ligne (cy à <25px)
    boxes.sort(key=lambda b: b[0])
    lines_boxes = []
    current = [boxes[0]]
    for b in boxes[1:]:
        if b[0] - current[-1][0] <= 25:
            current.append(b)
        else:
            lines_boxes.append(current)
            current = [b]
    lines_boxes.append(current)

    # Pour chaque ligne, agréger horizontalement et OCRiser la bande entière
    h_margin = 5
    v_margin = 3
    texts = []

    for lb in lines_boxes:
        lb.sort(key=lambda b: b[1])
        x1 = max(0, min(b[1] for b in lb) - h_margin)
        y1 = max(0, min(b[2] for b in lb) - v_margin)
        x2 = min(image.width, max(b[3] for b in lb) + h_margin)
        y2 = min(image.height, max(b[4] for b in lb) + v_margin)

        crop = image.crop((x1, y1, x2, y2))
        text = pytesseract.image_to_string(
            crop, lang=tess_lang, config="--oem 3 --psm 6"
        ).strip()

        if not text:
            continue
        texts.append(text)

        data = pytesseract.image_to_data(
            crop, lang=tess_lang, config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
        vals = [float(c) for c in data["conf"] if c != "-1"]
        avg_conf = sum(vals) / len(vals) if vals else 0.0
        confidences.append(avg_conf)

    if not texts:
        return "", 0.0

    text = "\n".join(texts)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return text.strip(), mean_conf


def ocr_single_page(page, dpi=400, lang="fra", debug=False, debug_path="debug_page_paddle.png"):
    rendered_image = _render_page_to_image(page, dpi)

    if debug:
        rendered_image.save(debug_path)

    text, conf = _ocr_image(rendered_image, lang)
    return text, conf


def ocr_missing_pages(pdf_path: str, extracted_document, min_gain: int = 100):
    """
    Équivalent PaddleOCR de ocr_extractor.ocr_missing_pages() — même
    signature, pour pouvoir être branché à pipeline.py par un simple
    changement d'import une fois la comparaison faite.

    NOTE : contrairement à Tesseract ("fra+ara" en un seul appel),
    PaddleOCR ne supporte pas nativement le mélange de langues sur une
    même image. On OCRise en français par défaut ici — passe lang="ara"
    si le document est majoritairement arabe.
    """
    doc = fitz.open(pdf_path)

    print("\n========== OCR (PaddleOCR) ==========\n")

    for page in extracted_document.pages:
        if not page.needs_ocr:
            continue

        pdf_page = doc[page.page_number - 1]
        print(f"OCR page {page.page_number}")

        ocr_text, confidence = ocr_single_page(pdf_page, dpi=400, lang="fra")

        native_chars = page.char_count
        ocr_chars = len(ocr_text.strip())

        print(f"    Native : {native_chars} chars")
        print(f"    OCR    : {ocr_chars} chars")
        print(f"    Conf   : {confidence:.1f}")

        if ocr_chars > native_chars + min_gain:
            print("    --> OCR selected\n")
            page.text = ocr_text
            page.char_count = ocr_chars
            page.extraction_method = "ocr"
        else:
            print("    --> Native text kept\n")

    doc.close()

    extracted_document.full_text = "\n".join(page.text for page in extracted_document.pages)
    return extracted_document


def extract_text_with_ocr(pdf_path: str, lang: str = "fra", dpi: int = 300) -> OCRDocument:
    """
    Extrait le texte d'un PDF scanné via OCR, page par page, dans UNE seule
    langue. Pour un document bilingue FR/AR en colonnes, utilise plutôt
    extract_text_with_ocr_bilingual_columns().
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

    doc = fitz.open(pdf_path)
    pages_result = []
    full_text_parts = []

    for page_index, page in enumerate(doc):
        image = _render_page_to_image(page, dpi=dpi)
        text, confidence = _ocr_image(image, lang=lang)

        pages_result.append(OCRPageResult(page_number=page_index + 1, text=text, mean_confidence=confidence))
        full_text_parts.append(text)

    n_pages = doc.page_count
    doc.close()

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
    droite) — même découpage que dans ocr_extractor.py, avec un moteur
    PaddleOCR par colonne au lieu de Tesseract.

    Args:
        split_ratio: proportion horizontale du split (0.5 = milieu de page).

    Returns:
        dict {"fr": OCRDocument, "ar": OCRDocument}
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

    doc = fitz.open(pdf_path)
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
    doc.close()

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


def check_paddleocr_languages_available() -> list:
    """Diagnostic : tente d'instancier fr/ar pour vérifier que les modèles
    PaddleOCR sont accessibles (réseau ou cache local présent)."""
    available = []
    for lang in ("fr", "ar"):
        try:
            _get_engine(lang)
            available.append(lang)
        except Exception as exc:
            print(f"[ocr_extractor_paddle] Langue '{lang}' indisponible : {exc}")
    return available


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage:")
        print("python src/ingestion/ocr_extractor_paddle.py <pdf> <page_number>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2])

    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        text, conf = ocr_single_page(page, dpi=400, lang="fra")

    print("=" * 80)
    print(f"Page {page_number}")
    print(f"Confidence: {conf:.2f}")
    print("=" * 80)
    print(text)