"""
compare_ocr_engines.py
-----------------------
Compare Tesseract (ocr_extractor) vs PaddleOCR (ocr_extractor_paddle) on the
same PDF page(s). Useful for deciding which engine to keep.

Usage:
    python -m scripts.compare_ocr_engines <pdf_path> [page_number]
    python -m scripts.compare_ocr_engines data/raw/fr/BO_7500_Fr.pdf 1
    python -m scripts.compare_ocr_engines data/raw/ar/BO_7360_Ar.pdf 2 5 10
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz

from src.ingestion import ocr_extractor as tesseract_engine
from src.ingestion import ocr_extractor_paddle as paddle_engine

RESULTS_DIR = Path(__file__).parent.parent / "testresults"


def _engine_label(lang: str) -> str:
    return "ar" if lang == "ara" else "fr"


def _detect_lang(pdf_path: str) -> str:
    stem = Path(pdf_path).stem.lower()
    if "_ar" in stem or stem.startswith("ar_"):
        return "ara"
    return "fra"


def _render_page(page, dpi=400):
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img.convert("RGB")


def compare_single_page(pdf_path: str, page_number: int):
    stem = Path(pdf_path).stem
    lang = _detect_lang(pdf_path)
    tess_lang = "ara" if lang == "ara" else "fra+ara"

    print(f"\n{'='*70}")
    print(f"  Page {page_number}")
    print(f"{'='*70}")

    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        pil_image = _render_page(page)

    # Tesseract
    t0 = time.time()
    try:
        tess_text, tess_conf = tesseract_engine._ocr_image(pil_image, lang=tess_lang)
        tess_time = time.time() - t0
    except Exception as e:
        tess_text, tess_conf, tess_time = "", 0.0, 0.0
        print(f"  [Tesseract] ERROR: {e}")

    # PaddleOCR (pure)
    t0 = time.time()
    try:
        paddle_text, paddle_conf = paddle_engine._ocr_image(pil_image, lang=lang)
        paddle_time = time.time() - t0
    except Exception as e:
        paddle_text, paddle_conf, paddle_time = "", 0.0, 0.0
        print(f"  [PaddleOCR] ERROR: {e}")

    # Hybride (détection PaddleOCR + reconnaissance Tesseract)
    t0 = time.time()
    try:
        hybrid_text, hybrid_conf = paddle_engine._ocr_image_hybrid(pil_image, lang=lang)
        hybrid_time = time.time() - t0
    except Exception as e:
        hybrid_text, hybrid_conf, hybrid_time = "", 0.0, 0.0
        print(f"  [Hybride] ERROR: {e}")

    # Save OCR text files
    out_dir = RESULTS_DIR / f"{stem}_p{page_number:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tesseract.txt").write_text(tess_text, encoding="utf-8")
    (out_dir / "paddleocr.txt").write_text(paddle_text, encoding="utf-8")
    (out_dir / "hybrid.txt").write_text(hybrid_text, encoding="utf-8")
    print(f"  → Saved to {out_dir}")

    # Summary table
    print(f"\n  {'Engine':<20} {'Chars':>8} {'Conf':>8} {'Time (s)':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'Tesseract':<20} {len(tess_text):>8} {tess_conf:>7.1f}% {tess_time:>9.2f}")
    print(f"  {'PaddleOCR':<20} {len(paddle_text):>8} {paddle_conf:>7.1f}% {paddle_time:>9.2f}")
    print(f"  {'Hybride':<20} {len(hybrid_text):>8} {hybrid_conf:>7.1f}% {hybrid_time:>9.2f}")

    if tess_text and hybrid_text:
        overlap = len(set(tess_text.split()) & set(hybrid_text.split()))
        total = len(set(tess_text.split()) | set(hybrid_text.split()))
        jaccard = overlap / total * 100 if total else 0
        print(f"\n  Jaccard (Tesseract × Hybride): {jaccard:.1f}%")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Fichier introuvable : {pdf_path}")
        sys.exit(1)

    pages = [int(a) for a in sys.argv[2:]] if len(sys.argv) > 2 else None

    with fitz.open(pdf_path) as doc:
        total = doc.page_count

    if pages:
        for p in pages:
            if 1 <= p <= total:
                compare_single_page(pdf_path, p)
            else:
                print(f"Page {p} hors limites (1-{total})")
    else:
        for p in range(1, total + 1):
            compare_single_page(pdf_path, p)


if __name__ == "__main__":
    main()
