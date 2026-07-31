"""
Column-aware OCR extraction for Bulletin Officiel PDFs.

PROBLEM DIAGNOSED (confirmed on BO_6788_Fr.pdf directly):
    These PDFs have NO embedded text layer at all -- every page is a scanned
    bilevel image (CCITTFaxDecode). Whatever produced BO_6788_Fr.txt / the
    entities JSON must have run OCR (pymupdf alone cannot extract text from
    an image-only PDF). Naive OCR (Tesseract's default page segmentation, or
    pymupdf's get_text() on a rasterized page without layout awareness)
    reads straight across the page left-to-right, line-by-line -- which
    means on a two-column layout it interleaves the two columns into one
    garbled stream. This is exactly the corruption seen in the JSON
    (e.g. "Code de commerce" merging mid-sentence with an unrelated
    electricity/water-financing TOC entry from the adjacent column).

FIX: detect the column gutter per page from OCR word bounding boxes, crop
the page into left/right column images, OCR each column independently, and
concatenate them in the correct reading order (left column complete, then
right column). Falls back to single-column OCR when no clear gutter is
found (covers tables, decree headers, cover pages, etc).

Usage:
    pip install pymupdf pytesseract pillow numpy pandas
    apt-get install tesseract-ocr tesseract-ocr-fra   # or your OS equivalent

    python column_aware_ocr.py BO_6788_Fr.pdf BO_6788_Fr_clean.txt
"""

import sys
import re
import numpy as np
import pandas as pd
import pytesseract
import fitz  # pymupdf
from PIL import Image

import os
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

OCR_LANG = "fra"          # switch to "ara" for Arabic editions, or "fra+ara"
RENDER_DPI = 300          # 300 is a good OCR quality/speed tradeoff; try 400 if accuracy is still poor
MIN_CONF = 20             # discard low-confidence OCR word boxes before gutter detection
HEADER_BAND_FRACTION = 0.06  # skip the running header ("BULLETIN OFFICIEL N°...") when detecting the gutter


def render_page(page, dpi=RENDER_DPI) -> Image.Image:
    """Render a pymupdf page to a PIL image at the given DPI."""
    pix = page.get_pixmap(dpi=dpi)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img.convert("L")  # grayscale is enough for these bilevel scans and OCRs faster


def detect_gutter(img: Image.Image):
    """
    Return the x-coordinate of the column gutter, or None if the page looks
    single-column (no reliable gap found).
    """
    w, h = img.size
    data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DATAFRAME)
    data = data[data.conf > MIN_CONF].dropna(subset=["text"])
    data = data[data.text.str.strip() != ""]
    if len(data) < 20:
        return None

    data["center_x"] = data["left"] + data["width"] / 2
    centers = np.sort(data["center_x"].values)

    # Only look for the gutter in the middle band of the page -- avoids
    # being misled by ordinary word-spacing gaps near the margins.
    mid_band = centers[(centers > w * 0.30) & (centers < w * 0.70)]
    if len(mid_band) < 2:
        return None

    mid_band = np.sort(mid_band)
    gaps = np.diff(mid_band)
    biggest_gap_idx = np.argmax(gaps)
    biggest_gap = gaps[biggest_gap_idx]

    # Require the gap to be meaningfully large (empirically, a real column
    # gutter is much wider than normal inter-word spacing at this DPI).
    if biggest_gap < w * 0.02:
        return None

    gutter_x = (mid_band[biggest_gap_idx] + mid_band[biggest_gap_idx + 1]) / 2
    return gutter_x


def ocr_page(page) -> str:
    """OCR a single pymupdf page, column-aware."""
    img = render_page(page)
    w, h = img.size
    header_cutoff = int(h * HEADER_BAND_FRACTION)

    gutter_x = detect_gutter(img)

    if gutter_x is None:
        # single column (or a table/cover page) -- OCR the whole page as-is
        return pytesseract.image_to_string(img, lang=OCR_LANG, config="--psm 4")

    header_img = img.crop((0, 0, w, header_cutoff))
    left_col = img.crop((0, header_cutoff, int(gutter_x), h))
    right_col = img.crop((int(gutter_x), header_cutoff, w, h))

    header_text = pytesseract.image_to_string(header_img, lang=OCR_LANG, config="--psm 6")
    left_text = pytesseract.image_to_string(left_col, lang=OCR_LANG, config="--psm 4")
    right_text = pytesseract.image_to_string(right_col, lang=OCR_LANG, config="--psm 4")

    return "\n".join([header_text.strip(), left_text.strip(), right_text.strip()])


def extract_pdf(pdf_path: str) -> list[str]:
    """Return a list of per-page cleaned text."""
    with fitz.open(pdf_path) as doc:
        pages_text = []
        for i, page in enumerate(doc):
            text = ocr_page(page)
            pages_text.append(text)
            print(f"  page {i+1}/{len(doc)} done ({len(text)} chars)", file=sys.stderr)
    return pages_text


# ---------------------------------------------------------------------------
# ARTICLE SPLITTING FIX
#
# The old regex was collapsing sub-numbered articles ("Article 544-1",
# "544-2", "544-3"...) down to a single "544", overwriting 12 distinct
# articles into one JSON entry. This version keeps the full number,
# including the "premier" (first) article and hyphenated sub-numbers, and
# also tags which decree/law each article belongs to (BO issues bundle many
# separate laws, each of which restarts its own "Article premier").
# ---------------------------------------------------------------------------

ARTICLE_RE = re.compile(
    r"Article\s+(premier|\d+(?:[\-‑]\d+)?)\s*[\.\-–]?\s*",
    re.IGNORECASE,
)
DECREE_RE = re.compile(
    r"(Dahir|D[ée]cret)\s+n[°ºo]\s*[\d\-]+[^\n]{0,120}",
    re.IGNORECASE,
)


def split_into_articles(full_text: str) -> list[dict]:
    """
    Split cleaned document text into articles, each tagged with its parent
    decree/law so numbering restarts across bundled texts don't collide.
    """
    # First, split the whole document by decree/law boundaries.
    decree_bounds = [m.start() for m in DECREE_RE.finditer(full_text)]
    decree_bounds.append(len(full_text))

    articles = []
    for d_idx in range(len(decree_bounds) - 1):
        start, end = decree_bounds[d_idx], decree_bounds[d_idx + 1]
        chunk = full_text[start:end]
        decree_match = DECREE_RE.search(chunk)
        decree_label = decree_match.group(0).strip() if decree_match else f"section_{d_idx}"

        matches = list(ARTICLE_RE.finditer(chunk))
        for i, m in enumerate(matches):
            art_start = m.end()
            art_end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk)
            articles.append({
                "parent_decree": decree_label,
                "number": m.group(1).lower(),
                "raw_header": m.group(0).strip(),
                "text": chunk[art_start:art_end].strip(),
            })
    return articles


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python column_aware_ocr.py input.pdf output.txt", file=sys.stderr)
        sys.exit(1)

    pdf_path, out_path = sys.argv[1], sys.argv[2]
    print(f"OCR'ing {pdf_path} (column-aware)...", file=sys.stderr)
    pages = extract_pdf(pdf_path)
    full_text = "\n\n".join(pages)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Saved cleaned text to {out_path}", file=sys.stderr)

    articles = split_into_articles(full_text)
    print(f"Detected {len(articles)} articles across {len(set(a['parent_decree'] for a in articles))} decree(s)/law(s).", file=sys.stderr)