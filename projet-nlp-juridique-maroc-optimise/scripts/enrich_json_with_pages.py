"""
enrich_json_with_pages.py
-------------------------
Post-processing script for Priority 1 (page references) and Priority 2
(legal-document boundaries). Reads an existing JSON from data/annotated/,
backfills each article with:
  - pdf_page:     1-based index in the PDF file
  - printed_page: the page number printed on the physical BO page
    (extracted from the PDF footer; falls back to pdf_page if unavailable)

Then groups articles into instruments via the reset-to-1 heuristic
(Priority 2), with the preamble disambiguation rule to avoid false splits.

Usage:
    python -m scripts.enrich_json_with_pages data/annotated/fr_BO_7492_Fr_entities.json
    python -m scripts.enrich_json_with_pages data/annotated/ --all
"""

import argparse
import json
import re
import sys
from pathlib import Path

import fitz

# ── Printed-page extraction from PDF footer ──────────────────────────────

# Pattern: a standalone 1-to-4-digit number at the bottom of the raw page
# text.  In BO PDFs the printed page number often appears in the
# bottom-centre margin, and survives as an isolated line at the end of
# `page.get_text("text")`.
_PRINTED_PAGE_RE = re.compile(r"(?:^|\n)\s*(\d{1,4})\s*(?:\n|$)")


def _extract_printed_pages(pdf_path: str) -> dict[int, int]:
    """
    Return a dict mapping 1-based PDF page index → printed page number.

    Strategy: for each page, examine the bottom 15% of the raw text
    (the footer band where BO page numbers are printed) and look for a
    standalone number.  If none is found, fall back to the last
    standalone number anywhere on the page.
    """
    printed = {}
    with fitz.open(pdf_path) as doc:
        for i in range(len(doc)):
            page = doc[i]
            h = page.rect.height
            footer_start = h * 0.85

            raw_text = page.get_text("text")

            # Try footer band first (last ~500 chars ≈ bottom of page)
            footer_portion = raw_text[-500:] if len(raw_text) > 500 else raw_text
            m = _PRINTED_PAGE_RE.search(footer_portion)
            if m:
                printed[i + 1] = int(m.group(1))
                continue

            # Fallback: last standalone number anywhere on the page
            all_matches = list(_PRINTED_PAGE_RE.finditer(raw_text))
            if all_matches:
                printed[i + 1] = int(all_matches[-1].group(1))
            else:
                printed[i + 1] = i + 1  # fallback to PDF page index

    return printed


# ── Article-to-page matching ─────────────────────────────────────────────

def _article_signature(text: str, max_chars: int = 40) -> str:
    """First meaningful characters of an article, used as a fingerprint
    to locate it in the PDF text."""
    sig = re.sub(r"\s+", " ", text).strip()
    return sig[:max_chars]


def _article_signature_ngrams(text: str, n: int = 10) -> set[str]:
    """Overlapping n-gram fingerprint of the article text, used as a
    fuzzy fallback when the exact 40-char substring match fails."""
    flat = re.sub(r"\s+", " ", text).strip().lower()
    return {flat[i:i+n] for i in range(len(flat) - n + 1)} if len(flat) >= n else {flat}


def _backfill_pages(
    articles: list[dict],
    pdf_path: str,
    printed_pages: dict[int, int],
) -> list[dict]:
    """
    Assign `pdf_page` and `printed_page` to each article by matching
    article text against each PDF page's native text.

    Uses a three-tier strategy:
      1. Exact 40-char signature substring (fast, works for most articles).
      2. Overlapping 10-gram scoring (robust against whitespace, hyphenation,
         and column-reordering differences between ``get_text("text")`` and
         the ingestion pipeline's ``get_text("rawdict")`` path).
      3. Short 20-char signature substring (broad fallback).

    Returns articles enriched with page info.
    """
    if not articles:
        return articles

    # Pre-compute per-page text once
    page_texts: list[str] = []
    page_texts_flat: list[str] = []
    with fitz.open(pdf_path) as doc:
        for i in range(len(doc)):
            raw = doc[i].get_text("text")
            page_texts.append(raw)
            page_texts_flat.append(raw.replace("\n", " ").replace("\r", " "))

    for art in articles:
        sig = _article_signature(art.get("text", ""))
        if not sig:
            art["pdf_page"] = None
            art["printed_page"] = None
            continue

        # Tier 1: exact 40-char substring
        best_page = None
        for p_idx, flat in enumerate(page_texts_flat):
            if sig in flat:
                best_page = p_idx + 1
                break

        # Tier 2: n-gram scoring
        if best_page is None and len(sig) >= 30:
            sig_ngrams = _article_signature_ngrams(art.get("text", ""), n=10)
            if sig_ngrams:
                best_score = 0
                for p_idx, flat in enumerate(page_texts_flat):
                    flat_lower = flat.lower()
                    score = sum(1 for g in sig_ngrams if g in flat_lower)
                    if score > best_score:
                        best_score = score
                        best_page = p_idx + 1
                # Require at least 30% of n-grams to match
                threshold = max(3, len(sig_ngrams) * 0.30)
                if best_score < threshold:
                    best_page = None

        # Tier 3: short 20-char fallback
        if best_page is None:
            sig_short = sig[:20]
            for p_idx, flat in enumerate(page_texts_flat):
                if sig_short in flat:
                    best_page = p_idx + 1
                    break

        art["pdf_page"] = best_page
        art["printed_page"] = printed_pages.get(best_page) if best_page else None

    return articles


# ── Priority 2: Instrument-boundary detection ───────────────────────────

# Patterns for instrument preamble (the text that precedes the first
# article of a new legal instrument, confirming a real reset).
# Note: no trailing (?:\n|$) — text flows continuously and we want to
# match even if a space follows the colon before the next line.
_PREAMBLE_END_RE = re.compile(
    r"(?:"
    r"DÉCR[EÈ]TE\s*[:\-–]"
    r"|ARR[EÊ]TE\s*[:\-–]"
    r"|D[ÉE]CIDE\s*[:\-–]"
    r"|ORDONNE\s*[:\-–]"
    r"|DISPOSENT\s*[:\-–]"
    r"|STATUENT\s*[:\-–]"
    r"|SONT\s+ABROG[ÉE]ES?\s*[:\-–]?"
    r")",
    re.IGNORECASE,
)

# Look backwards from a reset to see if the preceding article text
# contains an instrument preamble.
def _preceding_text_has_preamble(article_text: str) -> bool:
    """Check whether *article_text* (the text of the article immediately
    *before* a number reset) contains an instrument preamble that confirms
    the reset is a real boundary."""
    return bool(_PREAMBLE_END_RE.search(article_text))


_CLOSING_ARTICLE_RE = re.compile(
    r"(?:"
    r"charg[ée]\s+(?:de\s+l['\u2019])?ex[ée]cution\s+du\s+pr[ée]sent"
    r"|publi[ée]\s+au\s+Bulletin\s+officiel"
    r"|Fait\s+[àa]\s+Rabat"
    r"|Le\s+(?:Chef\s+du\s+)?(?:Gouvernement|Ministre|Pr[ée]sident)"
    r"|AZIZ\s+AKHANNOUCH|MOHAMMED\s+[A-Z]+"
    r"|Notifi[ée]\s+aux\s+int[ée]ress[ée]s"
    r")",
    re.IGNORECASE,
)


def _preceding_text_is_closing_article(article_text: str) -> bool:
    """Check whether *article_text* looks like the final/closing article
    of an instrument (execution clause, publication order, signature)."""
    return bool(_CLOSING_ARTICLE_RE.search(article_text))


def _classify_instrument_type(articles: list[dict], preamble_context: str = "") -> str:
    """
    Guess the instrument type from the preamble context + first article
    text.  The preamble for instrument 0 comes from the document's
    *preamble_text* (text before the first article marker); for subsequent
    instruments it's in the preceding article's text.

    Strategy: The instrument's own heading (TYPE + n°) appears BEFORE any
    "Vu" clause.  We find the last TYPE + n° match that is NOT preceded by
    "vu" on the same line, or we just restrict to text before the first
    "Vu" clause.
    """
    preamble_upper = preamble_context.upper()

    # Find all TYPE + n° matches
    matches = list(re.finditer(
        r"(DÉCRET|ARR[EÊ]T[EÉ](?:\s+CONJOINT)?|DAHIR|CIRCULAIRE|D[ÉE]CISION)"
        r"\s+N[°o]\s*\d",
        preamble_upper,
    ))
    if matches:
        # Find the first "Vu" clause — cited references that follow
        # "Vu" are NOT the instrument's own heading.
        first_vu = preamble_upper.find("\nVU ")
        if first_vu == -1:
            first_vu = preamble_upper.find("VU LE")
        if first_vu == -1:
            first_vu = preamble_upper.find("VU LA")

        # Priority 1: take the last match BEFORE the first Vu clause
        # (this is the instrument's own heading line).
        best_match = None
        best_pos = -1
        for m in matches:
            pos = m.start()
            if first_vu == -1 or pos < first_vu:
                if pos > best_pos:
                    best_pos = pos
                    best_match = m

        if best_match:
            match = best_match.group(1).replace("É", "E").replace("Ê", "E").replace("È", "E")
            if "CONJOINT" in match:
                return "ARRETE_CONJOINT"
            if "ARRETE" in match:
                return "ARRETE"
            if "DECRET" in match:
                return "DECRET"
            if "DAHIR" in match:
                return "DAHIR"
            if "CIRCULAIRE" in match:
                return "CIRCULAIRE"
            if "DECISION" in match:
                return "DECISION"

        # Priority 2: no heading before Vu — infer type from the last
        # enactment keyword (DÉCRÈTE / ARRÊTE / ...) in the preamble.
        # This handles subsequent instruments where the preamble is
        # embedded in the preceding article without a TYPE+n° heading.
        enactment_map = {
            "DÉCRÈTE": "DECRET",
            "ARRÊTE": "ARRETE",
            "ARRETE": "ARRETE",
            "DÉCIDE": "DECISION",
            "DECIDE": "DECISION",
            "ORDONNE": "DECRET",
            "STATUENT": "DECRET",
            "DISPOSENT": "DECRET",
        }
        for kw, typ in sorted(enactment_map.items(), key=lambda x: -len(x[0])):
            if preamble_upper.rfind(kw) > 0:
                return typ

    # Fallback: search the full combined text (preamble + first article)
    first_text = preamble_context
    for a in articles:
        first_text += " " + a.get("text", "")
        break
    text_upper = first_text.upper()
    if re.search(r"ARR[EÊ]T[EÉ]\s+CONJOINT", text_upper):
        return "ARRETE_CONJOINT"
    if "ARRETE" in text_upper or "ARRÊTÉ" in text_upper:
        return "ARRETE"
    if "DECRET" in text_upper:
        return "DECRET"
    if "DAHIR" in text_upper:
        return "DAHIR"
    if "CIRCULAIRE" in text_upper:
        return "CIRCULAIRE"
    if "DECISION" in text_upper or "DÉCISION" in text_upper:
        return "DECISION"
    return "DECRET"  # fallback


def _extract_reference(text: str, instr_type: str) -> str | None:
    """
    Extract the instrument's own reference number from the text.

    Strategy: The instrument's own number follows the type keyword
    (Décret/Arrêté/Dahir) BEFORE any "Vu" clause.  We locate the
    TYPE keyword, take the text up to the first "Vu" clause (if any),
    and return the first reference number found there (2-part or 3-part).
    """
    type_pat = re.compile(
        r"(?:D[ée]cret|Arr[êe]t[ée](?:\s+conjoint)?|"
        r"Dahir|Circulaire|D[ée]cision)",
        re.IGNORECASE,
    )
    type_match = type_pat.search(text)
    if not type_match:
        return None

    # Text from TYPE keyword up to first "Vu" clause (the instrument's own header)
    type_pos = type_match.start()
    vu_pos = text.upper().find("VU", type_pos)
    if vu_pos < 0:
        own_text = text[type_pos:]
    else:
        own_text = text[type_pos:vu_pos]

    # Try 3-part number first (most common for moroccan legal refs)
    m = re.search(r"(?:n[°oº]\s*)?(\d{1,4}[-–.]\d{2}[-–.]\d{2,4})", own_text)
    if m and not _ref_is_false_positive(m.group(1)):
        return m.group(1)

    # Then try 2-part number (e.g. "Arrêté n° 168-26")
    m = re.search(r"(?:n[°oº]\s*)?(\d{1,4}[-–.]\d{2,4})", own_text)
    if m and not _ref_is_false_positive(m.group(1)):
        return m.group(1)

    return None


def _ref_is_false_positive(ref: str) -> bool:
    """Return True if *ref* is a monetary amount, date, or page range."""
    # Monetary amount like "95.000" (the leading part of 95.000.000,00)
    if re.match(r"^\d{1,4}\.000", ref):
        return True
    # Date like "6-10-2022" or "01-01-2022"
    parts = re.split(r"[-–.]", ref)
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        first, second, third = int(parts[0]), int(parts[1]), int(parts[2])
        if first > 31 and second <= 12 and third <= 31:  # YYYY-MM-DD
            return True
        if first <= 31 and second <= 12 and third >= 1000:  # DD-MM-YYYY
            return True
    # Page range like "120-122" or partial date like "6-10"
    if re.match(r"^\d{1,4}[-–.]\d{1,4}$", ref):
        parts = ref.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            a, b = int(parts[0]), int(parts[1])
            if 0 < a < 10000 and 0 < b < 10000 and abs(a - b) < 100:
                return True
    return False


def _mergeable_refs(prev: dict, curr: dict) -> bool:
    """Return True if *curr* should be merged into *prev* as a sub-instrument.

    Merge scenarios:
    1. Same reference (DECRET→ARRETE or two ARRETE types).
    2. DECRET/DAHIR followed immediately by an implementing ARRETE.
    3. Current instrument has no preamble (lowercase start) — continuation.
    4. Both are consecutive ARRETE types and prev has a reference.
    """
    prev_ref = prev.get("reference")
    curr_ref = curr.get("reference")
    prev_type = prev.get("instrument_type", "")
    curr_type = curr.get("instrument_type", "")

    # Same reference
    if prev_ref and prev_ref == curr_ref:
        if prev_type in ("DECRET", "DAHIR"):
            return "ARRETE" in curr_type
        if "ARRETE" in prev_type and "ARRETE" in curr_type:
            return True
        return False

    # Current instrument preamble starts lowercase — it's a continuation
    curr_preamble = (curr.get("_preamble", "") or "").lstrip()
    if curr_preamble and curr_preamble[0].islower():
        if "ARRETE" in prev_type and "ARRETE" in curr_type:
            return True
        # Also merge DECRET/DAHIR → continuation arrêté
        if prev_type in ("DECRET", "DAHIR") and "ARRETE" in curr_type:
            return True

    # No ref on curr but prev has one and both are arrêtés
    if curr_ref is None and prev_ref is not None:
        if "ARRETE" in prev_type and "ARRETE" in curr_type:
            return True
    return False


def _group_into_instruments(
    articles: list[dict],
    preamble_text: str = "",
    decrees: list[dict] | None = None,
) -> list[dict]:
    """
    Post-process the flat article list into instruments.

    Two strategies in order of preference:

    1. **From pre-computed decrees** (preferred).  If *decrees* is given
       (the output of ``get_per_decree_preamble_map``, stored in the JSON
       as ``data["decrees"]``), each decree group with at least one article
       becomes one instrument.  Decree groups with zero articles (e.g. a
       LOI-CADRE that has no numbered articles) are still added as
       article-less instruments so they appear in the output.

    2. **Fallback heuristic (reset-to-1/PREMIER)**.  A reset is only a real
       boundary if the preceding article's text contains an instrument
       preamble (Vu ..., Considérant ..., DÉCRÈTE :, etc.) or a
       closing/execution clause.

    *preamble_text* is the text before the first article marker in the
    document (from segmenter.get_preamble).  It provides the enactment
    context for the very first instrument.

    Instruments store *article_indices* (indices into the flat *articles*
    array) rather than full article copies, avoiding structural duplication
    in the output JSON.
    """
    if not articles:
        return []

    # Strategy 1: use pre-computed decree boundaries
    if decrees:
        instruments = []
        for i, dec in enumerate(decrees):
            start_idx = dec.get("first_article_idx", 0)
            if i + 1 < len(decrees):
                end_idx = decrees[i + 1]["first_article_idx"]
            else:
                end_idx = len(articles)

            if start_idx >= len(articles):
                # Article-less decree (e.g. LOI-CADRE) — virtual instrument
                preamble = dec.get("preamble", "")
                instr_type = _classify_instrument_type([], preamble)
                ref = _extract_reference(preamble, instr_type)
                instruments.append({
                    "instrument_type": instr_type,
                    "reference": ref,
                    "article_indices": [],
                    "n_articles": 0,
                    "_preamble": preamble,
                })
                continue

            article_slice = articles[start_idx:end_idx]
            preamble_context = dec.get("preamble", "")
            instr_type = _classify_instrument_type(article_slice, preamble_context)
            # Extract reference from preamble context ONLY, not from the
            # first article text which may contain cross-references to
            # other instruments (e.g. "Vu le décret n° 2-15-743...")
            ref = _extract_reference(preamble_context, instr_type)
            instruments.append({
                "instrument_type": instr_type,
                "reference": ref,
                "article_indices": list(range(start_idx, end_idx)),
                "n_articles": end_idx - start_idx,
                "_preamble": preamble_context,
            })

        # Merge consecutive instruments that share the same reference
        # (DECRET + ARRETE under the same number, or multiple arrêtés)
        merged = []
        for instr in instruments:
            if merged and _mergeable_refs(merged[-1], instr):
                prev = merged[-1]
                prev["article_indices"].extend(instr["article_indices"])
                prev["article_indices"].sort()
                prev["n_articles"] = len(prev["article_indices"])
                # When merging a leading DECRET/DAHIR with an implementing
                # ARRETE, keep the DECRET/DAHIR type as the umbrella type.
                # When both are ARRETE variants, use the more specific one.
            else:
                merged.append(instr)
        instruments = merged

        for idx, instr in enumerate(instruments, 1):
            instr["instrument_id"] = f"instr_{idx}"
            instr.pop("_preamble", None)  # internal field, not for output
        return instruments

    # Strategy 2: fallback heuristic (reset-to-1 / PREMIER)
    instruments = []
    current_start = 0
    prev_preamble_context = preamble_text

    for i in range(1, len(articles)):
        prev_art = articles[i - 1]
        art = articles[i]
        num = art.get("number", "").strip().upper()

        is_reset = (num in ("1", "PREMIER", "1ER", "UNIQUE", "PREMIÈRE") or
                    num.startswith("PREMIER") or num == "1ER")

        if is_reset:
            prev_text = prev_art.get("text", "")
            prev_preamble = prev_art.get("preamble", "")
            combined = (prev_preamble + "\n" + prev_text).strip()
            if _preceding_text_has_preamble(combined) or _preceding_text_is_closing_article(prev_text):
                # Real boundary — finalise current instrument
                instruments.append(
                    _make_instrument(articles, current_start, i, prev_preamble_context)
                )
                current_start = i
                prev_preamble_context = combined

    if current_start < len(articles):
        instruments.append(
            _make_instrument(articles, current_start, len(articles), prev_preamble_context)
        )

    for idx, instr in enumerate(instruments, 1):
        instr["instrument_id"] = f"instr_{idx}"

    return instruments


def _make_instrument(
    all_articles: list[dict],
    start_idx: int,
    end_idx: int,
    preamble_context: str = "",
) -> dict:
    """Wrap a range of articles into an instrument dict.

    Stores *article_indices* (a list of indexes into *all_articles*)
    instead of full article objects, to avoid structural duplication
    in the output JSON.

    *preamble_context* is the text that precedes this instrument (either
    the document's preamble_text for instrument 0, or the preceding
    instrument's last article text for subsequent instruments).  It is
    used for classifying instrument_type and extracting the reference.
    """
    instrument_articles = all_articles[start_idx:end_idx]
    # Use the first article's own preamble field if available; it is more
    # accurate than the previous instrument's article text (which may be
    # a closing clause rather than a true preamble).
    first_art = instrument_articles[0]
    art_preamble = first_art.get("preamble", "") or ""
    ctx = art_preamble if len(art_preamble) > 50 else preamble_context
    first_text = ctx + " " + first_art.get("text", "")
    instr_type = _classify_instrument_type(instrument_articles, ctx)
    ref = _extract_reference(first_text, instr_type)

    return {
        "instrument_type": instr_type,
        "reference": ref,
        "article_indices": list(range(start_idx, end_idx)),
        "n_articles": end_idx - start_idx,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def enrich_json(
    json_path: Path,
    pdf_dir: Path | None = None,
    backfill_pages: bool = True,
    detect_instruments: bool = True,
) -> Path:
    """
    Load a JSON file, enrich it with page numbers and/or instrument
    boundaries, and save back to the same path (overwrite).

    The original PDF is located by replacing `data/annotated/` with
    `data/raw/` in the JSON path and adjusting the filename stem.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Locate the original PDF from the source path or json stem
    # source looks like: data/processed/fr/BO_7492_Fr.txt
    stem = Path(data.get("source", json_path.stem)).stem
    pdf_base = pdf_dir or Path("data/raw")

    # Try candidates in order of likelihood
    candidates = [
        pdf_base / f"{stem}.pdf",               # BO_7492_Fr.pdf
        pdf_base / f"{stem.replace('_Fr', '')}.pdf",   # BO_7492.pdf
        pdf_base / f"{stem.replace('_Ar', '')}.pdf",   # BO_7492.pdf (ar)
        pdf_base / f"{stem.replace('_entities', '')}.pdf",
    ]
    # Also search subdirectories (ar/, fr/)
    for sub in ("ar", "fr", ""):
        candidates.append(pdf_base / sub / f"{stem}.pdf")
        stripped = stem.replace("_Fr", "").replace("_Ar", "")
        if stripped != stem:
            candidates.append(pdf_base / sub / f"{stripped}.pdf")

    pdf_path = None
    for c in candidates:
        if c.exists():
            pdf_path = str(c)
            break

    if pdf_path and backfill_pages:
        print(f"  Backfilling page numbers from {pdf_path} ...")
        printed_pages = _extract_printed_pages(pdf_path)
        articles = data.get("articles", [])
        articles = _backfill_pages(articles, pdf_path, printed_pages)
        data["articles"] = articles
        data["total_pdf_pages"] = len(printed_pages)
        print(f"    {sum(1 for a in articles if a.get('pdf_page'))}/{len(articles)} articles mapped to pages")

    if detect_instruments and data.get("articles"):
        print(f"  Detecting instrument boundaries ({len(data['articles'])} articles) ...")
        instruments = _group_into_instruments(
            data["articles"],
            preamble_text=data.get("preamble_text", ""),
            decrees=data.get("decrees"),
        )
        data["instruments"] = instruments
        data["n_instruments"] = len(instruments)
        print(f"    {len(instruments)} instruments detected")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  -> {json_path}")
    return json_path


def _table_text_overlap(article_text: str, table: dict, min_overlap_chars: int = 20) -> bool:
    """
    Check whether the article text contains enough of the table cell text
    to consider the table "belonging" to this article.

    This avoids attaching a table on the same PDF page to articles that
    appear before or after the table on the same page (common when a page
    has an article ending, then a table, then another article).

    Args:
        article_text: The cleaned text of the article.
        table: A serialised table dict with a ``rows`` key.
        min_overlap_chars: Minimum total characters of cell text that
            must appear in the article text.

    Returns:
        True if the table overlaps with the article text.
    """
    # Concatenate all cell text into one searchable string
    all_cells = " ".join(
        " ".join(row) for row in table.get("rows", [])
    )
    if not all_cells.strip():
        return True  # empty table — can't decide, assume yes

    # Count how many characters of cell text appear in the article
    overlap = 0
    for ch in all_cells:
        if ch in article_text:
            overlap += 1

    return overlap >= min_overlap_chars


def _match_tables_by_content(
    table_keys: dict, articles: list[dict], min_ratio: float = 0.30
) -> dict[int, list[dict]]:
    """
    Fallback: link tables to articles by text-content overlap when ``pdf_page``
    is unavailable.  Assigns each table to the article with the highest ratio
    of cell-text characters present in the article text.
    """
    result: dict[int, list[dict]] = {}
    for key, table in table_keys.items():
        all_cells = " ".join(" ".join(row) for row in table.get("rows", []))
        if not all_cells.strip():
            continue
        best_idx = -1
        best_ratio = 0.0
        for idx, art in enumerate(articles):
            art_text = art.get("text", "")
            if not art_text:
                continue
            overlap = sum(1 for ch in all_cells if ch in art_text)
            ratio = overlap / max(len(all_cells), 1)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
        if best_ratio >= min_ratio:
            result.setdefault(best_idx, []).append(table)
    return result


def _table_has_content(table: dict) -> bool:
    """A table is contentful if at least one cell contains non-whitespace text.
    Tables whose cells are entirely empty (zero-row tables, OCR ghosts) are
    dropped to avoid inflating the output with noise."""
    return any(
        cell.strip()
        for row in table.get("rows", [])
        for cell in row
    )


def _deduplicate_tables(tables: list[dict]) -> list[dict]:
    """
    Remove exact duplicate table entries from a list.

    Two tables are considered equal if they have the same ``page_number``,
    ``bbox``, and ``rows``.  This is stricter than Python list equality
    (which requires identical objects) and avoids inflating JSON size
    when the same table is attached to multiple articles across pages.
    """
    seen = set()
    deduped = []
    for t in tables:
        rows_tuple = tuple(tuple(r) for r in t.get("rows", []))
        key = (t.get("page_number"), tuple(t.get("bbox", [])), rows_tuple)
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped


def enrich_json_with_tables(
    json_path: Path,
    pdf_dir: Path | None = None,
) -> Path:
    """
    Post-process an enriched JSON to add table references to each article.

    Uses pdfplumber to extract tables from the original PDF, then matches
    them to articles by:
      1. Page number (exact match).
      2. Text overlap (the article text must contain enough cell text).

    Adds an ``extracted_tables`` list to each article that has tables on
    its page AND whose text overlaps with the table content.

    Also adds a document-level ``deduplicated_tables`` index (tables that
    appear across multiple pages/positions in the document) and per-instrument
    ``extracted_tables`` with duplicates removed.
    """
    from src.ingestion.table_extractor import (
        extract_tables_from_pdf,
        get_table_bboxes_by_page,
    )

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    stem = Path(data.get("source", json_path.stem)).stem
    pdf_base = pdf_dir or Path("data/raw")
    candidates = [pdf_base / f"{stem}.pdf", pdf_base / f"{stem.replace('_Fr', '')}.pdf"]
    for sub in ("ar", "fr", ""):
        candidates.append(pdf_base / sub / f"{stem}.pdf")
        candidates.append(pdf_base / sub / f"{stem.replace('_Fr', '').replace('_Ar', '')}.pdf")
    pdf_path = None
    for c in candidates:
        if c.exists():
            pdf_path = str(c)
            break
    if not pdf_path:
        print(f"  PDF introuvable pour {json_path.name}, extraction des tableaux ignorée")
        return json_path

    result = extract_tables_from_pdf(pdf_path)
    bboxes_by_page = get_table_bboxes_by_page(result)
    tables_by_page: dict[int, list] = {}
    for t in result.tables:
        if not _table_has_content({"rows": t.rows}):
            continue
        page = t.page_number
        tables_by_page.setdefault(page, []).append({
            "page_number": page,
            "bbox": list(t.bbox),
            "n_rows": t.n_rows,
            "n_cols": t.n_cols,
            "rows": t.rows,
        })

    # Clear any table data from previous runs so re-runs are idempotent
    for art in data.get("articles", []):
        art.pop("extracted_tables", None)
    for instr in data.get("instruments", []):
        instr.pop("extracted_tables", None)
        for art in instr.get("articles", []):
            art.pop("extracted_tables", None)
    data.pop("deduplicated_tables", None)
    data.pop("unlinked_tables", None)
    data.pop("n_unlinked_tables", None)
    data.pop("n_total_tables", None)

    n_total = len(result.tables)
    n_linked = 0
    n_unlinked = 0
    unlinked_tables = []

    # Build a set of all table IDs for tracking which get linked
    all_table_keys = {}
    for t in result.tables:
        if not _table_has_content({"rows": t.rows}):
            continue
        key = (t.page_number, tuple(t.bbox), tuple(tuple(r) for r in t.rows))
        all_table_keys[key] = {"page_number": t.page_number, "bbox": list(t.bbox),
                               "n_rows": t.n_rows, "n_cols": t.n_cols, "rows": t.rows}

    for art in data.get("articles", []):
        p = art.get("pdf_page")
        art_text = art.get("text", "")
        if p and p in tables_by_page:
            matched_tables = []
            for t_obj in tables_by_page[p]:
                if _table_text_overlap(art_text, t_obj):
                    matched_tables.append(t_obj)
            if matched_tables:
                art["extracted_tables"] = matched_tables
                n_linked += 1
                # Mark these tables as linked
                for t in matched_tables:
                    key = (t.get("page_number"), tuple(t.get("bbox", [])),
                           tuple(tuple(r) for r in t.get("rows", [])))
                    all_table_keys.pop(key, None)

    # Fallback: content-based matching for tables that couldn't be linked by page
    all_articles = list(data.get("articles", []))
    if all_table_keys:
        content_matched = _match_tables_by_content(all_table_keys, all_articles)
        for art_idx, tables in content_matched.items():
            art = all_articles[art_idx]
            art["extracted_tables"] = art.get("extracted_tables", []) + tables
            n_linked += 1
            for t in tables:
                key = (t.get("page_number"), tuple(t.get("bbox", [])),
                       tuple(tuple(r) for r in t.get("rows", [])))
                all_table_keys.pop(key, None)

    # Tables that could not be linked by either method → store at document level
    unlinked_tables = list(all_table_keys.values())
    n_unlinked = len(unlinked_tables)
    data["unlinked_tables"] = unlinked_tables
    data["n_unlinked_tables"] = n_unlinked

    # Build per-instrument table index (deduplicated).
    # Instruments store article_indices referencing the flat articles array.
    all_data_articles = data.get("articles", [])
    for instr in data.get("instruments", []):
        instr_tables = []
        for idx in instr.get("article_indices", []):
            if idx < len(all_data_articles):
                for t in all_data_articles[idx].get("extracted_tables", []):
                    if t not in instr_tables:
                        instr_tables.append(t)
        if instr_tables:
            instr["extracted_tables"] = instr_tables

    # Build document-level deduplicated table index
    all_linked_tables = []
    for art in data.get("articles", []):
        all_linked_tables.extend(art.get("extracted_tables", []))
    deduped_doc_tables = _deduplicate_tables(all_linked_tables) if all_linked_tables else []
    if deduped_doc_tables:
        data["deduplicated_tables"] = deduped_doc_tables

    data["n_total_tables"] = n_total

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"    {n_total} table(s) extracted from PDF, {n_linked} linked to articles, "
          f"{n_unlinked} unlinked (stored in 'unlinked_tables')")
    print(f"    {len(deduped_doc_tables)} unique table(s) after deduplication")
    print(f"  -> {json_path}")
    return json_path


def process_all(
    path: Path,
    pdf_dir: Path | None = None,
    backfill_pages: bool = True,
    detect_instruments: bool = True,
    extract_tables: bool = False,
):
    """Process a single file or all JSON files in a directory."""
    fn = enrich_json_with_tables if extract_tables else enrich_json
    if path.is_dir():
        for f in sorted(path.glob("*_entities.json")):
            fn(f, pdf_dir=pdf_dir) if extract_tables else enrich_json(
                f, pdf_dir=pdf_dir,
                backfill_pages=backfill_pages,
                detect_instruments=detect_instruments,
            )
    else:
        fn(path, pdf_dir=pdf_dir) if extract_tables else enrich_json(
            path, pdf_dir=pdf_dir,
            backfill_pages=backfill_pages,
            detect_instruments=detect_instruments,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Enrich JSON with page references, instrument boundaries, and tables."
    )
    parser.add_argument("path", type=str, help="JSON file or directory")
    parser.add_argument("--all", action="store_true",
                        help="Process all JSON files in directory")
    parser.add_argument("--pdf-dir", type=str, default=None,
                        help="Directory containing original PDFs (default: data/raw/)")
    parser.add_argument("--skip-pages", action="store_true",
                        help="Skip page-number backfill")
    parser.add_argument("--skip-instruments", action="store_true",
                        help="Skip instrument detection")
    parser.add_argument("--tables", action="store_true",
                        help="Extract and link tables to articles (Priority 3)")
    args = parser.parse_args()

    path = Path(args.path)
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None

    if args.tables:
        process_all(path, pdf_dir=pdf_dir, extract_tables=True)
    elif args.all and path.is_dir():
        process_all(path, pdf_dir=pdf_dir,
                    backfill_pages=not args.skip_pages,
                    detect_instruments=not args.skip_instruments)
    else:
        enrich_json(path, pdf_dir=pdf_dir,
                    backfill_pages=not args.skip_pages,
                    detect_instruments=not args.skip_instruments)


if __name__ == "__main__":
    main()
