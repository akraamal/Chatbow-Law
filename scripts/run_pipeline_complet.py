"""
run_pipeline_complet.py
------------------------
Pipeline complet : ingestion (PDF → texte) + prétraitement (nettoyage)
+ extraction NLP (entités → JSON). S'applique sur un fichier unique,
un dossier, ou tout data/raw/ par défaut.

Usage :
    python -m scripts.run_pipeline_complet
    python -m scripts.run_pipeline_complet --file chemin/vers/document.pdf
    python -m scripts.run_pipeline_complet --dir chemin/vers/dossier
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import (
    run_ingestion_pipeline,
    stamp_interim_provenance,
    ensure_interim_fresh,
)

from src.preprocessing.cleaner_fr import clean_french_text
from src.preprocessing.cleaner_ar import clean_arabic_text
from src.preprocessing.arabic_utils import arabic_char_ratio

from src.extraction.etape4_pipeline import enrich_article_json, enrich_articles_batch
from src.extraction.citation_resolver import resolve_citations, LEGAL_TEXT_LABELS
from src.extraction.article_citation_patterns import find_article_citations
from src.extraction.entity_span_utils import normalize_entity
from src.extraction.document_metadata_extractor import extract_document_metadata
from src.extraction.ner_filter import filter_entities, align_entity_text
from src.extraction.ocr_corrector import correct_ocr
from src.preprocessing.segmenter import (
    segment_into_articles, get_preamble, get_sommaire, get_per_decree_preamble_map,
)
from src.export.article_to_markdown import build_full_markdown
from src.extraction.entity_ruler_builder_fr import build_fr_nlp, extract_legal_entities_fr
from src.extraction.entity_ruler_builder_ar import build_ar_nlp, extract_legal_entities_ar

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
ANNOTATED_DIR = Path("data/annotated")
ANNOTATED_MD_DIR = Path("data/annotated-MD")

# Cache NLP au niveau module (évite de reconstruire les modèles pour chaque PDF)
_NLP_CACHE = {}


# ============================================================================
# Étape 1 : Ingestion (PDF → interim/*.txt)
# ============================================================================

def _save_ingestion_result(result, pdf_path: Path) -> list[Path]:
    """Sauvegarde le texte extrait dans data/interim/{fr,ar}/ et
    retourne la liste des fichiers .txt créés."""
    saved = []
    stem = pdf_path.stem
    is_bilingual = result.detected_layout == "colonnes"

    if is_bilingual:
        if result.text_fr.strip():
            p = INTERIM_DIR / "fr" / f"{stem}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(result.text_fr, encoding="utf-8")
            stamp_interim_provenance(p, pdf_path)
            saved.append(p)
        if result.text_ar.strip():
            p = INTERIM_DIR / "ar" / f"{stem}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(result.text_ar, encoding="utf-8")
            stamp_interim_provenance(p, pdf_path)
            saved.append(p)
    else:
        len_fr = len(result.text_fr.strip())
        len_ar = len(result.text_ar.strip())
        if len_fr >= len_ar and len_fr > 0:
            p = INTERIM_DIR / "fr" / f"{stem}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(result.text_fr, encoding="utf-8")
            stamp_interim_provenance(p, pdf_path)
            saved.append(p)
        elif len_ar > 0:
            p = INTERIM_DIR / "ar" / f"{stem}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(result.text_ar, encoding="utf-8")
            stamp_interim_provenance(p, pdf_path)
            saved.append(p)

    if result.text_unknown.strip():
        p = INTERIM_DIR / f"{stem}_unknown.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(result.text_unknown, encoding="utf-8")
        saved.append(p)

    if result.tables:
        p = INTERIM_DIR / "tables" / f"{stem}_tables.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result.tables, ensure_ascii=False, indent=2), encoding="utf-8")

    return saved


def run_ingestion(pdf_path: Path) -> list[Path]:
    """Ingère un PDF et retourne la liste des fichiers texte créés dans interim/."""
    print(f"\n{'='*60}")
    print(f"ÉTAPE 1 — Ingestion : {pdf_path.name}")
    print(f"{'='*60}")
    result = run_ingestion_pipeline(str(pdf_path))
    files = _save_ingestion_result(result, pdf_path)
    for f in files:
        print(f"  → {f}")
    for w in result.warnings:
        print(f"  ⚠ {w}")
    return files


# ============================================================================
# Étape 2 : Prétraitement (interim/*.txt → processed/*.txt)
# ============================================================================

def _detect_language(text: str) -> str:
    return "ar" if arabic_char_ratio(text) > 0.30 else "fr"


def run_preprocessing(interim_file: Path, arabic_runs: list | None = None) -> Path | None:
    """Nettoie un fichier interim/ et retourne le chemin dans processed/.

    *arabic_runs* (optionnel) : collecte les tronçons arabes cités dans le
    texte français (titres d'émissions, clauses reprises verbatim) au lieu
    de les perdre — exposés ensuite dans le JSON (possible_embedded_arabic).
    """
    parts = list(interim_file.parts)
    parts = ["processed" if p == "interim" else p for p in parts]
    out = Path(*parts)

    print(f"\n  ÉTAPE 2 — Nettoyage : {interim_file.name}")
    raw = interim_file.read_text(encoding="utf-8")
    lang = _detect_language(raw)
    if lang == "ar":
        cleaned = clean_arabic_text(raw)
    else:
        cleaned = clean_french_text(raw, arabic_runs=arabic_runs)

    # Post-OCR correction (French only — corrects common character
    # transpositions and accent-loss patterns from PDF extraction)
    if lang == "fr":
        corrected = correct_ocr(cleaned)
        if cleaned != corrected:
            print(f"    OCR corrections appliquées")
        cleaned = corrected

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cleaned, encoding="utf-8")
    print(f"    → {out}")
    return out


# ============================================================================
# Étape 3 : Extraction NLP (processed/*.txt → annotated/*_entities.json)
# ============================================================================

def _entities_to_dicts(doc) -> list:
    return [
        {"label": ent.label_, "text": ent.text,
         "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]


def _process_article_entities(art, extract_fn, nlp):
    """Extrait les entités régulières d'un article via l'EntityRuler + regex
    (étape rapide, faite article par article). Retourne le dict article de base."""
    doc = extract_fn(art.text, nlp=nlp)
    return {
        "number": art.number, "raw_header": art.raw_header,
        "entities": filter_entities(_entities_to_dicts(doc)), "dates": [],
        "preamble": getattr(art, "preamble", "") or "",
    }


def _extract_bo_metadata_from_interim(pdf_stem: str, interim_files: list[Path]) -> dict:
    """Extract BO metadata by trying all interim files (fr, ar, unknown) since the
    header page can be misrouted by layout detection (e.g. first cover page routed
    to 'fr' while the PDF is labeled _Ar)."""
    # Strip control chars that PDF extraction sometimes inserts (backspace 0x08
    # between a word and a number, etc.) before passing to metadata extractor
    _CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    for f in interim_files:
        if f.parent.name not in ("fr", "ar", ""):
            continue
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        text = _CTRL.sub("", text)
        # Pass de langue dans l'ordre du dossier d'abord : sur un texte
        # français, le pass "ar" rend un bo_number issu du SEUL nom de
        # fichier (confidence low, date_publication None) qui court-
        # circuite le pass "fr" (filename+header, confidence high, date
        # de publication renseignée).  L'inverse reste possible quand
        # l'en-tête d'un _Ar est routé vers interim/fr/ — le pass "fr"
        # échoue alors et on retombe sur "ar".
        folder_lang = f.parent.name if f.parent.name in ("fr", "ar") else None
        order = ("fr", "ar") if folder_lang == "fr" else ("ar", "fr")
        for lang in order:
            meta = extract_document_metadata(text, doc_id=pdf_stem, lang=lang)
            if meta.get("bo_number"):
                return meta
    return {}


def _ensure_processed_fresh(processed_file: Path) -> None:
    """Bloque la régénération d'un JSON depuis un texte intermédiaire stale.

    Le fichier interim/ correspondant (même chemin relatif, dossier
    « interim » au lieu de « processed ») doit avoir été extrait par
    l'extracteur actuel et depuis le PDF actuel — sinon on régénérerait
    silencieusement des JSON sur du texte obsolète (cas BO_7510 : interim
    antérieur au correctif de lecture colonne par colonne)."""
    parts = list(processed_file.parts)
    interim = Path(*["interim" if p == "processed" else p for p in parts])
    if interim.exists():
        ensure_interim_fresh(interim)


def run_extraction(processed_file: Path, lang: str, nlp_fr=None, nlp_ar=None,
                   metadata_override: dict | None = None,
                   arabic_runs: list | None = None) -> Path | None:
    """Extrait les entités d'un fichier processed/ et sauvegarde le JSON.

    Utilise enrich_articles_batch() pour exécuter le NER statistique en
    une seule passe pour tous les articles, et saute le NER pour les
    articles sans entité juridique."""
    out_path = ANNOTATED_DIR / f"{processed_file.parent.name}_{processed_file.stem}_entities.json"

    print(f"\n  ÉTAPE 3 — Extraction NLP : {processed_file.name} ({lang})")
    _ensure_processed_fresh(processed_file)
    text = processed_file.read_text(encoding="utf-8")

    extract_fn = extract_legal_entities_fr if lang == "fr" else extract_legal_entities_ar
    nlp = nlp_fr if lang == "fr" else nlp_ar

    preamble = get_preamble(text, lang=lang)
    preamble_doc = extract_fn(preamble, nlp=nlp)
    articles = segment_into_articles(text, lang=lang)

    # Étape rapide : entités regex/EntityRuler par article
    article_dicts = [_process_article_entities(a, extract_fn, nlp) for a in articles]
    article_texts = [a.text for a in articles]

    # Étape batch : NER statistique UNE FOIS pour tous les articles
    articles_out = enrich_articles_batch(
        article_dicts, article_texts,
        doc_id=processed_file.stem, lang=lang,
    )

    metadata = metadata_override if metadata_override else extract_document_metadata(text, doc_id=processed_file.stem, lang=lang)

    # Per-decree preamble extraction (grouped Vu clauses)
    decrees = get_per_decree_preamble_map(text, lang=lang)

    # Helper: fix entity positions by searching the article text
    def _fix_entity_positions(art: dict, art_text: str) -> None:
        """Try to locate entities with start=-1 in *art_text* and update
        their positions.  Entities whose text cannot be found at all are
        removed from the article (they are invisible to downstream users).

        Uses align_entity_text which ALSO restores the exact source slice
        as entity text (clean_entity_text flattens \n → space without
        adjusting offsets, which previously produced text != source[start:end]
        and made source.find(entity.text) return -1)."""
        fixed = []
        for e in art.get("entities", []):
            aligned = align_entity_text(e, art_text)
            if aligned is not None:
                fixed.append(aligned)
        art["entities"] = fixed

    # Post-process: decree context propagation
    # 1) Build an article-to-decree map
    # 2) For each article, propagate the decree's preamble text entities
    #    (ARRETE/DECRET/DAHIR/LOI) into the article's entity list, so
    #    downstream consumers (citation graphs, RAG) see them.
    # 3) Replace DOCUMENT_SOURCE citations with the decree's entity.
    _DECREE_TITLE_RE = re.compile(
        r"(Arr[êe]t[ée]\s+conjoint|Arr[êe]t[ée]|D[ée]cret|Loi|Dahir|D[ée]cision)"
        r"\s+n[°°]\s*[\w\-\.]+",
        re.IGNORECASE,
    )
    article_to_decree = {}
    for i, dec in enumerate(decrees):
        start_idx = dec.get("first_article_idx", 0)
        if i + 1 < len(decrees):
            end_idx = decrees[i + 1]["first_article_idx"]
        else:
            end_idx = len(articles_out)
        for j in range(start_idx, end_idx):
            article_to_decree[j] = dec

    # Extract legal entities from each decree preamble once (full NER —
    # arrêtés/décrets/lois/dahirs, dates, ministères).  Crucial pour les
    # Décisions (ex. CSN Radio Mars) dont le préambule est la quasi-totalité
    # du contenu et qui n'ont pas de numérotation "Article" : sans cela,
    # ces instruments n'ont AUCUNE entité, structurellement.  Les entités
    # sont stockées sur le décret (decrees[i]["entities"]) et propagées
    # aux articles du décret plus bas.
    decree_entities_cache = {}
    for i, dec in enumerate(decrees):
        dec_preamble = dec.get("preamble", "")
        if not dec_preamble:
            dec["entities"] = []
            continue
        doc = extract_fn(dec_preamble, nlp=nlp)
        ents = filter_entities(_entities_to_dicts(doc))
        dec["entities"] = ents
        decree_entities_cache[id(dec)] = ents

    # Also extract document-level preamble entities for cross-decree references
    doc_preamble_entities = filter_entities(_entities_to_dicts(preamble_doc))
    legal_labels_doc = {"LOI", "DECRET", "DAHIR", "ARRETE", "BULLETIN_OFFICIEL"}
    doc_legal_entities = [
        e for e in doc_preamble_entities
        if e.get("label") in legal_labels_doc and len(e.get("text", "")) > 5
    ]

    for idx, art in enumerate(articles_out):
        dec = article_to_decree.get(idx)
        if not dec:
            continue

        art_text = art.get("text", "")

        # Propagate preamble entities into article (avoid duplicates).
        # Reset preamble-relative positions to -1 (sentinel) since they
        # refer to the wrong text span.  The citation re-resolution path
        # below re-finds entity text within the article on its own.
        preamble_ents = decree_entities_cache.get(id(dec), [])
        existing_texts = {e.get("text", "").lower() for e in art.get("entities", [])}
        for pe in preamble_ents:
            if pe["text"].lower() not in existing_texts:
                entry = dict(pe)
                entry["start"] = -1
                entry["end"] = -1
                art["entities"].append(entry)
                existing_texts.add(pe["text"].lower())

        # Replace DOCUMENT_SOURCE citations with decree entity
        dec_preamble = dec.get("preamble", "")
        if dec_preamble:
            ent_match = _DECREE_TITLE_RE.search(dec_preamble)
            if ent_match:
                dec_entity_text = ent_match.group(0).strip()
                raw_label = ent_match.group(1)
                if "ARRÊTÉ" in raw_label.upper() or "ARRETE" in raw_label.upper():
                    dec_label = "ARRETE"
                elif "DÉCRET" in raw_label.upper() or "DECRET" in raw_label.upper():
                    dec_label = "DECRET"
                elif "LOI" in raw_label.upper():
                    dec_label = "LOI"
                elif "DAHIR" in raw_label.upper():
                    dec_label = "DAHIR"
                elif "DÉCISION" in raw_label.upper() or "DECISION" in raw_label.upper():
                    dec_label = "DECISION"
                else:
                    dec_label = "ARRETE"

                for cit in art.get("citations", []):
                    if cit.get("target_label") == "DOCUMENT_SOURCE":
                        cit["target_label"] = dec_label
                        cit["target_text"] = dec_entity_text
                        cit["resolved"] = True

    # Re-resolve any remaining DOCUMENT_SOURCE citations with enriched entities.
    # Re-extract citations directly from article text (they may have been
    # captured differently in the initial pass) and resolve with the now-
    # enriched entity list.  Propagated entities have preamble-relative
    # start/end positions, so we search for their text in the article to
    # obtain correct article-relative positions.
    import re as _re
    legal_text_labels = {l.upper() for l in LEGAL_TEXT_LABELS}
    for art in articles_out:
        art_text = art.get("text", "")
        if not art_text:
            continue
        has_ds = any(c.get("target_label") == "DOCUMENT_SOURCE" for c in art.get("citations", []))
        if not has_ds:
            # Even without DOCUMENT_SOURCE citations, try to fix -1 positions
            _fix_entity_positions(art, art_text)
            continue

        raw_citations = [
            {"text": t, "start": s, "end": e}
            for s, e, t in find_article_citations(art_text)
        ]
        if not raw_citations:
            _fix_entity_positions(art, art_text)
            continue
        # Build a position-corrected entity list for resolution.  Propagated
        # entities have preamble-relative positions; we search for their
        # reference number (the "15-09" part of "loi n°15-09") in the article
        # text and use that for proximity matching.  If the number isn't found
        # verbatim, we still add the entity with position -1 (the resolver's
        # Step-5 anaphoric path will handle it via context-level matching).
        _NUM_IN_LABEL = _re.compile(r"[\d]+(?:[-–.][\d]+){1,2}")
        corrected_entities = []
        for e in art.get("entities", []):
            label = e.get("label", "")
            if label not in legal_text_labels:
                continue
            text = e.get("text", "")
            if not text:
                continue
            norm = normalize_entity(e)
            # Try to find the reference number in the article text
            num_match = _NUM_IN_LABEL.search(norm["text"])
            if num_match:
                ref_num = num_match.group()
                pos = art_text.find(ref_num)
                if pos >= 0:
                    corrected_entities.append({
                        "label": label, "text": norm["text"],
                        "start": pos, "end": pos + len(ref_num),
                    })
                    continue
            # Fallback: include entity with sentinel position — the resolver
            # will still use it in the anaphoric-marker path (Step 5) where
            # it searches among ALL entities of a matching label.
            corrected_entities.append({
                "label": label, "text": norm["text"],
                "start": -1, "end": -1,
            })
        # Also inject document-level preamble entities (from the overarching
        # law covering the entire BO issue) into the resolution list, so that
        # DOCUMENT_SOURCE citations can resolve against them.
        for de in doc_legal_entities:
            norm = normalize_entity(de)
            num_match = _NUM_IN_LABEL.search(norm["text"])
            if num_match:
                ref_num = num_match.group()
                pos = art_text.find(ref_num)
                if pos >= 0:
                    corrected_entities.append({
                        "label": norm["label"], "text": norm["text"],
                        "start": pos, "end": pos + len(ref_num),
                    })
                    continue
            corrected_entities.append({
                "label": norm["label"], "text": norm["text"],
                "start": -1, "end": -1,
            })
        if not corrected_entities:
            _fix_entity_positions(art, art_text)
            continue
        re_resolved = resolve_citations(
            raw_citations, corrected_entities, doc_id=processed_file.stem,
            full_text=art_text, lang=lang,
        )
        # Enrich existing citations, preserving any already-resolved ones
        existing = {c["text"]: c for c in art.get("citations", [])}
        for rc in re_resolved:
            if rc.text in existing and rc.resolved:
                existing[rc.text]["target_label"] = rc.target_label
                existing[rc.text]["target_text"] = rc.target_text
                existing[rc.text]["resolved"] = True

        # Fix remaining -1 positions in article entities
        _fix_entity_positions(art, art_text)

    result = {
        "source": str(processed_file), "lang": lang,
        **metadata,
        # metadata peut contenir "lang" (ex. pass "ar" court-circuité par
        # _extract_bo_metadata_from_interim sur un texte français) — la
        # langue du pipeline (dossier processed/) doit TOUJOURS gagner.
        "lang": lang,
        "preamble_text": preamble,
        "preamble_entities": [
            a for a in (
                align_entity_text(e, preamble) for e in
                filter_entities(_entities_to_dicts(preamble_doc))
            ) if a is not None
        ],
        "possible_embedded_arabic": arabic_runs or [],
        "sommaire": get_sommaire(text, lang=lang),
        "decrees": decrees,
        "n_articles": len(articles_out), "articles": articles_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Écriture atomique : fichier temporaire → rename pour éviter la troncation
    # en cas d'interruption ou de buffer non vidé (observé sur des JSON > 1 Mo)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(out_path.parent),
        prefix=f".{out_path.name}.", suffix=".tmp", delete=False,
    )
    try:
        json.dump(result, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, str(out_path))
    except Exception:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise

    n_ent = sum(len(a["entities"]) for a in articles_out) + len(result["preamble_entities"])
    print(f"    → {out_path}  ({result['n_articles']} articles, {n_ent} entités)")

    # Also write a Markdown version with formatted tables in data/annotated-MD/
    md_rel = out_path.relative_to(ANNOTATED_DIR).with_suffix(".md")
    md_path = ANNOTATED_MD_DIR / md_rel
    md_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        md_content = build_full_markdown(result)
        md_path.write_text(md_content, encoding="utf-8")
        print(f"    → {md_path}  (Markdown)")
    except Exception as e:
        print(f"    ⚠ Markdown export failed: {e}")

    return out_path


# ============================================================================
# Orchestration
# ============================================================================

def _pdf_initial_language(pdf_path: Path) -> str | None:
    stem = pdf_path.stem  # e.g. "BO_7132_Fr" or "BO_7511_Ar"
    if stem.endswith("_Fr"):
        return "fr"
    if stem.endswith("_Ar"):
        return "ar"
    return None


def process_single_pdf(pdf_path: Path, enrich: bool = False, tables: bool = False):
    """Pipeline complet sur un seul PDF."""
    # Étape 1
    interim_files = run_ingestion(pdf_path)
    if not interim_files:
        print("  Aucun fichier texte produit à l'ingestion.")
        return

    init_lang = _pdf_initial_language(pdf_path)
    if init_lang:
        print(f"  Langue initiale détectée depuis le nom : {init_lang}")

    # On ne garde que les fichiers fr/ar/ (pas unknown, pas tables)
    lang_files = [f for f in interim_files if f.parent.name in ("fr", "ar")]
    if not lang_files:
        print("  Aucun fichier texte fr/ar à traiter.")
        return

    # Try to extract BO metadata from ALL interim files before language routing
    # (header pages are often misrouted by layout detection, e.g. an _Ar PDF
    #  has its first page routed to interim/fr/ while the rest goes to interim/ar/)
    bo_metadata = _extract_bo_metadata_from_interim(pdf_path.stem, interim_files)

    for interim_file in lang_files:
        lang = interim_file.parent.name
        if init_lang and lang != init_lang:
            print(f"  Ignoré (ne correspond pas à la langue initiale) : {interim_file}")
            continue

        # Étape 2
        arabic_runs: list = []
        processed_file = run_preprocessing(interim_file, arabic_runs=arabic_runs)
        if processed_file is None:
            continue

        # Étape 3
        if lang not in _NLP_CACHE:
            _NLP_CACHE[lang] = (
                build_fr_nlp() if lang == "fr" else build_ar_nlp()
            )

        json_kw = {"nlp_fr": _NLP_CACHE.get("fr"), "nlp_ar": _NLP_CACHE.get("ar"),
                   "metadata_override": bo_metadata if bo_metadata.get("bo_number") else None}
        if lang == "fr":
            json_kw["arabic_runs"] = arabic_runs
        out_path = run_extraction(processed_file, lang, **json_kw)

        # Étape 4 : enrichissement (pages + instruments)
        if enrich and out_path and out_path.exists():
            print(f"\n  ÉTAPE 4 — Enrichissement : {out_path.name}")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.enrich_json_with_pages",
                     str(out_path)],
                    check=True, capture_output=True, text=True,
                )
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        print(f"    {line.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠ Enrichissement échoué : {e}")

        # Étape 5 : tables (Priority 3)
        if tables and out_path and out_path.exists():
            # Tables need pdf_page on articles — ensure page backfill (step 4)
            # even if --enrich wasn't passed.
            if not enrich:
                subprocess.run(
                    [sys.executable, "-m", "scripts.enrich_json_with_pages",
                     str(out_path)],
                    check=False, capture_output=True,
                )
            print(f"\n  ÉTAPE 5 — Tables : {out_path.name}")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.enrich_json_with_pages",
                     str(out_path), "--tables"],
                    check=True, capture_output=True, text=True,
                )
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        print(f"    {line.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠ Extraction des tableaux échouée : {e}")


def process_directory(dir_path: Path, enrich: bool = False, tables: bool = False):
    """Pipeline complet sur tous les PDF d'un dossier (récursif)."""
    pdfs = sorted(dir_path.glob("**/*.pdf"))
    if not pdfs:
        print(f"Aucun PDF trouvé dans {dir_path}")
        return
    print(f"{len(pdfs)} fichier(s) PDF trouvé(s).")
    for pdf in pdfs:
        try:
            process_single_pdf(pdf, enrich=enrich, tables=tables)
        except Exception as e:
            import traceback
            print(f"  ✗ Erreur sur {pdf.name} : {e}")
            traceback.print_exc()


def process_all_raw(enrich: bool = False, tables: bool = False):
    """Pipeline complet sur tous les PDF de data/raw/."""
    process_directory(RAW_DIR, enrich=enrich, tables=tables)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet : ingestion → prétraitement → extraction NLP → JSON.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Traiter un seul fichier PDF")
    group.add_argument("--dir", type=str, help="Traiter tous les PDF d'un dossier")
    parser.add_argument("--enrich", action="store_true", help="Enrichir le JSON avec pages et instruments (Priority 1+2)")
    parser.add_argument("--tables", action="store_true", help="Extraire et lier les tableaux (Priority 3)")
    args = parser.parse_args()

    # Mode enrich only : traite tous les JSONs existants sans refaire le pipeline
    if args.enrich and not args.file and not args.dir:
        annotated_dir = ANNOTATED_DIR
        if annotated_dir.exists():
            cmd = ["--tables"] if args.tables else ["--all"]
            print(f"Mode enrichissement seul sur {annotated_dir} ...")
            subprocess.run(
                [sys.executable, "-m", "scripts.enrich_json_with_pages",
                 str(annotated_dir)] + cmd,
                check=False,
            )
            return

    if args.file:
        process_single_pdf(Path(args.file), enrich=args.enrich, tables=args.tables)
    elif args.dir:
        process_directory(Path(args.dir), enrich=args.enrich, tables=args.tables)
    else:
        process_all_raw(enrich=args.enrich, tables=args.tables)


if __name__ == "__main__":
    main()
