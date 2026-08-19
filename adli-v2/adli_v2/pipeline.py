"""
adli_v2.pipeline
----------------
Pipeline v2 : réutilise l'ANCIEN pipeline (scripts.run_pipeline_complet,
scripts.enrich_json_with_pages) en lecture seule — aucun fichier v1 n'est
modifié — puis applique l'étape v2 (metadata + keyword_counts).

Chemins : tout sort dans adli-v2/data/ (interim, processed, annotated,
annotated-MD) ; les PDF d'entrée sont cherchés dans adli-v2/data/uploads.

Usage :
    python -m adli_v2.scripts.run_extraction --file chemin/vers/document.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(V2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_DATA = V2_ROOT / "data"
DEFAULT_UPLOADS = DEFAULT_DATA / "uploads"
DEFAULT_INTERIM = DEFAULT_DATA / "interim"
DEFAULT_PROCESSED = DEFAULT_DATA / "processed"
DEFAULT_ANNOTATED = DEFAULT_DATA / "annotated"
DEFAULT_MD = DEFAULT_DATA / "annotated-MD"


def process_pdf(
    pdf_path,
    *,
    interim_dir=DEFAULT_INTERIM,
    processed_dir=DEFAULT_PROCESSED,
    annotated_dir=DEFAULT_ANNOTATED,
    md_dir=DEFAULT_MD,
    uploads_dir=DEFAULT_UPLOADS,
    classify_domain=False,
):
    """Pipeline complet v2 sur un PDF :

    1. ancien pipeline (ingestion → nettoyage → segmentation → JSON) ;
    2. enrichissement v1 IN-PROCESS (pages + instruments) — le classifieur
       de domaine fine-tuné est volontairement désactivé (classify_domain=
       False) : les compteurs de mots-clés le remplacent en version 2 ;
    3. étape v2 : metadata + keyword_counts par document et par instrument.

    Retourne la liste des JSON annotés v2 produits.
    """
    import scripts.run_pipeline_complet as rpc
    from scripts.enrich_json_with_pages import enrich_json
    from adli_v2.metadata import post_enrich

    pdf_path = Path(pdf_path)
    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)
    annotated_dir = Path(annotated_dir)
    md_dir = Path(md_dir)
    uploads_dir = Path(uploads_dir)

    saved = {}
    for name in ("INTERIM_DIR", "PROCESSED_DIR", "ANNOTATED_DIR", "ANNOTATED_MD_DIR"):
        saved[name] = getattr(rpc, name)
    try:
        # Redirection des constantes du module v1 vers les répertoires v2
        # (mutation in-process, aucun fichier v1 modifié).
        rpc.INTERIM_DIR = interim_dir
        rpc.PROCESSED_DIR = processed_dir
        rpc.ANNOTATED_DIR = annotated_dir
        rpc.ANNOTATED_MD_DIR = md_dir

        rpc.process_single_pdf(pdf_path, enrich=False)

        results = []
        print("\n  ÉTAPE 4 — Enrichissement (instruments + pages)")
        for json_path in sorted(annotated_dir.glob("*_entities.json")):
            enrich_json(
                json_path,
                pdf_dir=uploads_dir,
                classify_domain=classify_domain,
            )
            print("\n  ÉTAPE 5 — Compteurs (mots-clés + métadonnées)")
            post_enrich(json_path)
            results.append(json_path)
        return results
    finally:
        for name, value in saved.items():
            setattr(rpc, name, value)
