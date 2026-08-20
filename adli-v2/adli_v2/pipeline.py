"""
adli_v2.pipeline
----------------
Pipeline v2 : réutilise l'ANCIEN pipeline (scripts.run_pipeline_complet,
scripts.enrich_json_with_pages) en lecture seule — aucun fichier v1 n'est
modifié — puis applique l'étape v2 (metadata + keyword_counts).

Chemins : tout sort dans adli-v2/data/ (interim, processed, annotated,
annotated-MD) ; les PDF d'entrée sont cherchés dans adli-v2/data/uploads.

Les répertoires de sortie sont transmis EN PARAMÈTRE aux fonctions v1
(scripts.run_pipeline_complet.process_single_pdf accepte désormais
interim_dir / processed_dir / annotated_dir / annotated_md_dir) : aucune
constante globale n'est mutée, plusieurs process_pdf() peuvent donc tourner
en parallèle sur des jeux de répertoires distincts sans se corrompre.

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

    Seuls les fichiers *_entities.json produits pour CE PDF sont enrichis
    (retournés par process_single_pdf) — jamais l'ensemble du corpus déjà
    présent dans annotated_dir.  Retourne la liste des JSON annotés v2
    produits.
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

    # Les répertoires sont passés en arguments : les constantes du module
    # v1 ne sont PAS modifiées, donc deux process_pdf() concurrents ne
    # peuvent pas fuir leurs chemins l'un dans l'autre (le semaphore
    # _pipeline_slots = 2 de l'analyseur reste valide).
    produced = rpc.process_single_pdf(
        pdf_path,
        enrich=False,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        annotated_dir=annotated_dir,
        annotated_md_dir=md_dir,
    )

    results = []
    for json_path in produced or []:
        if not json_path.exists():
            continue
        print("\n  ÉTAPE 4 — Enrichissement (instruments + pages)")
        enrich_json(
            json_path,
            pdf_dir=uploads_dir,
            classify_domain=classify_domain,
        )
        print("\n  ÉTAPE 5 — Compteurs (mots-clés + métadonnées)")
        post_enrich(json_path)
        results.append(json_path)
    return results