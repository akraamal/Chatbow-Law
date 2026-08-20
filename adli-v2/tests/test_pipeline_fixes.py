"""
test_pipeline_fixes.py
----------------------
Régression pour les corrections de adli_v2.pipeline :

  - P1 (critique) : process_pdf() ne mute plus les constantes globales du
    module v1 — deux appels concurrents avec des répertoires distincts ne
    peuvent pas fuir leurs chemins l'un dans l'autre (test de course réelle
    via ThreadPoolExecutor + barrière).
  - P2 (haute) : seuls les fichiers *_entities.json produits pour LE PDF
    courant sont enrichis — un JSON pré-existant du corpus n'est ni relu ni
    réécrit.

Les appels lourds (ingestion/NLP v1) sont remplacés par un double factice de
scripts.run_pipeline_complet.process_single_pdf qui reçoit les répertoires en
paramètres et écrit le JSON de « son » PDF dans « son » annotated_dir — c'est
exactement le contrat qui a remplacé la mutation de globales.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from adli_v2.pipeline import process_pdf


class _Recorder:
    """Enregistre les appels des doubles + une barrière optionnelle pour
    synchroniser les threads DANS la section critique du test de course."""

    def __init__(self):
        self.single_pdf = []
        self.enrich = []
        self.post = []
        self.sync_barrier = None


@pytest.fixture()
def patch_pipeline(monkeypatch):
    """Remplace process_single_pdf + enrich_json + post_enrich par des
    doubles rapides et enregistre les appels pour assertions."""
    import scripts.run_pipeline_complet as rpc
    from scripts import enrich_json_with_pages as ej_mod
    from adli_v2 import metadata as meta_mod

    recorder = _Recorder()

    def fake_process_single_pdf(pdf_path, enrich=False, **kwargs):
        recorder.single_pdf.append({
            "pdf": Path(pdf_path).name,
            "interim": str(kwargs["interim_dir"]),
            "processed": str(kwargs["processed_dir"]),
            "annotated": str(kwargs["annotated_dir"]),
            "annotated_md": str(kwargs["annotated_md_dir"]),
        })
        if recorder.sync_barrier is not None:
            recorder.sync_barrier.wait(timeout=30)
        stem = Path(pdf_path).stem
        out_dir = Path(kwargs["annotated_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{stem}_entities.json"
        out.write_text(json.dumps({
            "doc_id": stem, "lang": "fr", "articles": [], "instruments": [],
        }, ensure_ascii=False), encoding="utf-8")
        return [out]

    def fake_enrich(json_path, pdf_dir=None, classify_domain=False):
        recorder.enrich.append(Path(json_path).name)

    def fake_post(json_path):
        recorder.post.append(Path(json_path).name)

    monkeypatch.setattr(rpc, "process_single_pdf", fake_process_single_pdf)
    monkeypatch.setattr(ej_mod, "enrich_json", fake_enrich)
    monkeypatch.setattr(meta_mod, "post_enrich", fake_post)
    return recorder


def _dirs(root: Path) -> dict:
    return dict(
        interim_dir=root / "interim",
        processed_dir=root / "processed",
        annotated_dir=root / "annotated",
        md_dir=root / "annotated-MD",
        uploads_dir=root / "uploads",
    )


# ── P1 : course entre deux process_pdf() concurrents ──────────────────────

def test_concurrent_process_pdf_no_cross_directory_leak(tmp_path, patch_pipeline):
    """Deux process_pdf() lancés en parallèle sur deux PDF et deux jeux de
    répertoires distincts : chaque run reçoit SES répertoires, son JSON ne
    sort que dans SON annotated_dir, et les deux s'enrichissent sans se
    croiser."""
    recorder = patch_pipeline
    recorder.sync_barrier = threading.Barrier(2)

    def run(pdf_name, root):
        return process_pdf(root / f"{pdf_name}.pdf", **_dirs(root))

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(run, "doc_a", tmp_path / "A")
        f2 = ex.submit(run, "doc_b", tmp_path / "B")
        r1 = f1.result(timeout=60)
        r2 = f2.result(timeout=60)

    by_pdf = {c["pdf"]: c for c in recorder.single_pdf}
    assert set(by_pdf) == {"doc_a.pdf", "doc_b.pdf"}

    # Chaque appel a reçu SES répertoires, pas ceux de l'autre thread.
    assert by_pdf["doc_a.pdf"]["annotated"] == str(tmp_path / "A" / "annotated")
    assert by_pdf["doc_b.pdf"]["annotated"] == str(tmp_path / "B" / "annotated")
    assert by_pdf["doc_a.pdf"]["interim"] == str(tmp_path / "A" / "interim")
    assert by_pdf["doc_b.pdf"]["interim"] == str(tmp_path / "B" / "interim")
    assert by_pdf["doc_a.pdf"]["annotated_md"] == str(tmp_path / "A" / "annotated-MD")
    assert by_pdf["doc_b.pdf"]["annotated_md"] == str(tmp_path / "B" / "annotated-MD")

    # Chaque annotated_dir ne contient QUE son propre JSON (aucune
    # écriture croisée).
    assert list((tmp_path / "A" / "annotated").glob("*_entities.json")) == [
        tmp_path / "A" / "annotated" / "doc_a_entities.json"
    ]
    assert list((tmp_path / "B" / "annotated").glob("*_entities.json")) == [
        tmp_path / "B" / "annotated" / "doc_b_entities.json"
    ]

    # Les deux runs ont bien été enrichis, chacun sur son fichier.
    assert [p.name for p in r1] == ["doc_a_entities.json"]
    assert [p.name for p in r2] == ["doc_b_entities.json"]
    assert sorted(recorder.enrich) == ["doc_a_entities.json", "doc_b_entities.json"]
    assert sorted(recorder.post) == ["doc_a_entities.json", "doc_b_entities.json"]


# ── P2 : pas de ré-enrichissement quadratique du corpus ───────────────────

def test_process_pdf_only_enriches_its_own_files(tmp_path, patch_pipeline):
    """Un *_entities.json pré-existant dans annotated_dir ne doit NI être
    relu par enrich_json/post_enrich NI être réécrit : seule la production
    du PDF courant l'est (avant, la boucle globbait TOUT le corpus)."""
    recorder = patch_pipeline
    annotated = tmp_path / "annotated"
    annotated.mkdir()

    pre = annotated / "BO_9999_Fr_entities.json"
    pre.write_text(json.dumps({"doc_id": "BO_9999_Fr", "sentinel": "old"},
                              ensure_ascii=False), encoding="utf-8")
    pre_mtime_ns = pre.stat().st_mtime_ns
    pre_content = pre.read_text(encoding="utf-8")

    out = process_pdf(tmp_path / "new.pdf", **_dirs(tmp_path))

    # Seul le nouveau fichier est produit et enrichi.
    assert [p.name for p in out] == ["new_entities.json"]
    assert recorder.enrich == ["new_entities.json"]
    assert recorder.post == ["new_entities.json"]

    # Le pré-existant est intact : contenu identique ET mtime inchangé
    # (aucune réécriture silencieuse).
    assert pre.read_text(encoding="utf-8") == pre_content
    assert pre.stat().st_mtime_ns == pre_mtime_ns

    # Il n'y a toujours que deux fichiers dans le dossier.
    assert len(list(annotated.glob("*_entities.json"))) == 2