"""
adli_v2.app.analyzer
--------------------
Analyseur v2 centré décret — routes Flask.

Upload d'un PDF du Bulletin Officiel → pipeline v2 en arrière-plan
(adli_v2.scripts.run_extraction, sous-processus) avec logs streamés en
SSE, puis :
  /documents          liste des documents traités (métadonnées) ;
  /document/<doc_id>  vue décret-first : instruments triés (décrets en
                      premier), articles COMPLETS, compteurs de mots-clés ;
  /keywords           fréquences agrégées de mots-clés sur le corpus.

Interface : adli_v2/app/templates/analyzer_v2.html
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import flask
from flask import Blueprint

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adli_v2.catalog import TYPE_RANK  # noqa: E402
from adli_v2.pipeline import DEFAULT_ANNOTATED, DEFAULT_UPLOADS  # noqa: E402

analyzer_bp = Blueprint("analyzer_v2", __name__, template_folder="templates")

# ── Store de tâches (mémoire) ──────────────────────────────────────────
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_MAX_CONCURRENT_PIPELINES = 2
_pipeline_slots = threading.Semaphore(_MAX_CONCURRENT_PIPELINES)


def _new_task(filename: str = "") -> str:
    tid = uuid.uuid4().hex[:12]
    with _tasks_lock:
        _tasks[tid] = {
            "filename": filename, "logs": [], "done": False,
            "error": None, "result": [], "cancelled": False,
        }
    return tid


def _append_log(tid: str, line: str):
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid]["logs"].append(line)


def _set_done(tid: str, result: list[str] | None = None, error: str | None = None):
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid]["done"] = True
            _tasks[tid]["error"] = error
            if result is not None:
                _tasks[tid]["result"] = result


def _looks_like_pdf(f) -> bool:
    head = f.read(5)
    f.seek(0)
    return head.startswith(b"%PDF")


def _run_pipeline_task(tid: str, pdf_path: Path):
    with _pipeline_slots:
        if _tasks.get(tid, {}).get("cancelled"):
            _set_done(tid, error="Annulé")
            return
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "adli_v2.scripts.run_extraction",
             "--file", str(pdf_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT), env=env, text=True,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if line.strip():
                _append_log(tid, line)
        proc.wait()
        if proc.returncode != 0:
            _set_done(tid, error=f"Pipeline terminé avec le code {proc.returncode}")
        else:
            produced = sorted(DEFAULT_ANNOTATED.glob("*_entities.json"))
            _set_done(tid, result=[p.name for p in produced])


# ── Helpers de lecture des documents v2 ────────────────────────────────


def _annotated_files() -> list[Path]:
    return sorted(DEFAULT_ANNOTATED.glob("*_entities.json"))


def _load_doc(doc_id: str) -> dict | None:
    path = DEFAULT_ANNOTATED / f"{doc_id}.json"
    if not path.exists():
        path = DEFAULT_ANNOTATED / f"{doc_id}_entities.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _instruments_sorted(data: dict) -> list[dict]:
    instruments = data.get("instruments") or []
    return sorted(
        instruments,
        key=lambda i: (
            TYPE_RANK.get(str(i.get("instrument_type") or "").upper(), 10),
            str(i.get("reference") or ""),
        ),
    )


def _doc_summary(data: dict) -> dict:
    meta = data.get("metadata") or {}
    kc = data.get("keyword_counts") or {}
    return {
        "doc_id": data.get("doc_id"),
        "doc_name": meta.get("doc_name"),
        "lang": data.get("lang"),
        "bo_number": meta.get("bo_number"),
        "date_parution": meta.get("date_parution"),
        "n_articles": meta.get("n_articles"),
        "n_instruments": meta.get("n_instruments"),
        "categories": kc.get("per_category", {}),
    }


# ── Routes ─────────────────────────────────────────────────────────────


@analyzer_bp.route("/analyzer")
def index():
    return flask.render_template("analyzer_v2.html")


@analyzer_bp.route("/analyze", methods=["POST"])
def upload():
    if "file" not in flask.request.files:
        return {"error": "Aucun fichier fourni"}, 400
    pdf_file = flask.request.files["file"]
    if not pdf_file.filename.lower().endswith(".pdf"):
        return {"error": "Le fichier doit être un PDF"}, 400
    if not _looks_like_pdf(pdf_file):
        return {"error": "Le fichier n'est pas un PDF valide"}, 400

    stem = Path(pdf_file.filename).stem.replace(" ", "_")
    unique_id = uuid.uuid4().hex[:8]
    target = DEFAULT_UPLOADS / f"{stem}_{unique_id}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    pdf_file.save(str(target))

    tid = _new_task(filename=pdf_file.filename)
    threading.Thread(target=_run_pipeline_task, args=(tid, target), daemon=True).start()
    return flask.jsonify({"task_id": tid})


@analyzer_bp.route("/stream/<task_id>")
def stream(task_id: str):
    def generate():
        last = 0
        while True:
            with _tasks_lock:
                task = _tasks.get(task_id)
                if task is None:
                    yield "event: done\ndata: {\"error\": \"tâche inconnue\"}\n\n"
                    return
                new_lines = task["logs"][last:]
                done, error, result = task["done"], task["error"], task["result"]
            for line in new_lines:
                yield f"data: {json.dumps(line, ensure_ascii=False)}\n\n"
            last += len(new_lines)
            if done:
                payload = {"done": True, "error": error, "result": result}
                yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            time.sleep(0.5)

    return flask.Response(generate(), mimetype="text/event-stream")


@analyzer_bp.route("/analysis/<task_id>")
def analysis(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return {"error": "tâche inconnue"}, 404
    return {
        "done": task["done"], "error": task["error"], "result": task["result"],
    }


@analyzer_bp.route("/documents")
def documents():
    docs = []
    for path in _annotated_files():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        docs.append(_doc_summary(data))
    docs.sort(key=lambda d: str(d.get("bo_number") or ""), reverse=True)
    return flask.jsonify({"documents": docs})


@analyzer_bp.route("/document/<doc_id>")
def document(doc_id: str):
    data = _load_doc(doc_id)
    if data is None:
        return {"error": f"document inconnu : {doc_id}"}, 404
    instruments = _instruments_sorted(data)
    # Articles complets, indexés par position, avec leur page PDF.
    articles = []
    for i, art in enumerate(data.get("articles") or []):
        articles.append({
            "index": i,
            "number": art.get("number"),
            "raw_header": art.get("raw_header"),
            "pdf_page": art.get("pdf_page"),
            "text": art.get("text"),
        })
    return flask.jsonify({
        "doc_id": data.get("doc_id"),
        "lang": data.get("lang"),
        "metadata": data.get("metadata"),
        "keyword_counts": data.get("keyword_counts", {}),
        "instruments": instruments,
        "articles": articles,
        "total_pdf_pages": data.get("total_pdf_pages"),
    })


@analyzer_bp.route("/keywords")
def keywords():
    """Fréquences agrégées de mots-clés sur tous les documents traités."""
    per_category: dict[str, int] = {}
    per_term: dict[str, int] = {}
    n_docs = 0
    for path in _annotated_files():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        kc = data.get("keyword_counts") or {}
        for cat, total in (kc.get("per_category") or {}).items():
            per_category[cat] = per_category.get(cat, 0) + total
        for term, total in (kc.get("per_term") or {}).items():
            per_term[term] = per_term.get(term, 0) + total
        n_docs += 1
    top_terms = sorted(per_term.items(), key=lambda kv: kv[1], reverse=True)[:25]
    return flask.jsonify({
        "n_documents": n_docs,
        "per_category": dict(sorted(per_category.items(), key=lambda kv: kv[1], reverse=True)),
        "top_terms": top_terms,
    })