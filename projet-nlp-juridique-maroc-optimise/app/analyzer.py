#!/usr/bin/env python3
"""Analyseur de Bulletins Officiels — routes Flask.

Upload d'un PDF du Bulletin Officiel, lancement du pipeline en arrière-plan
(scripts/run_pipeline_complet.py), streaming des logs via SSE, puis
visualisation des instruments, articles et entités extraits, avec un chat
documentaire basé sur le document analysé.

Interface : app/templates/analyzer.html
"""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"

analyzer_bp = Blueprint("analyzer", __name__)

# ── In-memory task store ──────────────────────────────────────────────

_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()

# Chat contexts: doc_id -> full result data (kept after task cleanup)
_chat_contexts: dict[str, dict] = {}
_chat_lock = threading.Lock()

# Pipeline simultanés max : chaque run charge spaCy + camel-tools en
# sous-processus (plusieurs Go de RAM) — sans limite, 5 uploads simultanés
# font planter la machine. Les demandes en excès attendent un slot.
_MAX_CONCURRENT_PIPELINES = 2
_pipeline_slots = threading.Semaphore(_MAX_CONCURRENT_PIPELINES)

# TTL de nettoyage : une tâche jamais consommée par /result (onglet fermé)
# ou un contexte de chat jamais réutilisé ne doivent pas rester en RAM
# indéfiniment.
_TASK_TTL_SECONDS = 2 * 3600
_CHAT_CONTEXT_TTL_SECONDS = 24 * 3600
_JANITOR_INTERVAL_SECONDS = 300


def _janitor_loop():
    """Purge périodique de _tasks et _chat_contexts (fuites mémoire quand
    le navigateur ferme l'onglet sans consommer /result ni /chat)."""
    while True:
        time.sleep(_JANITOR_INTERVAL_SECONDS)
        now = time.time()
        with _tasks_lock:
            for tid in list(_tasks):
                if now - _tasks[tid].get("created_at", now) > _TASK_TTL_SECONDS:
                    _tasks.pop(tid, None)
        with _chat_lock:
            for doc_id in list(_chat_contexts):
                if now - _chat_contexts[doc_id].get("_created_at", now) > _CHAT_CONTEXT_TTL_SECONDS:
                    _chat_contexts.pop(doc_id, None)


_janitor_thread = threading.Thread(target=_janitor_loop, daemon=True)
_janitor_thread.start()


def _new_task() -> str:
    tid = uuid.uuid4().hex[:12]
    with _tasks_lock:
        _tasks[tid] = {
            "logs": [],
            "done": False,
            "error": None,
            "result_path": None,
            "created_at": time.time(),
        }
    return tid


def _append_log(tid: str, line: str):
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid]["logs"].append(line)


def _set_done(tid: str, result_path: Path | None = None, error: str | None = None):
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid]["done"] = True
            _tasks[tid]["error"] = error
            if result_path:
                _tasks[tid]["result_path"] = str(result_path)


# ── Colour palette for entity labels ──────────────────────────────────

ENTITY_COLORS = {
    "LOI":               "#e74c3c",
    "DAHIR":             "#8e44ad",
    "DECRET":            "#2980b9",
    "ARRETE":            "#16a085",
    "DECISION":          "#d35400",
    "DELIBERATION":      "#c0392b",
    "CIRCULAIRE":        "#7f8c8d",
    "AVIS":              "#95a5a6",
    "MINISTERE":         "#f39c12",
    "DATE_HIJRI":        "#2c3e50",
    "DATE_GREGORIAN":    "#34495e",
    "VILLE":             "#1abc9c",
    "BULLETIN_OFFICIEL": "#e67e22",
}


def get_entity_color(label: str) -> str:
    return ENTITY_COLORS.get(label, "#95a5a6")


def _looks_like_pdf(f) -> bool:
    """Vérifie le magic-bytes « %PDF- » plutôt que la seule extension.
    Les librairies de parsing PDF (PyMuPDF/pdfplumber/PaddleOCR) ont des
    CVEs connues : on refuse les fichiers renommés .pdf dont le contenu
    n'est pas un vrai PDF, avant qu'ils n'atteignent le pipeline."""
    try:
        f.stream.seek(0)
        head = f.stream.read(5)
        f.stream.seek(0)
        return head == b"%PDF-"
    except Exception:
        return False


# ── Background pipeline runner ───────────────────────────────────────

def _run_pipeline_task(tid: str, pdf_path: Path):
    """Run the full pipeline in a subprocess and stream logs via task store."""
    _append_log(tid, f"  Fichier : {pdf_path.name}")
    _append_log(tid, f"  Lancement du pipeline...")
    _append_log(tid, "")

    # Limite la RAM : attendre un slot au lieu d'enchaîner les pipelines.
    if not _pipeline_slots.acquire(timeout=0):
        _append_log(tid, f"  Limite de {_MAX_CONCURRENT_PIPELINES} pipelines simultanés atteinte — file d'attente...")
        _pipeline_slots.acquire()

    try:
        _run_pipeline_subprocess(tid, pdf_path)
    finally:
        _pipeline_slots.release()


def _run_pipeline_subprocess(tid: str, pdf_path: Path):

    # -u + PYTHONUNBUFFERED: without this, stdout is fully buffered because
    # it's a pipe rather than a TTY. Long silent steps (like OCR) then
    # produce no output until they finish or the buffer fills, which makes
    # the pipeline look frozen and starves the SSE stream below of any
    # data for long enough that the browser's EventSource times out.
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "scripts.run_pipeline_complet",
         "--file", str(pdf_path), "--enrich", "--tables"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=child_env,
    )

    # Read output line by line in real-time
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n\r")
        if line:
            _append_log(tid, line)

    proc.wait()

    # Clean up the temp PDF
    try:
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception:
        pass

    if proc.returncode != 0:
        _set_done(tid, error=f"Pipeline échoué (code {proc.returncode})")
        return

    # Locate the result JSON
    stem = pdf_path.stem  # e.g. BO_7492_abc12345
    candidates = list(ANNOTATED_DIR.glob(f"*{stem}*entities*"))
    if not candidates:
        _set_done(tid, error="Fichier de résultat introuvable")
        return

    _set_done(tid, result_path=candidates[0])


# ── Routes ────────────────────────────────────────────────────────────

@analyzer_bp.route("/analyzer")
def index():
    return flask.render_template("analyzer.html")


@analyzer_bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in flask.request.files:
        return {"error": "Aucun fichier fourni"}, 400

    pdf_file = flask.request.files["file"]
    if not pdf_file.filename.lower().endswith(".pdf"):
        return {"error": "Le fichier doit être un PDF"}, 400
    if not _looks_like_pdf(pdf_file):
        return {"error": "Le fichier n'est pas un PDF valide (extension trompeuse ?)"}, 400

    stem = Path(pdf_file.filename).stem.replace(" ", "_")
    unique_id = uuid.uuid4().hex[:8]
    tmp_pdf = PROJECT_ROOT / "data" / "raw" / "fr" / f"{stem}_{unique_id}.pdf"
    tmp_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf_file.save(str(tmp_pdf))

    tid = _new_task()
    thread = threading.Thread(
        target=_run_pipeline_task, args=(tid, tmp_pdf), daemon=True
    )
    thread.start()

    return flask.jsonify({"task_id": tid})


@analyzer_bp.route("/stream/<task_id>")
def stream(task_id: str):
    """SSE endpoint: yields log lines in real-time, then done/error event."""

    def generate():
        task = _tasks.get(task_id)
        if not task:
            yield "event: error\ndata: Tâche introuvable\n\n"
            return

        last_idx = 0
        last_sent = time.time()
        HEARTBEAT_INTERVAL = 10  # seconds
        while True:
            with _tasks_lock:
                current_logs = list(task["logs"])
                done = task["done"]
                err = task["error"]

            # Yield new lines
            while last_idx < len(current_logs):
                line = current_logs[last_idx]
                yield f"data: {line}\n\n"
                last_idx += 1
                last_sent = time.time()

            if done:
                if err:
                    yield f"event: error\ndata: {err}\n\n"
                else:
                    yield "event: done\ndata: done\n\n"
                break

            # Long silent steps (e.g. OCR) can go many seconds with no new
            # log line. Without something sent periodically, the browser's
            # EventSource (or an intermediate proxy) treats the connection
            # as dead and fires an error, which can trigger the frontend to
            # re-submit the whole upload and restart the pipeline from
            # scratch. A no-op SSE comment keeps the connection alive
            # without being treated as a log line by the client.
            if time.time() - last_sent > HEARTBEAT_INTERVAL:
                yield ": keepalive\n\n"
                last_sent = time.time()

            time.sleep(0.15)

    return flask.Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@analyzer_bp.route("/result/<task_id>")
def result(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        return {"error": "Tâche introuvable"}, 404
    if not task["done"]:
        return {"error": "Tâche pas encore terminée"}, 425  # Too Early
    if task["error"]:
        return {"error": task["error"]}, 500

    result_path = task.get("result_path")
    if not result_path or not Path(result_path).exists():
        return {"error": "Résultat introuvable"}, 500

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Cache du contexte chat (TTL géré par le janitor ci-dessus)
    doc_id = data.get("doc_id", "")
    if doc_id:
        data["_created_at"] = time.time()
        with _chat_lock:
            _chat_contexts[doc_id] = data

    # Note : le fichier de résultat n'est PLUS supprimé et la tâche n'est
    # PLUS dépilée ici — un second onglet / un rafraîchissement rechargent
    # le même résultat au lieu d'obtenir un 404 + fichier perdu. Le
    # nettoyage mémoire est confié au janitor (TTL).

    return flask.jsonify(build_response(data))


def build_response(data: dict) -> dict:
    """Build a simplified JSON response for the frontend."""
    articles = data.get("articles", [])
    instruments = data.get("instruments", [])
    preamble_entities = data.get("preamble_entities", [])

    articles_out = []
    for a in articles:
        entities = a.get("entities", [])
        article_text = a.get("text", "")
        articles_out.append({
            "number": a.get("number", ""),
            "text": article_text,
            "pdf_page": a.get("pdf_page"),
            "printed_page": a.get("printed_page"),
            "entities": [
                {
                    "start": e["start"],
                    "end": e["end"],
                    "label": e["label"],
                    "text": e.get("text", ""),
                    "color": get_entity_color(e["label"]),
                }
                for e in entities
                if e.get("start", -1) >= 0 and e.get("end", -1) >= 0
            ],
        })

    instruments_out = []
    for instr in instruments:
        instr_articles = [
            articles_out[i] for i in instr.get("article_indices", [])
            if i < len(articles_out)
        ]
        tables = instr.get("extracted_tables", [])
        instruments_out.append({
            "instrument_id": instr.get("instrument_id"),
            "instrument_type": instr.get("instrument_type"),
            "reference": instr.get("reference"),
            "n_articles": instr.get("n_articles"),
            "article_indices": instr.get("article_indices"),
            "articles": instr_articles,
            "tables": [
                {
                    "page": t.get("page_number"),
                    "n_rows": t.get("n_rows"),
                    "n_cols": t.get("n_cols"),
                    "headers": t.get("rows", [[]])[0] if t.get("rows") else [],
                    "data": t.get("rows", [])[1:] if t.get("rows") else [],
                }
                for t in tables
            ],
        })

    entity_counts = {}
    for a in articles:
        for e in a.get("entities", []):
            lbl = e.get("label", "")
            entity_counts[lbl] = entity_counts.get(lbl, 0) + 1
    for e in preamble_entities:
        lbl = e.get("label", "")
        entity_counts[lbl] = entity_counts.get(lbl, 0) + 1

    preamble_text = data.get("preamble_text", "")
    preamble_out = [
        {
            "start": e["start"],
            "end": e["end"],
            "label": e["label"],
            "text": e.get("text", ""),
            "color": get_entity_color(e["label"]),
        }
        for e in preamble_entities
        if e.get("start", -1) >= 0 and e.get("end", -1) >= 0
    ]

    return {
        "doc_id": data.get("doc_id", ""),
        "bo_number": data.get("bo_number", ""),
        "date_publication": data.get("date_publication", ""),
        "n_articles": len(articles),
        "n_instruments": len(instruments),
        "preamble_text": data.get("preamble_text", ""),
        "preamble_entities": preamble_out,
        "entity_counts": [
            {"label": lbl, "count": cnt, "color": get_entity_color(lbl)}
            for lbl, cnt in sorted(entity_counts.items(), key=lambda x: -x[1])
        ],
        "instruments": instruments_out,
    }


@analyzer_bp.route("/health")
def health():
    return {"ok": True}


# ── Chatbot documentaire (règles, basé sur le document analysé) ───────

def _search_articles(data: dict, query: str) -> list:
    """Return articles whose text or number matches query (case-insensitive)."""
    q = query.lower()
    results = []
    for a in data.get("articles", []):
        txt = a.get("text", "").lower()
        num = str(a.get("number", "")).lower()
        if q in txt or q in num:
            results.append(a)
    return results[:5]


def _chat_answer(data: dict, question: str) -> str:
    q = question.lower().strip()

    n_arts = len(data.get("articles", []))
    n_instrs = len(data.get("instruments", []))
    bo = data.get("bo_number", "?")
    date_pub = data.get("date_publication", "?")

    # ── Count questions ──
    if any(w in q for w in ["combien", "nombre", "how many", "count"]):
        if "article" in q or "section" in q:
            return f"Ce document contient **{n_arts} articles** au total."
        if "instrument" in q or "décret" in q or "dahir" in q or "loi" in q or "arrêté" in q:
            return f"Ce document contient **{n_instrs} instruments** : " + \
                   ", ".join(f"{i.get('instrument_type','?')} {i.get('reference','')}" for i in data.get("instruments", []))
        if "entité" in q or "entite" in q or "entity" in q:
            counts = {}
            for a in data.get("articles", []):
                for e in a.get("entities", []):
                    lbl = e.get("label", "")
                    counts[lbl] = counts.get(lbl, 0) + 1
            parts = [f"{lbl} : {c}" for lbl, c in sorted(counts.items(), key=lambda x: -x[1])]
            return f"Répartition des entités :\n" + "\n".join(parts)

    # ── Metadata ──
    if any(w in q for w in ["bo numéro", "numero bo", "bulletin", "bo n"]):
        return f"Bulletin Officiel **n° {bo}** du **{date_pub}**."
    if any(w in q for w in ["date", "publication"]):
        return f"Date de publication : **{date_pub}**."

    # ── List instruments ──
    if any(w in q for w in ["liste", "list", "quels sont", "instruments"]):
        lines = []
        for i, instr in enumerate(data.get("instruments", []), 1):
            ref = instr.get("reference", "")
            typ = instr.get("instrument_type", "?")
            na = instr.get("n_articles", 0)
            lines.append(f"**{i}.** {typ} {ref} — {na} articles")
        return "\n".join(lines) if lines else "Aucun instrument détecté."

    # ── Show article ──
    import re
    m = re.search(r"(?:article|art[. ]*)[ ]*(\d+)", q)
    if m:
        art_num = m.group(1)
        for a in data.get("articles", []):
            if a.get("number") == art_num:
                txt = a.get("text", "").strip()
                preview = txt[:600] + "…" if len(txt) > 600 else txt
                return f"**Article {art_num}** (page {a.get('pdf_page','?')}) :\n\n{preview}"
        return f"Article **{art_num}** introuvable dans ce document."

    # ── Search keyword in articles ──
    if len(q) > 3:
        hits = _search_articles(data, q)
        if hits:
            lines = [f"**Article {a.get('number','?')}** — {a.get('text','')[:200]}…" for a in hits]
            return f"Résultats pour « {question} » :\n\n" + "\n".join(lines)

    return f"Je n'ai pas trouvé de réponse à « {question} ».\n\n" \
           "Essayez :\n- « Combien d'articles ? »\n- « Liste des instruments »\n" \
           "- « Article 5 »\n- « Recherche [mot-clé] »\n- « BO numéro ? »"


@analyzer_bp.route("/chat", methods=["POST"])
def chat():
    body = flask.request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    doc_id = (body.get("doc_id") or "").strip()

    if not question:
        return flask.jsonify({"answer": "Veuillez poser une question."})

    with _chat_lock:
        data = _chat_contexts.get(doc_id)

    if not data:
        return flask.jsonify({"answer": "Aucun document analysé en mémoire. Lancez d'abord une analyse."})

    answer = _chat_answer(data, question)
    return flask.jsonify({"answer": answer})
