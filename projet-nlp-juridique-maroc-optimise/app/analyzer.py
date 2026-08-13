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


def _new_task(filename: str = "") -> str:
    tid = uuid.uuid4().hex[:12]
    with _tasks_lock:
        _tasks[tid] = {
            "logs": [],
            "done": False,
            "error": None,
            "result_path": None,
            "cancel": False,
            "proc": None,
            "filename": filename,
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


def _task_cancelled(tid: str) -> bool:
    with _tasks_lock:
        return bool(_tasks.get(tid, {}).get("cancel"))


def _cancel_task(tid: str) -> bool:
    """Demande l'interruption d'une tâche en cours : pose le drapeau
    `cancel` (lu par la boucle de lecture du sous-processus) et tue le
    sous-processus pipeline si celui-ci est déjà lancé."""
    with _tasks_lock:
        task = _tasks.get(tid)
        if not task or task.get("done"):
            return False
        task["cancel"] = True
        proc = task.get("proc")
    if proc and proc.poll() is None:
        _append_log(tid, "  Interruption demandée — arrêt du pipeline...")
        try:
            proc.terminate()
        except Exception:
            pass
    return True


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
        # Interruption pendant l'attente d'un slot ?
        if _task_cancelled(tid):
            _append_log(tid, "  Analyse annulée avant démarrage.")
            _set_done(tid, error="Analyse interrompue par l'utilisateur")
            return
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
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid]["proc"] = proc

    # Read output line by line in real-time
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n\r")
        if line:
            _append_log(tid, line)
        if _task_cancelled(tid):
            _append_log(tid, "  ⚠ Analyse interrompue par l'utilisateur.")
            try:
                proc.terminate()
            except Exception:
                pass
            break

    proc.wait()

    # Clean up the temp PDF
    try:
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception:
        pass

    if _task_cancelled(tid):
        _set_done(tid, error="Analyse interrompue par l'utilisateur")
        return

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
    _record_analysis(tid, candidates[0])


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

    tid = _new_task(filename=pdf_file.filename)
    thread = threading.Thread(
        target=_run_pipeline_task, args=(tid, tmp_pdf), daemon=True
    )
    thread.start()

    return flask.jsonify({"task_id": tid})


# ── Historique persistant des analyses ────────────────────────────────

HISTORY_FILE = PROJECT_ROOT / "data" / "analyses_history.json"
_history_lock = threading.Lock()
_MAX_HISTORY = 50


def _load_history() -> list[dict]:
    """Charge le registre des analyses terminées (persisté sur disque)."""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _save_history(history: list[dict]):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _record_analysis(tid: str, result_path: Path):
    """Ajoute une analyse terminée au registre persistant (dédupliquée par
    doc_id — une réanalyse du même document remplace l'ancienne entrée)."""
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    doc_id = data.get("doc_id", "")
    entry = {
        "doc_id": doc_id,
        "task_id": tid,
        "filename": data.get("source_file", "") or data.get("filename", ""),
        "bo_number": data.get("bo_number", ""),
        "date_publication": data.get("date_publication", ""),
        "n_instruments": len(data.get("instruments", [])),
        "n_articles": len(data.get("articles", [])),
        "result_path": str(result_path),
        "created_at": time.time(),
    }

    with _history_lock:
        history = [h for h in _load_history() if h.get("doc_id") != doc_id]
        history.insert(0, entry)
        _save_history(history[:_MAX_HISTORY])


@analyzer_bp.route("/cancel/<task_id>", methods=["POST"])
def cancel(task_id: str):
    """Interrompt une analyse en cours (pipeline en arrière-plan)."""
    if not _cancel_task(task_id):
        return {"ok": False, "error": "Tâche introuvable ou déjà terminée"}, 404
    return {"ok": True}


@analyzer_bp.route("/analyses")
def analyses():
    """Liste des analyses précédentes (documents déjà analysés)."""
    history = _load_history()
    # Seules les entrées dont le résultat existe encore sont listées.
    visible = [h for h in history if h.get("result_path") and Path(h["result_path"]).exists()]
    return flask.jsonify({"analyses": visible})


@analyzer_bp.route("/open-analysis/<doc_id>")
def open_analysis(doc_id: str):
    """Recharge une analyse passée : le résultat complet (résultats + chat
    documentaire) redevient disponible comme si l'analyse venait de finir."""
    for entry in _load_history():
        if entry.get("doc_id") != doc_id:
            continue
        result_path = entry.get("result_path")
        if not result_path or not Path(result_path).exists():
            return {"error": "Résultat de l'analyse introuvable"}, 404
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_created_at"] = time.time()
        with _chat_lock:
            _chat_contexts[doc_id] = data
        return flask.jsonify(build_response(data))
    return {"error": "Analyse introuvable dans l'historique"}, 404


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


def _count_entities(data: dict) -> dict:
    """Compte les entités du document : articles + préambule du document +
    préambules par décret (titres d'instruments).

    Sans les préambules par décret, les dahirs de promulgation (dont le
    numéro ne figure QUE dans leur titre, jamais dans le corps d'un
    article) restaient invisibles au comptage : BO_6758 affichait
    « DAHIR : 9 » alors que le document contient 21 vrais dahirs.
    """
    counts: dict[str, int] = {}

    def _add(ents) -> None:
        for e in ents:
            lbl = e.get("label", "")
            if lbl:
                counts[lbl] = counts.get(lbl, 0) + 1

    for a in data.get("articles", []):
        _add(a.get("entities", []))
    _add(data.get("preamble_entities", []))
    for dec in data.get("decrees", []):
        _add(dec.get("entities", []))

    return counts


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

    entity_counts = _count_entities(data)

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


def _western_digits(s: str) -> str:
    """Convertit les chiffres arabes ٠-٩ en chiffres occidentaux."""
    return str(s).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _digits_only(s: str) -> str:
    return "".join(c for c in _western_digits(s) if c.isdigit())


def _canonical_instrument_type(i_type: str) -> str:
    """Normalise un type d'instrument (« Décret »/« décret »/« decret »)
    vers son nom canonique pour la comparaison avec le routage."""
    if not i_type:
        return ""
    import unicodedata
    n = "".join(
        c for c in unicodedata.normalize("NFD", i_type.lower())
        if unicodedata.category(c) != "Mn"
    )
    return {
        "dahir": "Dahir", "loi": "Loi", "decret": "Décret",
        "arrete": "Arrêté", "decision": "Décision", "avis": "Avis",
        "instruction": "Instruction", "ordonnance": "Ordonnance",
    }.get(n, n.title())


def _find_instrument_by_reference(data: dict, query: str) -> dict | None:
    """Retrouve l'instrument dont la référence figure dans la question
    (« décret n° 2-25-1080 », « المرسوم رقم 2.24.874 »)."""
    import re
    m = re.search(r"(?:n\s*[°o]?\s*|رقم\s*)([\d٠-٩]+(?:[-.][\d٠-٩]+)+)", query)
    if not m:
        return None
    want = _digits_only(m.group(1))
    if len(want) < 3:
        return None
    for instr in data.get("instruments", []):
        if want == _digits_only(instr.get("reference", "")):
            return instr
    return None


def _chat_answer(data: dict, question: str) -> str:
    q = question.lower().strip()

    n_arts = len(data.get("articles", []))
    n_instrs = len(data.get("instruments", []))
    bo = data.get("bo_number", "?")
    date_pub = data.get("date_publication", "?")

    # ── Count questions ──
    if any(w in q for w in ["combien", "nombre", "how many", "count", "عدد"]):
        if "instrument" in q or "décret" in q or "dahir" in q or "loi" in q or "arrêté" in q:
            exact = _find_instrument_by_reference(data, q)
            if exact:
                return (f"L'instrument **{exact.get('instrument_type','?')} "
                        f"{exact.get('reference','')}** contient **{exact.get('n_articles','?')} articles**.")
            return f"Ce document contient **{n_instrs} instruments** : " + \
                   ", ".join(f"{i.get('instrument_type','?')} {i.get('reference','')}" for i in data.get("instruments", []))
        if "article" in q or "section" in q:
            return f"Ce document contient **{n_arts} articles** au total."
        if "entité" in q or "entite" in q or "entity" in q:
            counts = _count_entities(data)
            parts = [f"{lbl} : {c}" for lbl, c in sorted(counts.items(), key=lambda x: -x[1])]
            return f"Répartition des entités :\n" + "\n".join(parts)

    # ── Metadata ──
    if any(w in q for w in ["bo numéro", "numero bo", "bulletin", "bo n"]):
        return f"Bulletin Officiel **n° {bo}** du **{date_pub}**."
    if any(w in q for w in ["date", "publication"]):
        return f"Date de publication : **{date_pub}**."

    # ── Instrument précis par référence (« décret n° 2-25-1080 ») ──
    exact = _find_instrument_by_reference(data, q)
    if exact:
        art_idxs = exact.get("article_indices", [])
        previews = []
        for i in art_idxs[:3]:
            if isinstance(i, int) and i < len(data.get("articles", [])):
                a = data["articles"][i]
                txt = (a.get("text") or "").strip()[:220]
                previews.append(f"**Article {a.get('number','?')}** — {txt}…")
        head = (f"**{exact.get('instrument_type','?')} {exact.get('reference','')}** — "
                f"**{exact.get('n_articles','?')} articles**, BO n°{bo}.")
        return head + ("\n\n" + "\n\n".join(previews) if previews else "")

    # ── Liste des instruments (générale ou par type, ordonnée par importance) ──
    try:
        from src.rag.query_routing import route_query
        wanted_type = route_query(question).get("type")
    except Exception:
        wanted_type = None
    asks_list = any(w in q for w in ["liste", "list", "quels sont", "quelles sont",
                                     "lister", "énumérer", "enumere", "recense"])
    importance_signals = ("plus importants", "plus importantes", "importants",
                          "importantes", "majeurs", "majeures", "principaux",
                          "principales", "récents", "récentes", "important",
                          "importante", "tous les", "toutes les")
    if asks_list or (wanted_type and any(s in q for s in importance_signals)):
        matched = data.get("instruments", [])
        if wanted_type:
            matched = [i for i in matched
                       if _canonical_instrument_type(i.get("instrument_type")) == wanted_type]
        if not matched:
            return f"Aucun instrument de type « {wanted_type or '?'} » trouvé dans ce document."
        matched = sorted(matched, key=lambda i: -(i.get("n_articles") or 0))
        lines = []
        for i, instr in enumerate(matched[:8], 1):
            ref = instr.get("reference", "")
            typ = instr.get("instrument_type", "?")
            na = instr.get("n_articles", 0)
            lines.append(f"**{i}.** {typ} {ref} — {na} articles")
        if len(matched) > 8:
            lines.append(f"… et {len(matched) - 8} autre(s) instrument(s).")
        title = f"Instruments « {wanted_type or 'tous types'} » triés par importance (nombre d'articles) :"
        return title + "\n\n" + "\n".join(lines)

    # ── Domaine principal du document ──
    if any(w in q for w in ["domaine", "sujet principal", "principalement",
                            "thème", "thèmes", "themes", "matière traité"]):
        try:
            from src.classification.keyword_classifier import classify_text_with_scores
            text = " ".join((a.get("text") or "") for a in data.get("articles", []))[:80000]
            if len(text.strip()) < 50:
                return "Texte insuffisant pour classifier le domaine de ce document."
            scores = classify_text_with_scores(text, lang=data.get("lang", "fr")) or {}
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
            if not top or top[0][1] <= 0:
                return "Aucun domaine dominant détecté dans ce document."
            lines = [f"**{d}** : {c} occurrence(s)" for d, c in top]
            return f"Domaine(s) principal(aux) de ce document :\n\n" + "\n".join(lines)
        except Exception:
            pass

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
           "Essayez :\n- « Combien d'articles ? »\n- « Liste des décrets »\n" \
           "- « Les lois les plus importantes »\n- « Quel est le domaine principal ? »\n" \
           "- « Article 5 »\n- « Recherche [mot-clé] »\n- « décret n° 2-25-1080 »\n" \
           "- « BO numéro ? »"


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
