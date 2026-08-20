"""
adli_v2.app.analyzer
--------------------
Analyseur v2 — portage fidèle de l'analyseur v1 (même page, mêmes contrats
API : /upload, /stream, /cancel, /result, /analyses, /open-analysis,
/chat), adossé au pipeline v2 (adli_v2.scripts.run_extraction, données
dans adli-v2/data/).

Les instruments y sont affichés EXACTEMENT comme en v1 (badges de type,
cartes repliables, articles complets surlignés, onglets Instruments /
Articles / Tableaux).  S'ajoutent en API (non utilisées par la page, pour
rester identique à v1) : /documents, /document/<doc_id>, /keywords.

Interface : adli_v2/app/templates/analyzer_v2.html (= copie de la v1).
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adli_v2.pipeline import DEFAULT_ANNOTATED, DEFAULT_UPLOADS  # noqa: E402

analyzer_bp = Blueprint("analyzer_v2", __name__, template_folder="templates")

# ── Store de tâches (mémoire) ──────────────────────────────────────────
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()

# Chat contexts: doc_id -> result data (kept after task cleanup)
_chat_contexts: dict[str, dict] = {}
_chat_lock = threading.Lock()

_MAX_CONCURRENT_PIPELINES = 2
_pipeline_slots = threading.Semaphore(_MAX_CONCURRENT_PIPELINES)

_TASK_TTL_SECONDS = 2 * 3600
_CHAT_CONTEXT_TTL_SECONDS = 24 * 3600
_JANITOR_INTERVAL_SECONDS = 300


def _janitor_loop():
    while True:
        time.sleep(_JANITOR_INTERVAL_SECONDS)
        now = time.time()
        with _tasks_lock:
            for tid in list(_tasks):
                t = _tasks[tid]
                if now - t["created_at"] > _TASK_TTL_SECONDS:
                    del _tasks[tid]
        with _chat_lock:
            for doc_id in list(_chat_contexts):
                if now - _chat_contexts[doc_id].get("_created_at", 0) > _CHAT_CONTEXT_TTL_SECONDS:
                    del _chat_contexts[doc_id]


threading.Thread(target=_janitor_loop, daemon=True).start()


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


# ── Palette de couleurs des entités (identique à v1) ───────────────────

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
    try:
        f.stream.seek(0)
        head = f.stream.read(5)
        f.stream.seek(0)
        return head == b"%PDF-"
    except Exception:
        return False


# ── Exécution du pipeline v2 en arrière-plan ──────────────────────────

def _run_pipeline_task(tid: str, pdf_path: Path):
    _append_log(tid, f"  Fichier : {pdf_path.name}")
    _append_log(tid, "  Lancement du pipeline...")
    _append_log(tid, "")

    if not _pipeline_slots.acquire(timeout=0):
        _append_log(tid, f"  Limite de {_MAX_CONCURRENT_PIPELINES} pipelines simultanés atteinte — file d'attente...")
        _pipeline_slots.acquire()

    try:
        if _task_cancelled(tid):
            _append_log(tid, "  Analyse annulée avant démarrage.")
            _set_done(tid, error="Analyse interrompue par l'utilisateur")
            return
        _run_pipeline_subprocess(tid, pdf_path)
    finally:
        _pipeline_slots.release()


def _run_pipeline_subprocess(tid: str, pdf_path: Path):
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONPATH"] = str(REPO_ROOT / "adli-v2")

    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "adli_v2.scripts.run_extraction",
         "--file", str(pdf_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        env=child_env,
    )
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid]["proc"] = proc

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

    stem = pdf_path.stem
    candidates = list(DEFAULT_ANNOTATED.glob(f"*{stem}*entities*"))
    if not candidates:
        _set_done(tid, error="Fichier de résultat introuvable")
        return

    _set_done(tid, result_path=candidates[0])


# ── Historique des analyses (dérivé du dossier annoté v2) ─────────────

def _history_entries() -> list[dict]:
    """Analyses précédentes : une entrée par JSON annoté v2 existant,
    comptant uniquement les décrets (cœur de l'outil v2)."""
    DECREE_TYPES = ("DECRET", "DECRET_LOI", "DECRET-LOI", "DÉCRET")
    entries = []
    for path in sorted(DEFAULT_ANNOTATED.glob("*_entities.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        n_decrees = sum(
            1 for i in (data.get("instruments") or [])
            if str(i.get("instrument_type") or "").upper() in DECREE_TYPES
        )
        entries.append({
            "doc_id": data.get("doc_id") or path.stem,
            "result_path": str(path),
            "filename": path.stem,
            "bo_number": data.get("bo_number"),
            "date_publication": data.get("date_publication"),
            "n_instruments": n_decrees,
            "n_articles": len(data.get("articles") or []),
        })
    entries.sort(key=lambda e: str(e.get("bo_number") or ""), reverse=True)
    return entries


# ── Routes ─────────────────────────────────────────────────────────────


@analyzer_bp.route("/analyzer")
def index():
    return flask.render_template("analyzer_v2.html")


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
    tmp_pdf = DEFAULT_UPLOADS / f"{stem}_{unique_id}.pdf"
    tmp_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf_file.save(str(tmp_pdf))

    tid = _new_task(filename=pdf_file.filename)
    threading.Thread(target=_run_pipeline_task, args=(tid, tmp_pdf), daemon=True).start()
    return flask.jsonify({"task_id": tid})


@analyzer_bp.route("/cancel/<task_id>", methods=["POST"])
def cancel(task_id: str):
    if not _cancel_task(task_id):
        return {"ok": False, "error": "Tâche introuvable ou déjà terminée"}, 404
    return {"ok": True}


@analyzer_bp.route("/analyses")
def analyses():
    return flask.jsonify({"analyses": _history_entries()})


@analyzer_bp.route("/open-analysis/<doc_id>")
def open_analysis(doc_id: str):
    for entry in _history_entries():
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
    """SSE : lignes de logs en temps réel, puis événement done/error."""

    def generate():
        task = _tasks.get(task_id)
        if not task:
            yield "event: error\ndata: Tâche introuvable\n\n"
            return

        last_idx = 0
        last_sent = time.time()
        HEARTBEAT_INTERVAL = 10
        while True:
            with _tasks_lock:
                current_logs = list(task["logs"])
                done = task["done"]
                err = task["error"]

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
        return {"error": "Tâche pas encore terminée"}, 425
    if task["error"]:
        return {"error": task["error"]}, 500

    result_path = task.get("result_path")
    if not result_path or not Path(result_path).exists():
        return {"error": "Résultat introuvable"}, 500

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data.get("doc_id", "")
    if doc_id:
        data["_created_at"] = time.time()
        with _chat_lock:
            _chat_contexts[doc_id] = data

    return flask.jsonify(build_response(data))


@analyzer_bp.route("/health")
def health():
    return {"ok": True}


# ── Construction de la réponse frontend (identique à v1) ──────────────


def _count_entities(data: dict) -> dict:
    """Compte les entités : articles + préambule du document + préambules
    par décret (titres d'instruments)."""
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
    """Réponse frontend : mêmes cartes et onglets que v1, mais seuls les
    DÉCRETS sont affichés, enrichis des métadonnées v2 (titre, date,
    référence) et des compteurs de mots-clés (document + instrument)."""
    articles = data.get("articles", [])
    instruments = data.get("instruments", [])
    preamble_entities = data.get("preamble_entities", [])

    DECREE_TYPES = ("DECRET", "DECRET_LOI", "DECRET-LOI", "DÉCRET")
    instruments = [
        i for i in instruments
        if str(i.get("instrument_type") or "").upper() in DECREE_TYPES
    ]

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
            "reference_label": instr.get("reference_label"),
            "title": instr.get("title"),
            "decree_date_gregorian": (
                instr.get("decree_date_gregorian") or instr.get("date_gregorian")
            ),
            "decree_date_hijri": instr.get("decree_date_hijri"),
            "signatories": instr.get("signatories") or [],
            "signatories_flat": instr.get("signatories_flat") or [],
            "n_articles": instr.get("n_articles"),
            "article_indices": instr.get("article_indices"),
            "articles": instr_articles,
            "keyword_counts": instr.get("keyword_counts", {}),
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

    meta = data.get("metadata") or {}
    return {
        "doc_id": data.get("doc_id", ""),
        "bo_number": data.get("bo_number", ""),
        "bo_number_confidence": meta.get("bo_number_confidence"),
        "date_publication": (
            meta.get("date_parution")
            or data.get("bo_date_publication")
            or data.get("date_publication", "")
        ),
        "edition_label": meta.get("edition_label") or data.get("edition_label"),
        "n_articles": len(articles),
        "n_instruments": len(instruments_out),
        "preamble_text": data.get("preamble_text", ""),
        "preamble_entities": preamble_out,
        "entity_counts": [
            {"label": lbl, "count": cnt, "color": get_entity_color(lbl)}
            for lbl, cnt in sorted(entity_counts.items(), key=lambda x: -x[1])
        ],
        "keyword_counts": data.get("keyword_counts", {}),
        "instruments": instruments_out,
    }


# ── Chatbot documentaire (règles, porté de v1 tel quel) ───────────────

def _search_articles(data: dict, query: str) -> list:
    q = query.lower().strip()
    qd = _digits_only(q)
    results = []
    for a in data.get("articles", []):
        txt = a.get("text", "").lower()
        num = str(a.get("number", "")).lower()
        hit = (
            (qd and len(qd) >= 3 and qd in _digits_only(txt))
            or q in txt
            or q in num
        )
        if hit:
            results.append(a)
    return results[:5]


def _western_digits(s: str) -> str:
    return str(s).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _digits_only(s: str) -> str:
    return "".join(c for c in _western_digits(s) if c.isdigit())


def _canonical_instrument_type(i_type: str) -> str:
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


def _chat_decrees(data: dict) -> list:
    DECREE_TYPES = ("DECRET", "DECRET_LOI", "DECRET-LOI", "DÉCRET")
    return [
        i for i in data.get("instruments", [])
        if str(i.get("instrument_type") or "").upper() in DECREE_TYPES
    ]


def _find_instrument_by_reference(data: dict, query: str) -> dict | None:
    """Trouve un décret par sa référence, quelle que soit la formulation :
    « n° 2-25-439 », « 2.25.439 », « décret 2 25 439 », « رقم ٢.٢٥.٤٣٩ »,
    « numéro 2-25-439 »...  Compare les chiffres après normalisation
    (séparateurs . - , espaces et chiffres arabes ignorés)."""
    import re
    q = _western_digits(query)
    patterns = [
        # avec préfixe explicite (n°, n, numéro, رقم)
        r"(?:n\s*[°o]?\s*|num[ée]ro\s*|رقم\s*)(\d+(?:[.,\-\s]\d+){1,3})",
        # référence multi-parties seule, n'importe où dans la question
        r"(?<![\d.,])(\d{1,4}(?:[.,\-\s]\d{1,4}){1,3})(?![\d.,])",
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(re.findall(pat, q))
    for cand in candidates:
        want = _digits_only(cand)
        if len(want) < 3:
            continue
        for instr in _chat_decrees(data):
            if want == _digits_only(instr.get("reference", "")):
                return instr
    return None


def _normalize_name(s: str) -> str:
    """Normalise un nom pour comparaison : minuscules, sans accents ni
    diacritiques, espaces multiples aplatis."""
    import unicodedata
    s = unicodedata.normalize("NFD", str(s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def _find_signer_in_query(data: dict, query: str) -> list[str]:
    """Noms de signataires mentionnés dans la question (normalisés).
    « qu'a signé Aziz Rabbah ? » -> ['aziz rabbah'] ;
    « décrets signés par le ministre de l'économie » -> ['nadia fettah', ...]."""
    qn = _normalize_name(query)
    matched: set[str] = set()
    for instr in _chat_decrees(data):
        for s in instr.get("signatories") or []:
            nm = _normalize_name(s.get("name") or "")
            if not nm:
                continue
            role_n = _normalize_name(s.get("role") or "")
            if (
                nm in qn
                or any(len(t) > 3 and t in qn for t in nm.split())
                or (role_n and len(role_n) > 6 and role_n in qn)
            ):
                matched.add(nm)
    return sorted(matched)


def _chat_answer(data: dict, question: str) -> str:
    q = question.lower().strip()

    n_arts = len(data.get("articles", []))
    n_instrs = len(_chat_decrees(data))
    bo = data.get("bo_number", "?")
    date_pub = data.get("date_publication", "?")

    signer_intent = any(w in q for w in [
        "signé", "signe", "signer", "signataire", "signature",
        "signed", "signs", "signed by", "وقع", "توقيع", "موق", "بالعطف",
    ])

    if any(w in q for w in ["combien", "nombre", "how many", "count", "عدد"]):
        if signer_intent or "signataire" in q:
            exact = _find_instrument_by_reference(data, q)
            if exact:
                n = len(exact.get("signatories") or [])
                return f"Le décret **{exact.get('reference','')}** a été signé par **{n} personne(s)**."
        if "instrument" in q or "décret" in q or "dahir" in q or "loi" in q or "arrêté" in q:
            exact = _find_instrument_by_reference(data, q)
            if exact:
                return (f"L'instrument **{exact.get('instrument_type') or '?'} "
                        f"{exact.get('reference','')}** contient **{exact.get('n_articles','?')} articles**.")
            return f"Ce document contient **{n_instrs} décrets** : " + \
                   ", ".join(f"décret n° {i.get('reference','')}" for i in _chat_decrees(data))
        if "article" in q or "section" in q:
            return f"Ce document contient **{n_arts} articles** au total."
        if "entité" in q or "entite" in q or "entity" in q:
            counts = _count_entities(data)
            parts = [f"{lbl} : {c}" for lbl, c in sorted(counts.items(), key=lambda x: -x[1])]
            return f"Répartition des entités :\n" + "\n".join(parts)

    if any(w in q for w in ["bo numéro", "numero bo", "bulletin", "bo n"]):
        return f"Bulletin Officiel **n° {bo}** du **{date_pub}**."
    if any(w in q for w in ["date", "publication"]):
        return f"Date de publication : **{date_pub}**."

    # ── Signataires : qui signe quel décret, et quel décret une personne signe ──
    if signer_intent or "sign" in q or "وقع" in q or "توقيع" in q:
        exact = _find_instrument_by_reference(data, q)
        if exact:
            sigs = exact.get("signatories") or []
            if not sigs:
                return f"Le décret **{exact.get('reference','')}** n'a pas de signataire enregistré."
            lines = [f"**{s.get('role') or 'Signataire'}** : {s.get('name')}" for s in sigs]
            return f"Le **décret {exact.get('reference','')}** est signé par :\n\n" + "\n".join(lines)

        matched = _find_signer_in_query(data, q)
        if matched:
            lines = []
            for instr in _chat_decrees(data):
                for s in instr.get("signatories") or []:
                    if _normalize_name(s.get("name") or "") in matched:
                        lines.append(
                            f"**décret {instr.get('reference','')}** — {s.get('name')} ({s.get('role') or 'Signataire'})"
                        )
            if lines:
                return f"Décrets signés par **{', '.join(matched)}** :\n\n" + "\n".join(lines)

        all_sigs = []
        for instr in _chat_decrees(data):
            for s in instr.get("signatories") or []:
                all_sigs.append((instr.get("reference"), s))
        if all_sigs:
            lines = [
                f"**décret {ref}** — {s.get('name')} ({s.get('role') or 'Signataire'})"
                for ref, s in all_sigs
            ]
            return f"Signataires des décrets de ce document :\n\n" + "\n".join(lines)

    exact = _find_instrument_by_reference(data, q)
    if exact:
        content_intent = any(w in q for w in [
            "contenu", "content", "résumé", "resume", "summary",
            "que dit", "que prévoit", "what does", "what is",
            "c'est quoi", "quoi", "dispose", "mضمون", "محتوى", "محتوي", "مضمون",
        ])
        art_idxs = exact.get("article_indices", [])
        n_max = min(len(art_idxs), 6) if content_intent else min(len(art_idxs), 3)
        previews = []
        for i in art_idxs[:n_max]:
            if isinstance(i, int) and i < len(data.get("articles", [])):
                a = data["articles"][i]
                txt = (a.get("text") or "").strip()
                shown = txt[:600] + ("…" if len(txt) > 600 else "")
                previews.append(f"**Article {a.get('number','?')}** — {shown}")
        head = (f"**{exact.get('instrument_type') or '?'} {exact.get('reference','')}** — "
                f"**{exact.get('n_articles','?')} articles**, BO n°{bo}.")
        title = exact.get("title") or exact.get("reference_label") or ""
        if title and title.lower() != (f"{exact.get('reference_label') or ''}").lower():
            head += f"\n\n*{title}*"
        date = exact.get("decree_date_gregorian") or exact.get("date_gregorian") or ""
        if date:
            head += f"\n\n📅 {date}"
        body = "\n\n".join(previews) if previews else ""
        if content_intent and len(art_idxs) > n_max:
            body += f"\n\n… et {len(art_idxs) - n_max} autre(s) article(s)."
        return head + ("\n\n" + body if body else "")

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
        matched = _chat_decrees(data)
        if wanted_type:
            matched = [i for i in matched
                       if _canonical_instrument_type(i.get("instrument_type")) == wanted_type]
        if not matched:
            return f"Aucun instrument de type « {wanted_type or '?'} » trouvé dans ce document."
        matched = sorted(matched, key=lambda i: -(i.get("n_articles") or 0))
        lines = []
        for i, instr in enumerate(matched[:8], 1):
            ref = instr.get("reference", "")
            typ = instr.get("instrument_type") or "?"
            na = instr.get("n_articles", 0)
            lines.append(f"**{i}.** {typ} {ref} — {na} articles")
        if len(matched) > 8:
            lines.append(f"… et {len(matched) - 8} autre(s) décret(s).")
        title = f"Décrets « {wanted_type or 'tous types'} » triés par importance (nombre d'articles) :"
        return title + "\n\n" + "\n".join(lines)

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

    import re
    m = re.search(r"(?:article|art[. ]*)[ ]*(\d+)", q)
    if m:
        art_num = m.group(1)
        for a in data.get("articles", []):
            if a.get("number") == art_num:
                txt = a.get("text", "").strip()
                return f"**Article {art_num}** (page {a.get('pdf_page','?')}) :\n\n{txt}"
        return f"Article **{art_num}** introuvable dans ce document."

    if len(q) > 3:
        hits = _search_articles(data, q)
        if hits:
            lines = []
            for a in hits:
                txt = a.get("text", "")
                shown = txt[:600] + ("…" if len(txt) > 600 else "")
                lines.append(f"**Article {a.get('number','?')}** — {shown}")
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


# ── API v2 (complémentaire, non utilisée par la page) ─────────────────


@analyzer_bp.route("/documents")
def documents():
    docs = []
    for entry in _history_entries():
        docs.append({
            "doc_id": entry["doc_id"],
            "doc_name": entry["filename"],
            "bo_number": entry["bo_number"],
            "date_parution": entry["date_publication"],
            "n_instruments": entry["n_instruments"],
            "n_articles": entry["n_articles"],
        })
    return flask.jsonify({"documents": docs})


@analyzer_bp.route("/document/<doc_id>")
def document(doc_id: str):
    path = DEFAULT_ANNOTATED / f"{doc_id}.json"
    if not path.exists():
        path = DEFAULT_ANNOTATED / f"{doc_id}_entities.json"
    if not path.exists():
        return {"error": f"document inconnu : {doc_id}"}, 404
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return flask.jsonify({
        "doc_id": data.get("doc_id"),
        "lang": data.get("lang"),
        "metadata": data.get("metadata"),
        "keyword_counts": data.get("keyword_counts", {}),
        "instruments": data.get("instruments", []),
        "n_instruments": len(data.get("instruments") or []),
        "n_articles": len(data.get("articles") or []),
    })


@analyzer_bp.route("/keywords")
def keywords():
    per_category: dict[str, int] = {}
    per_term: dict[str, int] = {}
    n_docs = 0
    for entry in _history_entries():
        try:
            with open(entry["result_path"], encoding="utf-8") as f:
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