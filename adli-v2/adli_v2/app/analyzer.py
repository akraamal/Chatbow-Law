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
import re
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

# Plafond de pipelines simultanés. Depuis la paramétrisation des répertoires
# (adli_v2.pipeline.process_pdf passe interim/processed/annotated/md en
# arguments aux fonctions v1, plus aucune mutation de constantes globales),
# deux runs concurrents ne peuvent plus fuir leurs chemins l'un dans
# l'autre : ce plafond est donc un simple limiteur de RESSOURCES (CPU/RAM
# des deux sous-processus), pas une nécessité de correction. Chaque run
# tourne d'ailleurs dans son propre sous-processus (adli_v2.scripts.
# run_extraction) : état de module v1 isolé par construction.
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


_MD_SPECIAL_RE = re.compile(r"[*`]")

def _md_safe(text: str) -> str:
    """Neutralise les caractères à sens Markdown (*, `) dans du texte BRUT
    extrait du PDF (OCR, artefacts d'extraction) avant de l'insérer dans une
    réponse de chat formatée en Markdown. Sans ça, un astérisque isolé dans
    le texte source romprait l'appariement **/`*` côté client et ferait
    déraper la mise en forme du reste du message. Les marqueurs ** et * 
    insérés par CE module (autour des labels qu'on contrôle) ne passent
    jamais par cette fonction — seul le contenu issu du PDF y passe.
    """
    if not text:
        return text
    return _MD_SPECIAL_RE.sub("", text)


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

    # Action « Analyse en profondeur » : vue d'ensemble complète du document
    # (bouton de la page ou question équivalente) — synthèse LLM directe.
    if any(w in q for w in _ANALYSIS_DEEP_WORDS):
        return _llm_analysis_answer(data, question)

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
                shown = _md_safe(txt)[:600] + ("…" if len(txt) > 600 else "")
                previews.append(f"**Article {a.get('number','?')}** — {shown}")
        head = (f"**{exact.get('instrument_type') or '?'} {exact.get('reference','')}** — "
                f"**{exact.get('n_articles','?')} articles**, BO n°{bo}.")
        title = _md_safe(exact.get("title") or exact.get("reference_label") or "")
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
                shown = _md_safe(txt)[:600] + ("…" if len(txt) > 600 else "")
                lines.append(f"**Article {a.get('number','?')}** — {shown}")
            return f"Résultats pour « {question} » :\n\n" + "\n".join(lines)

    # Repli LLM : la cascade de règles ne couvre pas cette question — on
    # laisse le modèle analyser le contenu réel du document, avec vérification
    # mécanique des citations (aucune hallucination n'est jamais montrée).
    return _llm_analysis_answer(data, question)


# ── Analyse du contenu par LLM (repli au-delà de la cascade de règles) ──

# Budget de contexte par requête LLM pour l'analyse d'UN document. Groq
# renvoie HTTP 413 « request_too_large » bien avant la fenêtre de contexte
# nominative dès que le prompt dépasse la taille acceptée par son système
# compound (constaté : 413 à ~42k caractères de prompt sur BO_7488-bis,
# succès à ~11k). format_context tronque en dernier recours à
# max_context_chars + 2000 pour rester sous cette limite.
ANALYSIS_MAX_CONTEXT_CHARS = 9000

# Un BO complet peut dépasser 1 Mo de texte (BO_7488-bis : 89 instruments,
# 1224 articles) : une seule requête n'emporte qu'un tronçon tronqué du
# document — voire un refus 413. Pour la vue d'ensemble (« Analyse en
# profondeur »), les articles sont donc découpés en paquets séquentiels
# tenant chacun dans ANALYSIS_CHUNK_CONTEXT_CHARS, le modèle est interrogé
# paquet par paquet et les sections sont concaténées. Le nombre de paquets
# est plafonné pour contenir la latence (~3 s par appel) ; quand tout le
# document n'a pas tenu, une note finale signale la couverture réelle.
ANALYSIS_CHUNK_CONTEXT_CHARS = 9000
ANALYSIS_MAX_CHUNKS = 12

# Marge par article dans l'estimation de remplissage d'un paquet : en-tête
# « [Source i] » + métadonnées (BO, doc_id, type/référence, page) + séparateur.
_ARTICLE_META_OVERHEAD_CHARS = 200

# Sources citées listées au maximum en bas de chaque section (au-delà, « … »).
_ANALYSIS_SECTION_MAX_CITES = 10

_LLM_UNAVAILABLE_MSG = ("Je n'ai pas pu interroger le modèle de langage "
    "(clé API, réseau ou service indisponible). Réessayez.")

_ANALYSIS_OVERVIEW_WORDS = (
    "résumé", "resume", "synthèse", "synthese", "thème", "theme", "thèmes",
    "themes", "discute", "discuter", "porte sur", "décris", "décrire",
    "détail", "detail", "article par article", "traite", "de quoi",
    "à quoi", "summary", "about", "overview", "aperçu", "analyse",
)

_ANALYSIS_DEEP_WORDS = (
    "analyse en profondeur", "analyse complète", "décris en détail",
    "décris-moi en détail", "que traite ce document", "de quoi parle ce document",
)


def _articles_as_rag_sources(data: dict) -> list[dict]:
    """Convertit les articles du document analysé en sources RAG au format
    attendu par src/rag/prompt_builder.format_context().

    L'ordre suit l'ordre du document (numéro d'article), pas un score de
    pertinence : pour une analyse de contenu on veut VOIR le document dans
    son déroulement, pas un top-k sémantique. Chaque article est enrichi
    du doc_id / bo_number du document et, quand il appartient à un décret,
    du type et de la référence de ce décret (champs que format_context
    affiche dans l'en-tête de chaque source).
    """
    doc_id = data.get("doc_id", "")
    bo = data.get("bo_number", "")
    by_index: dict[int, dict] = {}
    for instr in data.get("instruments", []):
        for idx in instr.get("article_indices") or []:
            if isinstance(idx, int):
                by_index[idx] = instr
    out: list[dict] = []
    for i, a in enumerate(data.get("articles", [])):
        src = dict(a)
        src["article_number"] = a.get("number", a.get("article_number", ""))
        src["doc_id"] = doc_id
        src["bo_number"] = bo
        instr = by_index.get(i)
        if instr:
            src.setdefault("instrument_type", instr.get("instrument_type"))
            src.setdefault("reference", instr.get("reference"))
        out.append(src)
    return out


def _article_context_len(article: dict) -> int:
    """Taille estimée d'un article dans le bloc de contexte formaté
    (texte + en-tête de métadonnées + séparateur)."""
    text = article.get("text_clean") or article.get("text") or ""
    return len(text) + _ARTICLE_META_OVERHEAD_CHARS


def _plan_analysis_chunks(sources: list[dict]) -> tuple[list[list[dict]], int]:
    """Découpe les sources en paquets séquentiels tenant chacun dans
    ANALYSIS_CHUNK_CONTEXT_CHARS (estimation via _article_context_len).

    L'ordre du document est conservé et un article n'est jamais coupé en
    deux entre deux paquets ; un article isolé plus gros que le budget sera
    tronqué par le filet de sécurité de format_context. Le nombre de paquets
    est plafonné à ANALYSIS_MAX_CHUNKS : renvoie (paquets, articles_couverts),
    les articles au-delà du plafond étant exclus de la couverture.
    """
    chunks: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    covered = 0
    for src in sources:
        cost = _article_context_len(src)
        if current and used + cost > ANALYSIS_CHUNK_CONTEXT_CHARS:
            chunks.append(current)
            covered += len(current)
            current, used = [], 0
            if len(chunks) >= ANALYSIS_MAX_CHUNKS:
                break
        current.append(src)
        used += cost
    if current and len(chunks) < ANALYSIS_MAX_CHUNKS:
        chunks.append(current)
        covered += len(current)
    return chunks, covered


def _analysis_section_title(chunk: list[dict]) -> str | None:
    """En-tête de section d'après l'instrument du premier article du paquet."""
    first = chunk[0]
    itype = first.get("instrument_type")
    ref = first.get("reference")
    if ref:
        return f"{itype or 'Texte'} n° {ref}"
    if itype:
        return str(itype)
    return None


def _chunked_analysis_answer(lang: str, question: str,
                             sources: list[dict]) -> str:
    """Vue d'ensemble d'un document volumineux par paquets séquentiels.

    Chaque paquet produit une section ancrée (grounding vérifié contre les
    seuls articles du paquet — les numéros [Source N] restent donc valides),
    précédée du type/référence de l'instrument qui l'ouvre. Un paquet qui
    échoue (erreur LLM transitoire, réponse non ancrée) est sauté sans
    compromettre les suivants ; si aucun paquet ne répond, on retombe sur
    le refus hors périmètre (au moins un appel a abouti) ou le message
    générique d'échec LLM.
    """
    from src.rag.citation_verifier import parse_grounding, verify_grounding
    from src.rag.llm_client import LLMClient
    from src.rag.prompt_builder import (
        OUT_OF_SCOPE_SENTENCE_AR,
        OUT_OF_SCOPE_SENTENCE_FR,
        build_synthesis_prompt,
    )

    refusal = OUT_OF_SCOPE_SENTENCE_AR if lang == "ar" else OUT_OF_SCOPE_SENTENCE_FR
    chunks, covered = _plan_analysis_chunks(sources)

    try:
        llm = LLMClient()
    except Exception:
        return _LLM_UNAVAILABLE_MSG

    sections: list[str] = []
    llm_reached = False
    for chunk in chunks:
        system_instruction, user_prompt = build_synthesis_prompt(
            question, chunk, max_context_chars=ANALYSIS_CHUNK_CONTEXT_CHARS,
        )
        try:
            answer_text = llm.generate(system_instruction, user_prompt)
        except Exception:
            continue
        llm_reached = True
        clean, source_ids = parse_grounding(answer_text)
        grounded, _stats = verify_grounding(source_ids, chunk)
        if refusal in clean or not clean.strip() or not grounded:
            continue
        body = clean.strip()
        title = _analysis_section_title(chunk)
        if title:
            body = f"**{title}**\n\n{body}"
        cites = []
        for i in grounded[:_ANALYSIS_SECTION_MAX_CITES]:
            art = chunk[i - 1]
            page = art.get("pdf_page") or art.get("printed_page")
            ref = f"art. {art.get('article_number', '?')}" + (f", p. {page}" if page else "")
            cites.append(ref)
        if len(grounded) > _ANALYSIS_SECTION_MAX_CITES:
            cites.append("…")
        body += "\n\n📄 Sources : " + ", ".join(cites)
        sections.append(body)

    if not sections:
        return refusal if llm_reached else _LLM_UNAVAILABLE_MSG

    out = "\n\n".join(sections)
    if covered < len(sources):
        note_fr = (f"\n\n_Analyse partielle : {covered} premiers articles couverts "
                   f"sur {len(sources)} — précisez un décret ou un thème "
                   "pour aller plus loin._")
        note_ar = (f"\n\n_تحليل جزئي: أول {covered} مادة من أصل {len(sources)} — "
                   "اطرح سؤالاً محدداً عن مرسوم أو موضوع للمزيد._")
        out += note_ar if lang == "ar" else note_fr
    return out


def _llm_analysis_answer(data: dict, question: str) -> str:
    """Réponse LLM fondée sur le contenu réel du document analysé.

    Utilisée en repli quand la cascade de règles (comptages, signataires,
    références, article exact, recherche plein texte, classification de
    domaine) ne couvre pas la question. Réutilise TOUT le mécanisme
    anti-hallucination du chatbot RAG v1 : citations mot à mot vérifiées
    mécaniquement pour une question factuelle (src/rag/citation_verifier.py),
    ancrage par numéros de sources pour une vue d'ensemble — rien
    d'invérifiable n'est jamais montré à l'utilisateur.

    Vue d'ensemble sur un document volumineux (contexte total > budget) :
    bascule en mode fragmenté (_chunked_analysis_answer) — une seule requête
    n'emporterait qu'un tronçon tronqué du document, voire un refus 413 de
    l'API.
    """
    from groq import APIStatusError
    from src.rag.citation_verifier import (
        parse_citations,
        verify_citations,
        parse_grounding,
        verify_grounding,
    )
    from src.rag.llm_client import LLMClient
    from src.rag.prompt_builder import (
        OUT_OF_SCOPE_SENTENCE_AR,
        OUT_OF_SCOPE_SENTENCE_FR,
        REFUSAL_SENTENCE_AR,
        REFUSAL_SENTENCE_FR,
        UNSUPPORTED_SENTENCE_AR,
        UNSUPPORTED_SENTENCE_FR,
        build_prompt,
        build_synthesis_prompt,
    )

    lang = (data.get("lang") or "").lower()
    sources = _articles_as_rag_sources(data)

    low = question.lower().strip()
    is_overview = any(w in low for w in _ANALYSIS_OVERVIEW_WORDS)

    if is_overview:
        total_chars = sum(_article_context_len(a) for a in sources)
        if total_chars > ANALYSIS_MAX_CONTEXT_CHARS:
            return _chunked_analysis_answer(lang, question, sources)

    build = build_synthesis_prompt if is_overview else build_prompt
    system_instruction, user_prompt = build(
        question, sources, max_context_chars=ANALYSIS_MAX_CONTEXT_CHARS,
    )

    try:
        llm = LLMClient()
        if is_overview:
            answer_text = llm.generate(system_instruction, user_prompt)
        else:
            answer_text = llm.generate_with_citation_guarantee(
                system_instruction, user_prompt
            )
    except APIStatusError as e:
        # Erreur typée du SDK plutôt qu'un appariement de chaîne sur
        # str(e) : 413 = requête trop volumineuse pour l'API (payload ou
        # contexte). Ne survient plus en pratique pour la vue d'ensemble
        # (mode fragmenté ci-dessus) — filet de sécurité.
        if getattr(e, "status_code", None) == 413:
            return ("Le document est trop long pour une analyse en une seule "
                "requête. Réessayez avec une question plus ciblée "
                "(un décret, un article, un thème).")
        return _LLM_UNAVAILABLE_MSG
    except Exception:
        return _LLM_UNAVAILABLE_MSG
    
    refusal = OUT_OF_SCOPE_SENTENCE_AR if lang == "ar" else OUT_OF_SCOPE_SENTENCE_FR
    refusal_phrase = REFUSAL_SENTENCE_AR if lang == "ar" else REFUSAL_SENTENCE_FR
    unsupported = UNSUPPORTED_SENTENCE_AR if lang == "ar" else UNSUPPORTED_SENTENCE_FR

    if is_overview:
        clean, source_ids = parse_grounding(answer_text)
        grounded, _stats = verify_grounding(source_ids, sources)
        if refusal in clean or not clean.strip() or not grounded:
            return refusal
        cites = []
        for i in grounded:
            art = sources[i - 1]
            page = art.get("pdf_page") or art.get("printed_page")
            ref = f"art. {art.get('article_number', '?')}" + (f", p. {page}" if page else "")
            cites.append(ref)
        return clean + "\n\n📄 Sources : " + ", ".join(cites)

    clean, spans = parse_citations(answer_text)
    verified, _stats = verify_citations(spans, sources)
    if refusal_phrase in clean:
        return refusal_phrase
    if spans and not verified:
        return unsupported
    cites = []
    for c in verified:
        art = sources[c["source"] - 1]
        page = c.get("page") or art.get("pdf_page") or art.get("printed_page")
        if page:
            cites.append(f"p. {page}")
        else:
            cites.append(f"art. {art.get('article_number', '?')}")
    return clean + (f"\n\n📄 {', '.join(cites)}" if cites else "")


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