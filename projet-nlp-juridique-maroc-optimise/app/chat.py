"""
app/chat.py
--------------
Chatbot RAG « ADLI Morocco » : routes Flask du chat juridique.
Interface : app/templates/index.html + app/static/js/chat.js

Nécessite :
    - un index de recherche déjà construit : python -m scripts.build_search_index
    - la variable d'environnement GROQ_API_KEY définie

Si l'un des deux manque, l'app démarre quand même : elle affiche un
bandeau d'avertissement clair plutôt que de planter, et /api/chat renvoie
une erreur JSON explicite au lieu d'un 500 muet.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

chat_bp = Blueprint("chat", __name__)

_chatbot = None
_chatbot_error: str | None = None
_chatbot_error_at: float = 0.0
# Verrou de construction : sans lui, deux requêtes parallèles pendant le
# premier chargement (page + JS) déclencheraient CHACUNE la construction
# du chatbot (~27 s de modèle d'embedding) en double.
_chatbot_lock = threading.Lock()
# Verrou d'inférence : le modèle d'embedding n'est pas garanti thread-safe ;
# sur une machine CPU, deux questions simultanées doivent se sérialiser
# (au lieu de charger la RAM/le CPU deux fois par requête).
_inference_lock = threading.Lock()
# Intervalle minimum entre deux tentatives d'initialisation après un échec :
# le latching permanent empêchait de récupérer sans redémarrer le process
# (même après avoir corrigé .env / construit l'index).  Un cooldown évite de
# re-tentrer le chargement du modèle (plusieurs Go) à chaque requête.
_CHATBOT_RETRY_COOLDOWN_SECONDS = 30.0


def preload_chatbot():
    """Charge le chatbot en tâche de fond au démarrage du serveur.

    Le modèle d'embedding (~1,1 Go, torch + sentence-transformers) met
    ~27 s à se charger sur cette machine : sans ce préchargement, le
    PREMIER GET / bloque la page aussi longtemps (get_chatbot() est
    synchrone). Le préchargement le fait en parallèle du bind du port :
    le navigateur ouvre une page déjà rendue, avec un bandeau « chargement
    en cours » si le modèle n'est pas encore prêt.
    """
    get_chatbot()


def chatbot_status() -> tuple[str, str | None]:
    """('ready'|'loading'|'error', message_erreur) — sans jamais construire
    le chatbot (l'appelé décide s'il affiche un bandeau ou une erreur)."""
    if _chatbot is not None:
        return "ready", None
    if _chatbot_error is not None:
        return "error", _chatbot_error
    return "loading", None


def get_chatbot():
    global _chatbot, _chatbot_error, _chatbot_error_at

    if _chatbot is not None:
        return _chatbot, None
    if _chatbot_error is not None and (
        time.time() - _chatbot_error_at < _CHATBOT_RETRY_COOLDOWN_SECONDS
    ):
        return None, _chatbot_error

    with _chatbot_lock:
        # Double-checked : un autre thread (ex. le préchargement) a pu
        # finir de construire pendant qu'on attendait le verrou.
        if _chatbot is not None:
            return _chatbot, None
        try:
            from src.rag.chatbot import LegalRAGChatbot
            _chatbot = LegalRAGChatbot()
            _chatbot_error = None
            return _chatbot, None
        except Exception as e:
            _chatbot = None
            _chatbot_error = (
                f"{e} — vérifie que l'index existe "
                f"(python -m scripts.build_search_index) et que GROQ_API_KEY "
                f"est définie dans l'environnement."
            )
            _chatbot_error_at = time.time()
            return None, _chatbot_error


@chat_bp.route("/")
def index():
    # N'appelle PAS get_chatbot() : la construction du modèle (~27 s) ne
    # doit jamais bloquer le rendu de la page — un bandeau informe.
    _, error = chatbot_status()
    return render_template(
        "index.html",
        setup_error=error,
        model_loading=(_chatbot is None and error is None),
    )


@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    # Plafond serveur (défense en profondeur, même si le client est bridé) :
    # un historique interminable renchérirait chaque reformulation et
    # offrirait une surface d'injection inutile.
    history = (data.get("history") or [])[-8:]
    lang = data.get("lang") or None

    if not query:
        return jsonify({"error": "La question est vide."}), 400

    bot, error = get_chatbot()
    if error:
        return jsonify({"error": error}), 503

    try:
        # Sérialise les questions simultanées (embedding non thread-safe).
        with _inference_lock:
            result = bot.answer(query, history=history, lang=lang)
    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement de la question : {e}"}), 500

    return jsonify(result)


@chat_bp.route("/download/<doc_id>")
def download_source(doc_id):
    """Sert le PDF source correspondant à doc_id, si présent dans data/raw/."""
    safe_id = Path(doc_id).name
    for lang_dir in ("fr", "ar", ""):
        candidate = PROJECT_ROOT / "data" / "raw" / lang_dir / f"{safe_id}.pdf"
        if candidate.exists():
            return send_file(candidate, as_attachment=True, download_name=candidate.name)
    abort(404, description=f"PDF source introuvable pour {safe_id}")
