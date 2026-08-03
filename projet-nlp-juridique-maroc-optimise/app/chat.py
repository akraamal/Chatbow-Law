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
# Intervalle minimum entre deux tentatives d'initialisation après un échec :
# le latching permanent empêchait de récupérer sans redémarrer le process
# (même après avoir corrigé .env / construit l'index).  Un cooldown évite de
# re-tentrer le chargement du modèle (plusieurs Go) à chaque requête.
_CHATBOT_RETRY_COOLDOWN_SECONDS = 30.0


def get_chatbot():
    global _chatbot, _chatbot_error, _chatbot_error_at

    if _chatbot is not None:
        return _chatbot, None
    if _chatbot_error is not None and (
        time.time() - _chatbot_error_at < _CHATBOT_RETRY_COOLDOWN_SECONDS
    ):
        return None, _chatbot_error

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
    _, error = get_chatbot()
    return render_template("index.html", setup_error=error)


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
