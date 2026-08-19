"""
adli_v2.app.main
----------------
Application Flask v2 — deux interfaces :
    /            → Chatbot RAG v1 réutilisé tel quel (app/chat de l'ancien
                   projet, chargé en lecture seule) ;
    /analyzer    → Analyseur v2 centré décret (upload, métadonnées,
                   instruments décrets en premier, articles complets,
                   fréquences de mots-clés).

Usage (depuis la racine du dépôt) :
    python -m adli_v2.app.main
    -> http://localhost:5001
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flask  # noqa: E402

from app.chat import chat_bp  # noqa: E402 — chatbot RAG v1 (lecture seule)
from adli_v2.app.analyzer import analyzer_bp  # noqa: E402

app = flask.Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Le blueprint chat v1 pointe vers les templates/static de l'ancien app/ :
# on réutilise ses assets sans les dupliquer.  Les templates des blueprints
# v2 (analyzer_v2.html) restent résolus via leur propre dossier.
app.template_folder = str(REPO_ROOT / "app" / "templates")
app.static_folder = str(REPO_ROOT / "app" / "static")

app.register_blueprint(chat_bp)
app.register_blueprint(analyzer_bp)


def _preload_chatbot():
    from app.chat import preload_chatbot

    preload_chatbot()


if __name__ == "__main__":
    print("  -> http://localhost:5001  (/, /analyzer)")
    threading.Thread(target=_preload_chatbot, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=False)