#!/usr/bin/env python3
"""Application web « ADLI Morocco » — point d'entrée unique.

Assemble les deux interfaces Flask en une seule application :
    /            → Chatbot RAG juridique (app/chat.py)
    /analyzer    → Analyseur de Bulletins Officiels (app/analyzer.py)

Usage :
    python -m app.main
    -> http://localhost:5000
"""
import sys
import threading
from pathlib import Path

import flask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.analyzer import analyzer_bp  # noqa: E402
from app.chat import chat_bp  # noqa: E402

app = flask.Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

app.register_blueprint(chat_bp)
app.register_blueprint(analyzer_bp)


def _preload_chatbot():
    from app.chat import preload_chatbot

    preload_chatbot()


if __name__ == "__main__":
    print("  -> http://localhost:5000")
    # Préchargement du chatbot en arrière-plan : le modèle d'embedding
    # (~1,1 Go) met ~27 s à se charger ; lancé en parallèle du bind du port,
    # la première page s'affiche instantanément (bandeau « chargement » si
    # besoin) au lieu de rester bloquée sur la construction synchrone.
    threading.Thread(target=_preload_chatbot, daemon=True).start()
    # use_reloader=False: with the reloader on, Werkzeug forks a second
    # process that re-imports everything (spaCy, camel-tools, etc.) before
    # actually binding the port, which delays startup further and can
    # produce a stale/duplicate process. Since lanceur_web.py already waits
    # on /health before opening the browser, we don't need the reloader.
    # debug=False: debug=True expose le débogueur interactif Werkzeug
    # (console Python exécutable à distance via les pages d'erreur) sur le
    # réseau — RCE dès que l'app est joignable hors localhost.
    app.run(use_reloader=False, host="0.0.0.0", port=5000, threaded=True)
