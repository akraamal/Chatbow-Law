#!/usr/bin/env python3
"""Lanceur de l'interface web ADLI Morocco (chatbot RAG + analyseur de BO)."""
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://localhost:5000"


def _wait_for_server_then_open_browser(timeout: float = 60.0):
    """Poll /health until the Flask server responds, then open the browser.

    Without this, the browser tab is opened immediately while Flask/spaCy
    are still importing and the port isn't bound yet, which produces a
    connection error on the very first load.
    """
    deadline = time.time() + timeout
    print("  ⏳ Attente du démarrage du serveur...")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=2) as resp:
                if resp.status == 200:
                    print("  ✅ Serveur prêt")
                    webbrowser.open(URL)
                    return
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    print(f"  ⚠️  Le serveur met du temps à démarrer. "
          f"Ouvrez {URL} manuellement une fois prêt.")


print("  Lancement de l'interface web...")
print(f"  {URL}")

# Watch for the server to come up in the background; open the browser only
# once it actually answers, instead of racing the subprocess below.
watcher = threading.Thread(target=_wait_for_server_then_open_browser, daemon=True)
watcher.start()

# Run the Flask app as a module so imports resolve correctly
subprocess.run([sys.executable, "-m", "app.main"], cwd=str(ROOT))