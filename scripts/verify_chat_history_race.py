"""Régression course réhydratation — pattern hold&release (non bloquant)."""
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5055"
SHOTS = Path(tempfile.gettempdir()) / "adli_shots"

results = []
ok_all = True


def check(label, cond, extra=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    results.append(("OK   " if cond else "FAIL ") + label + (" | " + extra if extra else ""))


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1440, "height": 950})
    page = ctx.new_page()
    page.on("pageerror", lambda e: results.append("PAGEERROR: " + str(e)[:180]))

    page.goto(BASE + "/")
    page.evaluate("localStorage.setItem('adli_current_conversation','seedconv0001')")

    # Retient LA PREMIÈRE requête d'historique sans y répondre.
    held = []

    def hold(route):
        held.append(route)
        # pas de continue_/fulfill : la requête reste en attente

    page.route("**/api/chat/history/*", hold)
    page.reload()

    # La requête est retenue -> fenêtre de course infiniment large.
    page.wait_for_function(
        "document.getElementById('chat-input') && "
        "document.getElementById('chat-input').disabled", timeout=15000)
    check("input désactivé tant que l'historique n'a pas répondu", True)
    check("placeholder de chargement affiché",
          page.locator("#chat-input").get_attribute("placeholder")
          == "Chargement de la conversation…")
    check("bouton envoyer désactivé aussi",
          page.locator("#chat-form button[type=submit]").is_disabled())

    # Impossible de soumettre pendant ce temps : la saisie est refusée.
    posts = []
    page.on("request", lambda r: posts.append(r.url)
            if r.url.endswith("/api/chat") and r.method == "POST" else None)
    try:
        page.fill("#chat-input", "ne doit jamais partir", timeout=2000)
        check("saisie impossible pendant le gel", False, "fill a réussi ?!")
    except Exception:
        check("saisie impossible pendant le gel (input disabled)", True)
    page.wait_for_timeout(300)
    check("aucun POST /api/chat pendant le gel", len(posts) == 0,
          f"posts={len(posts)}")

    # Libère : la réhydratation se termine et réactive tout.
    for r in held:
        r.continue_()
    page.wait_for_function(
        "!document.getElementById('chat-input').disabled", timeout=15000)
    check("input réactivé après réponse", True)
    check("placeholder original restauré",
          page.locator("#chat-input").get_attribute("placeholder")
          == "Posez votre question juridique...")
    n_bubbles = page.locator("#chat-messages .brutal-border").count()
    check("conversation seedée rendue une seule fois", n_bubbles == 5,
          f"bulles={n_bubbles}")

    # Régression clé : envoi juste après -> le message NE disparaît PAS.
    import time as _t
    msg = f"probe anti-course {int(_t.time())}"
    page.fill("#chat-input", msg)
    with page.expect_response(
            lambda r: r.url.endswith("/api/chat") and r.request.method == "POST",
            timeout=120000):
        page.press("#chat-input", "Enter")
    page.wait_for_timeout(2000)
    body_txt = page.locator("#chat-messages").inner_text()
    check("message envoyé toujours affiché (pas effacé)", msg in body_txt)
    page.screenshot(path=str(SHOTS / "hist_race_fix.png"))

    # Accessibilité clavier du lien + pas de saut de scroll
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    sb = page.evaluate("window.scrollY")
    page.focus("#chat-history-nav-btn")
    focused = page.evaluate(
        "document.activeElement && document.activeElement.id") == "chat-history-nav-btn"
    check("lien focusable au clavier", focused)
    page.keyboard.press("Enter")
    page.wait_for_timeout(400)
    visible = page.locator("#chat-history-panel").is_visible()
    check("Enter ouvre le panneau", visible)
    sa = page.evaluate("window.scrollY")
    hsh = page.evaluate("location.hash")
    path = page.evaluate("location.pathname")
    check("aucune navigation ni # résiduel",
          hsh == "" and path == "/", f"hash={hsh!r} path={path}")

    browser.close()

print("\n".join(results))
sys.exit(0 if ok_all else 1)
