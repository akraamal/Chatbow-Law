"""E2E tri « Analyses précédentes » — assertions scopées sur les cartes seed."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5055"
ANNOTATED = Path(r"C:\Users\yahia\Downloads\projet-nlp-juridique-maroc-optimise-v5"
                 r"\adli-v2\data\annotated")
SHOTS = Path(tempfile.gettempdir()) / "adli_shots"
SHOTS.mkdir(exist_ok=True)

results = []
ok_all = True


def check(label, cond, extra=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    results.append(("OK   " if cond else "FAIL ") + label + (" | " + extra if extra else ""))


def seed_doc(num: str, bo: str, created_at=None):
    doc_id = f"E2E_SORT_{num}_seed"
    data = {
        "doc_id": doc_id, "lang": "fr", "bo_number": bo,
        "date_publication": "2026-01-01",
        "instruments": [{"instrument_type": "DECRET",
                         "reference": f"2-26-{num}", "n_articles": 1,
                         "article_indices": [0]}],
        "articles": [{"number": "1", "text": f"Texte {num}.", "pdf_page": 3}],
    }
    if created_at is not None:
        data["created_at"] = created_at
    p = ANNOTATED / f"{doc_id}_entities.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


seeded = []
try:
    pa = seed_doc("1", "1")                    # mtime la plus ancienne
    pb = seed_doc("2", "2")
    pc = seed_doc("10", "10", created_at=time.time())  # JSON prioritaire
    os.utime(pa, (1000000000, 1000000000))
    os.utime(pb, (int(time.time()) - 500,) * 2)
    os.utime(pc, (1000000005, 1000000005))
    seeded += [pa, pb, pc]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 950})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)[:150]))

        def goto_history():
            page.goto(BASE + "/analyzer")
            page.wait_for_selector("#historyList .history-open", timeout=30000)
            page.wait_for_timeout(250)

        def e2e_nums():
            """Numéros BO des cartes seedées, dans l'ordre affiché."""
            titles = page.locator("#historyList div.mb-2").all_inner_texts()
            out = []
            for t in titles:
                if "E2E_SORT" not in t:
                    continue
                out.append(t.split("BO n° ")[1].split(" ")[0])
            return out

        def select(mode):
            page.select_option("#history-sort", mode)
            page.wait_for_timeout(200)

        goto_history()
        check("défaut date-desc: 10(récent), 2, 1(ancien)", e2e_nums() == ["10", "2", "1"],
              str(e2e_nums()))
        first_txt = page.locator("#historyList div.mb-2", has_text="E2E_SORT") \
            .first.inner_text()
        check("libellé «Analysé le» présent", "Analysé le" in first_txt,
              first_txt.replace("\n", " ")[:90])
        unk = page.evaluate("formatAnalysisDate(undefined)")
        check("formatAnalysisDate(undefined) -> libellé dédié",
              "inconnue" in unk, unk)
        pub_kept = "2026-01-01" in first_txt or "2026-01-" in first_txt
        check("date_publication toujours affichée séparément", pub_kept)

        select("date-asc")
        check("date-asc: 1, 2, 10", e2e_nums() == ["1", "2", "10"], str(e2e_nums()))

        select("name-asc")
        check("name-asc naturel: 1, 2, 10", e2e_nums() == ["1", "2", "10"],
              str(e2e_nums()))

        select("name-desc")
        check("name-desc: 10, 2, 1", e2e_nums() == ["10", "2", "1"], str(e2e_nums()))
        page.screenshot(path=str(SHOTS / "sort_namedesc.png"))

        # G — Voir sur la PREMIÈRE carte E2E sous name-desc (= doc 10)
        page.locator("#historyList div.mb-2", has_text="E2E_SORT") \
            .first.locator(".history-open").click()
        page.wait_for_selector("#btnDeepAnalysis", state="visible", timeout=60000)
        st = page.locator("#chatStatus").inner_text()
        check("Voir ouvre le bon document après tri",
              "E2E_SORT_10_seed" in st, st)

        # H — Nouveau chat sur la première carte E2E sous date-asc (= doc 1)
        goto_history()
        select("date-asc")
        page.locator("#historyList div.mb-2", has_text="E2E_SORT") \
            .first.locator(".history-chat").click()
        page.wait_for_selector("#btnDeepAnalysis", state="visible", timeout=60000)
        st = page.locator("#chatStatus").inner_text()
        check("Nouveau chat ouvre le bon document après tri",
              "E2E_SORT_1_seed" in st, st)

        # Aucun re-fetch au changement de tri : compter les appels /analyses
        counts_before = []
        page.on("request", lambda r: counts_before.append(1)
                if r.url.endswith("/analyses") else None)
        select("name-asc")
        select("date-desc")
        check("changement de tri sans re-fetch", len(counts_before) == 0,
              f"fetches={len(counts_before)}")
        check("aucune erreur console/page", not errors, "; ".join(errors)[:120])

        browser.close()
finally:
    for pth in seeded:
        try:
            pth.unlink()
        except OSError:
            pass
print("\n".join(results))
print("seed nettoyé:", len(seeded))
sys.exit(0 if ok_all else 1)
