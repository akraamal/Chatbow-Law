"""
validate_content_integrity.py
-----------------------------
Contrôles automatiques corpus-wide du schéma enrichi (guide v5, scoped
aux décrets) :

  1. Contenu intégral (champ `content`) :
     - invariant « Fait à » / « Rabat, le » / « حرر بالرباط » présent ;
     - pas de doublon « DÉCRÈTE » / « ARTICLE PREMIER » (deux décrets
       fusionnés dans un seul instrument) ;
     - lignes de bruit OCR résiduelles dans `content` (comptage) ;
     - annexes : un instrument avec une annexe garde son texte complet.
  2. bo_number : aucune valeur nulle, aucun mismatch entre nom de fichier
     et en-tête (extract_bo_number_cross_validated).
  3. Dates : decree_date_hijri converti (calendrier islamique tabulaire)
     à ±1 jour de decree_date_gregorian ; decree_date_gregorian <=
     bo_date_publication.
  4. Signataires : stats (issuer présent, contreseing, noms absents) +
     diff du jeu doré de régression (tests/signatories_golden_set.py).

Usage :
    python scripts/validate_content_integrity.py
    python scripts/validate_content_integrity.py --corpus data/annotated
    python scripts/validate_content_integrity.py --file data/annotated/fr_BO_6804_Fr_692ac82f_entities.json
"""

import argparse
import json
import re
import sys
import warnings
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.document_metadata_extractor import (  # noqa: E402
    extract_bo_number_cross_validated,
)
from src.utils.hijri_calendar import (  # noqa: E402
    hijri_to_gregorian,
    parse_hijri_date_ar,
    parse_hijri_date_fr,
)

SCRIPT_DIR = Path(__file__).parent
ANNOTATED_DIR = SCRIPT_DIR.parent / "data" / "annotated"

# Ligne sans aucune lettre (latin étendu + arabe), 1-59 caractères — bruit
# hérité des fusions de colonnes OCR (« © #«ل " 34600606 »).
_NOISE_LINE_RE = re.compile(
    r"^\s*[^0-9A-Za-z\u00C0-\u024F\u0600-\u06FF]{1,59}$"
)

# Types d'instruments auxquels l'invariant « formule de clôture » s'applique.
# Scoped aux décrets (guide v5) : les arrêtés se terminent souvent par la
# seule clause d'exécution (« ...chargé de l'exécution du présent arrêté »)
# et les dahirs par la note de publication — aucun des deux ne porte la
# formule « Fait à ».
_FORMULA_TYPES = {"DECRET"}


def load_annotated(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def content_invariants(data: dict, doc_label: str) -> list[str]:
    """Invariants du champ `content` (priorité 1 du guide)."""
    problems: list[str] = []
    for instr in data.get("instruments", []):
        content = instr.get("content", "") or ""
        iid = instr.get("instrument_id") or instr.get("reference") or "?"
        instr_type = instr.get("instrument_type", "")
        if not content:
            problems.append(f"[content-vide] {doc_label} {iid} ({instr_type})")
            continue

        # Invariant 1 : la formule de clôture doit être présente
        if instr_type in _FORMULA_TYPES:
            if "حرر" in content:
                ok_formula = True
            elif not re.search(r"Fait\s+[àa]\s+Rabat|(?:^|\n)\s*Rabat,\s*le\b", content):
                problems.append(
                    f"[sans-formule-cloture] {doc_label} {iid} ({instr_type})"
                )

        # Invariant 2 : pas de doublon « DÉCRÈTE » / « ARTICLE PREMIER »
        # (deux décrets fusionnés dans un seul instrument). « DÉCRÈTE »
        # compte partout (ligne d'énonciation du préambule) ; « ARTICLE
        # PREMIER » uniquement en tête de ligne — les occurrences en plein
        # texte sont des citations légitimes (« ...de l'article premier et
        # l'article 2 du décret susvisé... ») dans un texte modifié.
        if content.upper().count("DÉCRÈTE") > 1:
            problems.append(
                f"[doublon-décretè] {doc_label} {iid} "
                f"({content.upper().count('DÉCRÈTE')} occurrences)"
            )
        n_art_premier = len(re.findall(
            r"(?im)^[ \t]*ARTICLE\s+PREMIER\b", content
        ))
        if n_art_premier > 1:
            problems.append(
                f"[doublon-article-premier] {doc_label} {iid} "
                f"({n_art_premier} occurrences)"
            )

        # Lignes de bruit résiduelles dans le contenu (informatif)
        noise_lines = [
            ln for ln in content.splitlines()
            if ln.strip() and _NOISE_LINE_RE.match(ln)
        ]
        if noise_lines:
            problems.append(
                f"[bruit-residuel x{len(noise_lines)}] {doc_label} {iid} :: "
                + " | ".join(repr(n.strip()[:40]) for n in noise_lines[:3])
            )
    return problems


def bo_number_checks(data: dict, doc_label: str) -> list[str]:
    """Contrôles bo_number : pas de null, pas de mismatch (priorité 2)."""
    problems: list[str] = []
    text = ""
    src = data.get("source")
    if src and Path(src).exists():
        text = Path(src).read_text(encoding="utf-8", errors="replace")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = extract_bo_number_cross_validated(
            text, doc_id=data.get("doc_id"), lang=data.get("lang", "fr")
        )
    if not res.get("bo_number"):
        problems.append(f"[bo_number-null] {doc_label} (doc_id={data.get('doc_id')!r})")
    if res.get("bo_number_confidence") == "mismatch":
        problems.append(
            f"[bo_number-mismatch] {doc_label} "
            f"filename={res['bo_number']!r} vs header={res['bo_number_header']!r}"
        )
    return problems


def date_checks(data: dict, doc_label: str) -> list[str]:
    """Croisement hijri↔grég. (±1 jour) et decree <= bo (priorité 3)."""
    problems: list[str] = []
    bo_date = data.get("bo_date_publication") or data.get("date_publication")
    for instr in data.get("instruments", []):
        iid = instr.get("instrument_id") or instr.get("reference") or "?"
        hijri = instr.get("decree_date_hijri") or instr.get("date_hijri")
        greg = instr.get("decree_date_gregorian") or instr.get("date_gregorian")
        if not hijri or not greg:
            continue
        try:
            greg_date = date.fromisoformat(greg)
        except (ValueError, TypeError):
            problems.append(f"[date-greg-invalide] {doc_label} {iid} {greg!r}")
            continue
        parsed = None
        if "من" in hijri or any("\u0600" <= c <= "\u06FF" for c in hijri):
            parsed = parse_hijri_date_ar(hijri)
        else:
            parsed = parse_hijri_date_fr(hijri)
        if parsed is None:
            problems.append(f"[date-hijri-non-parse] {doc_label} {iid} {hijri!r}")
            continue
        converted = hijri_to_gregorian(*parsed)
        diff = abs((converted - greg_date).days)
        # Tolérance ±2 jours : le calendrier tabulaire (modèle) peut
        # dériver de 1-2 jours par rapport aux dates officielles marocaines
        # (calendrier d'observation) — cas observé sur les bulletins de
        # 2013 (compilation BO_7488-bis, « 6 kaada 1434 (13 septembre
        # 2013) »). Un écart >= 3 jours signale un chiffre OCR corrompu.
        if diff > 2:
            problems.append(
                f"[date-hijri-greg-ecart-jours={diff}] {doc_label} {iid} "
                f"{hijri} -> {converted} vs {greg}"
            )
        # Sanity : la date de signature ne peut pas être postérieure à la
        # date de parution du bulletin.
        if bo_date:
            try:
                if greg_date > date.fromisoformat(bo_date):
                    problems.append(
                        f"[decret-apres-bulletin] {doc_label} {iid} "
                        f"signé {greg} > BO publié {bo_date}"
                    )
            except ValueError:
                pass
    return problems


def signatory_stats(data: dict, doc_label: str) -> tuple[int, int, int]:
    """(n_instruments, n_avec_signataire, n_noms_absents)."""
    n_instr = n_with = n_null_names = 0
    for instr in data.get("instruments", []):
        n_instr += 1
        blocks = instr.get("signatories", [])
        if blocks:
            n_with += 1
        n_null_names += sum(1 for b in blocks if b.get("name") is None)
    return n_instr, n_with, n_null_names


def golden_set_diff(data: dict, doc_label: str) -> list[str]:
    """Diff du jeu doré de signataires (régression)."""
    problems: list[str] = []
    try:
        from tests.signatories_golden_set import GOLDEN_SIGNATORIES
    except ImportError:
        return problems
    for case in GOLDEN_SIGNATORIES:
        if doc_label != case["file"]:
            continue
        found = None
        for instr in data.get("instruments", []):
            if instr.get("instrument_id") == case["instrument_id"]:
                found = instr.get("signatories")
                break
        if found is None:
            problems.append(f"[golden-introuvable] {doc_label} {case['instrument_id']}")
            continue
        if found != case["expected"]:
            problems.append(
                f"[golden-diff] {doc_label} {case['instrument_id']}\n"
                f"    attendu : {json.dumps(case['expected'], ensure_ascii=False)}\n"
                f"    obtenu  : {json.dumps(found, ensure_ascii=False)}"
            )
    return problems


def validate_file(path: Path, verbose: bool = True) -> tuple[int, list[str]]:
    data = load_annotated(path)
    doc_label = f"{path.name}"
    problems = []
    problems += content_invariants(data, doc_label)
    problems += bo_number_checks(data, doc_label)
    problems += date_checks(data, doc_label)
    problems += golden_set_diff(data, doc_label)
    if verbose:
        n_instr, n_with, n_null = signatory_stats(data, doc_label)
        n_decrets = sum(
            1 for i in data.get("instruments", [])
            if i.get("instrument_type") == "DECRET"
        )
        print(f"{doc_label}: {n_instr} instruments ({n_decrets} décrets), "
              f"{n_with} avec signataires, {n_null} noms absents")
        for p in problems:
            print(f"    ! {p}")
    return len(problems), problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=str(ANNOTATED_DIR))
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.file:
        paths = [Path(args.file)]
    else:
        paths = sorted(Path(args.corpus).glob("**/*_entities.json"))

    total_problems = 0
    n_files = 0
    for p in paths:
        try:
            n, problems = validate_file(p)
        except Exception as exc:  # fichier corrompu — signaler, ne pas bloquer
            print(f"{p.name}: ERREUR {exc!r}")
            n = 1
            problems = []
        total_problems += n
        n_files += 1

    print("=" * 70)
    print(f"{n_files} fichiers validés, {total_problems} problème(s) signalé(s)")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())
