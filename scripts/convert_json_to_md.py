"""
convert_json_to_md.py
---------------------
Convertit un fichier JSON annoté (sortie de etape4_pipeline) en Markdown.

Deux modes :

  --mode rag   (par défaut) : markdown propre destiné à l'EMBEDDING /
      l'indexation RAG. Pas de balises `[TYPE]` dans le texte courant
      (elles polluent l'embedding sans ajouter de signal sémantique),
      entités/dates dédupliquées et condensées en quelques lignes au
      lieu d'un tableau complet répété à chaque article, tableaux
      structurés (deduplicated_tables) rendus en vrais tableaux
      Markdown au lieu d'être absents ou aplatis.

  --mode audit : comportement proche de l'original — texte annoté avec
      des marqueurs `**[TYPE]**` inline et tableaux détaillés par
      article — pour la relecture humaine, PAS pour l'indexation
      (ce mode produit un fichier bien plus lourd et redondant).

Usage :
    python scripts/convert_json_to_md.py data/annotated/fr_BO_7510_Fr_entities.json
    python scripts/convert_json_to_md.py data/annotated/fr_BO_7510_Fr_entities.json -o rapport.md
    python scripts/convert_json_to_md.py data/annotated/fr_BO_7510_Fr_entities.json --mode audit
"""

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

# ── Nettoyage de texte ───────────────────────────────────────────────────

# Artefacts isolés (accolades orphelines issues de tableaux mal détourés,
# astérisques de séparation "* * *") : une ligne composée UNIQUEMENT de ce
# genre de caractères n'apporte rien au texte narratif ni à l'embedding.
_ARTIFACT_LINE_RE = re.compile(r"^[\s{}*]+$", re.MULTILINE)


def _strip_artifacts(text: str) -> str:
    if not text:
        return text
    text = _ARTIFACT_LINE_RE.sub("", text)
    # Collapse les lignes vides multiples laissées par le retrait ci-dessus
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_cell(text: str) -> str:
    """Nettoie une cellule de tableau pour un rendu Markdown sûr."""
    text = (text or "").replace("\n", " ").replace("|", "/").strip()
    return text if text else " "


def _fmt_label(label: str) -> str:
    """Affiche un label d'entité en français lisible."""
    mapping = {
        "LOI": "Loi",
        "DECRET": "Décret",
        "DAHIR": "Dahir",
        "ARRETE": "Arrêté",
        "MINISTERE": "Ministère",
        "DATE_HIJRI": "Date (hégirienne)",
        "DATE_GREGORIAN": "Date (grégorienne)",
        "BULLETIN_OFFICIEL": "Bulletin officiel",
        "PERSON": "Personne",
        "ORG": "Organisation",
        "CITATION": "Citation",
    }
    return mapping.get(label, label)


# ── Regroupement / déduplication (mode rag) ──────────────────────────────

def _group_by_label(items: list[dict]) -> "OrderedDict[str, OrderedDict[str, int]]":
    """
    Regroupe une liste d'entités {label, text} par label, en dédupliquant
    et comptant les occurrences du même texte au sein d'un même article
    (ex : "ministre de l'agriculture" apparaît souvent 2-3 fois dans le
    même article — inutile de le répéter 3 fois dans le rendu).
    """
    groups: "OrderedDict[str, OrderedDict[str, int]]" = OrderedDict()
    for item in items:
        label = _fmt_label(item.get("label", "?"))
        text = item.get("text", "").replace("\n", " ").strip()
        if not text:
            continue
        groups.setdefault(label, OrderedDict())
        groups[label][text] = groups[label].get(text, 0) + 1
    return groups


def _render_grouped_compact(groups: "OrderedDict[str, OrderedDict[str, int]]") -> list[str]:
    """Rend un regroupement label->texte->compte en quelques lignes compactes."""
    lines = []
    for label, texts in groups.items():
        parts = []
        for text, count in texts.items():
            parts.append(f"{text} (×{count})" if count > 1 else text)
        lines.append(f"- **{label}** : " + " ; ".join(parts))
    return lines


def _render_grouped_inline(groups: "OrderedDict[str, OrderedDict[str, int]]") -> str:
    """Comme _render_grouped_compact, mais sur une seule ligne (pas de
    liste à puces) — utilisé pour la ligne "Références" en mode rag."""
    segments = []
    for label, texts in groups.items():
        parts = [f"{t} (×{c})" if c > 1 else t for t, c in texts.items()]
        segments.append(f"**{label}** : " + " ; ".join(parts))
    return " | ".join(segments)


def _dedupe_simple(items: list[dict]) -> "OrderedDict[str, int]":
    """Déduplique une liste plate de {text} (personnes, organisations)."""
    out: "OrderedDict[str, int]" = OrderedDict()
    for item in items:
        text = item.get("text", "").replace("\n", " ").strip()
        if not text:
            continue
        out[text] = out.get(text, 0) + 1
    return out


def _render_citations_compact(citations: list[dict]) -> list[str]:
    lines = []
    for c in citations:
        ct = c.get("text", "").replace("\n", " ").strip()
        target = (c.get("target_text") or "").strip()
        resolved = c.get("resolved", False)
        status = "résolue" if resolved else "non résolue"
        if not target or target == ct:
            lines.append(f"- {ct} _( {status} )_")
        else:
            lines.append(f"- {ct} → {target} _( {status} )_")
    return lines


def _rows_to_md_table(rows: list[list[str]]) -> str:
    """Convertit des lignes de tableau (list de list de cellules) en un
    tableau Markdown propre. Rend les tableaux extraits séparément
    (deduplicated_tables) au lieu de les laisser aplatis dans le texte
    ou absents du rendu final."""
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    header = [_clean_cell(c) for c in rows[0]] + [" "] * (n_cols - len(rows[0]))
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * n_cols) + "|"]
    for row in rows[1:]:
        cells = [_clean_cell(c) for c in row] + [" "] * (n_cols - len(row))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Mode audit (comportement d'origine, pour relecture humaine) ─────────

def _mark_entities(text: str, entities: list[dict]) -> str:
    """Insère des marqueurs `**[TYPE]** ...` autour des entités dans le texte."""
    if not entities:
        return text
    sorted_ents = sorted(
        [e for e in entities if "start_char" in e and "end_char" in e],
        key=lambda e: e["start_char"],
    )
    if not sorted_ents:
        return text
    parts, pos = [], 0
    for e in sorted_ents:
        s, end = e["start_char"], e["end_char"]
        if s < pos:
            continue
        if s > pos:
            parts.append(text[pos:s])
        label = _fmt_label(e.get("label", "?"))
        parts.append(f"**[{label}]** {text[s:end]}")
        pos = end
    if pos < len(text):
        parts.append(text[pos:])
    return "".join(parts)


# ── Conversion ────────────────────────────────────────────────────────

def convert(json_path: str, output_path: str | None = None, mode: str = "rag") -> str:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    md = []

    # ── En-tête du document ──────────────────────────────────────────
    doc_id = data.get("doc_id", "?")
    md.append(f"# {doc_id}")
    md.append("")
    md.append("| Champ | Valeur |")
    md.append("|-------|--------|")
    md.append(f"| **BO n°** | {data.get('bo_number', '?')} |")
    md.append(f"| **Date publication** | {data.get('date_publication', '?')} |")
    md.append(f"| **Édition** | {data.get('edition_label', '')} |")
    md.append(f"| **Articles** | {data.get('n_articles', len(data.get('articles', [])))} |")
    md.append("")

    # ── Préambule ────────────────────────────────────────────────────
    preamble = _strip_artifacts(data.get("preamble_text", ""))
    preamble_entities = data.get("preamble_entities", [])
    md.append("---")
    md.append("## Préambule")
    md.append("")
    if preamble:
        if mode == "audit":
            md.append(_mark_entities(preamble, preamble_entities))
        else:
            md.append(preamble)
        md.append("")
    if preamble_entities:
        if mode == "audit":
            md.append("### Entités du préambule")
            md.append("")
            md.append("| # | Type | Texte |")
            md.append("|---|------|-------|")
            for i, e in enumerate(preamble_entities, 1):
                md.append(f"| {i} | {_fmt_label(e.get('label', '?'))} | `{e.get('text', '').strip()}` |")
            md.append("")
        else:
            md.extend(_render_grouped_compact(_group_by_label(preamble_entities)))
            md.append("")

    # ── Tableaux (rendus séparément — jamais aplatis dans le texte) ──
    tables = data.get("deduplicated_tables", data.get("tables", []))
    if tables:
        md.append("---")
        md.append("## Tableaux")
        md.append("")
        for i, t in enumerate(tables, 1):
            md.append(f"**Tableau {i}** (page {t.get('page_number', '?')})")
            md.append("")
            md.append(_rows_to_md_table(t.get("rows", [])))
            md.append("")

    # ── Articles ──────────────────────────────────────────────────────
    articles = data.get("articles", [])
    for idx, art in enumerate(articles, 1):
        num = art.get("number", f"? ({idx})")
        header = art.get("raw_header", "")

        md.append("---")
        md.append(f"## Article {num}")
        if header and header != f"Article {num}":
            md.append(f"*Marqueur : `{header}`*")
        md.append("")

        text = _strip_artifacts(art.get("text", ""))
        entities = art.get("entities", [])
        dates = art.get("dates", [])
        citations = art.get("citations", [])
        persons = art.get("persons", [])
        organizations = art.get("organizations", [])

        if text:
            md.append(_mark_entities(text, entities) if mode == "audit" else text)
            md.append("")

        if mode == "audit":
            if entities:
                md.append("### Entités juridiques")
                md.append("")
                md.append("| # | Type | Texte |")
                md.append("|---|------|-------|")
                for i, e in enumerate(entities, 1):
                    md.append(f"| {i} | {_fmt_label(e.get('label', '?'))} | `{e.get('text', '').strip()}` |")
                md.append("")
            if dates:
                md.append("### Dates")
                md.append("")
                for d in dates:
                    md.append(f"- {_fmt_label(d.get('label', ''))} : `{d.get('text', '').strip()}`")
                md.append("")
            if persons:
                md.append("### Personnes")
                md.append("")
                for p in persons:
                    md.append(f"- `{p.get('text', '').strip()}`")
                md.append("")
            if organizations:
                md.append("### Organisations")
                md.append("")
                for o in organizations:
                    md.append(f"- `{o.get('text', '').strip()}`")
                md.append("")
        else:
            # Mode rag : entités + dates regroupées et dédupliquées en
            # quelques lignes, pas un tableau complet répété à chaque
            # article (souvent les mêmes 4-5 références citées 2-3 fois).
            combined_groups = _group_by_label(entities + dates)
            if combined_groups:
                md.append("**Références** : " + _render_grouped_inline(combined_groups))
                md.append("")
            for label, items in (("Personnes", persons), ("Organisations", organizations)):
                deduped = _dedupe_simple(items)
                if deduped:
                    rendered = " ; ".join(
                        f"{t} (×{c})" if c > 1 else t for t, c in deduped.items()
                    )
                    md.append(f"**{label}** : {rendered}")
                    md.append("")

        if citations:
            md.append("### Citations")
            md.append("")
            md.extend(_render_citations_compact(citations))
            md.append("")

    # ── Statistiques ─────────────────────────────────────────────────
    md.append("---")
    md.append("## Statistiques")
    md.append("")
    md.append("| Métrique | Total |")
    md.append("|----------|-------|")
    md.append(f"| Articles | {len(articles)} |")
    md.append(f"| Entités juridiques | {sum(len(a.get('entities', [])) for a in articles)} |")
    md.append(f"| Dates | {sum(len(a.get('dates', [])) for a in articles)} |")
    md.append(f"| Citations | {sum(len(a.get('citations', [])) for a in articles)} |")
    md.append(f"| Personnes | {sum(len(a.get('persons', [])) for a in articles)} |")
    md.append(f"| Organisations | {sum(len(a.get('organizations', [])) for a in articles)} |")
    md.append(f"| Tableaux | {len(tables)} |")
    md.append("")

    result = "\n".join(md)

    out = Path(output_path) if output_path else Path(json_path).with_suffix(".md")
    out.write_text(result, encoding="utf-8")
    print(f"Rapport écrit dans {out} (mode={mode})")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path")
    parser.add_argument("-o", "--output", dest="output_path", default=None)
    parser.add_argument(
        "--mode", choices=["rag", "audit"], default="rag",
        help="rag = markdown compact pour indexation/embedding (défaut) ; "
             "audit = rendu détaillé pour relecture humaine",
    )
    args = parser.parse_args()

    convert(args.json_path, args.output_path, mode=args.mode)