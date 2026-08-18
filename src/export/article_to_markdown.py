"""
src/export/article_to_markdown.py
Convertit les articles juridiques (JSON) en Markdown avec tableaux formatés.

Détecte les motifs tabulaires dans le texte brut (lignes avec des séparations
régulières, colonnes alignées) et les rend en syntaxe Markdown.
"""

import re


# ── Heuristiques de détection de tableaux ──────────────────────────────────

# Une ligne ressemble à un en-tête ou une ligne de tableau si elle contient
# deux séquences de "  " (au moins 3 espaces) qui ne sont pas de la
# justification à droite d'un seul mot court.
_TABLE_LINE_RE = re.compile(r"  [ ]{2,}")

# Seuil : au moins N lignes consécutives avec motif tabulaire
_MIN_TABLE_ROWS = 2


def _looks_like_table_row(line: str) -> bool:
    """Heuristique : une ligne contient ≥ 2 colonnes distantes."""
    stripped = line.strip()
    if not stripped:
        return False
    # Compte les "sauts" d'au moins 3 espaces
    gaps = _TABLE_LINE_RE.findall(line)
    return len(gaps) >= 1  # au moins 2 colonnes


def _extract_columns(line: str) -> list[str]:
    """Découpe une ligne en colonnes aux sauts d'espace."""
    parts = re.split(r"  {2,}", line.strip())
    return [p.strip() for p in parts if p.strip()]


def _detect_and_format_tables(text: str) -> str:
    """
    Détecte les blocs tabulaires dans le texte et les convertit en tableaux
    Markdown.  Les blocs non tabulaires restent en l'état.
    """
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        # Chercher une séquence de lignes tabulaires consécutives
        table_rows = []
        while i < len(lines) and _looks_like_table_row(lines[i]):
            table_rows.append(lines[i])
            i += 1

        if len(table_rows) >= _MIN_TABLE_ROWS:
            # Convertir en tableau Markdown
            columns_per_row = [_extract_columns(r) for r in table_rows]
            if columns_per_row:
                max_cols = max(len(cols) for cols in columns_per_row)
                # En-tête (première ligne)
                header = columns_per_row[0]
                # Compléter les lignes trop courtes
                header = header + [""] * (max_cols - len(header))
                result.append("| " + " | ".join(header) + " |")
                # Séparateur
                result.append("|" + "|".join(" --- " for _ in range(max_cols)) + "|")
                # Lignes de données
                for cols in columns_per_row[1:]:
                    cols = cols + [""] * (max_cols - len(cols))
                    result.append("| " + " | ".join(cols) + " |")
                result.append("")
        else:
            # Ligne non tabulaire — la recopier telle quelle
            if i < len(lines):
                result.append(lines[i])
                i += 1

    return "\n".join(result)


def _build_article_md(article: dict) -> str:
    """Convertit un article (dict JSON) en bloc Markdown."""
    number = article.get("number", "?")
    text = article.get("text", "")
    raw_header = article.get("raw_header", "")

    md = f"## Article {number}\n\n"
    if raw_header:
        md += f"*{raw_header}*\n\n"

    # Texte avec tableaux formatés
    md += _detect_and_format_tables(text)
    md += "\n\n"

    # Métadonnées enrichies
    entities = article.get("entities", [])
    if entities:
        md += "*Entités :* "
        md += ", ".join(
            f"`{e.get('text', '')}` ({e.get('label', '?')})"
            for e in entities
        )
        md += "\n\n"

    citations = article.get("citations", [])
    if citations:
        md += "*Citations :*\n\n"
        for c in citations:
            target = c.get("target_text", "?")
            resolved = "✓" if c.get("resolved") else "✗"
            md += f"- `{c.get('text', '')}` → {target} [{resolved}]\n"
        md += "\n"

    persons = article.get("persons", [])
    if persons:
        md += "*Personnes :* " + ", ".join(p.get("text", "") for p in persons) + "\n\n"

    organizations = article.get("organizations", [])
    if organizations:
        md += "*Organisations :* " + ", ".join(o.get("text", "") for o in organizations) + "\n\n"

    dates = article.get("dates", [])
    if dates:
        md += "*Dates :* " + ", ".join(d.get("text", "") for d in dates) + "\n\n"

    return md


def build_full_markdown(json_result: dict) -> str:
    """
    Construit le document Markdown complet à partir du JSON d'annotation.

    Inclut :
    - En-tête du document (BO number, date, etc.)
    - Préambule (Vu, Vu, ARRÊTE)
    - Articles (détection de tableaux automatique)
    """
    lines = []

    # ── En-tête du document ──────────────────────────────────────────────
    bo_number = json_result.get("bo_number", "")
    bo_date = json_result.get("date", "")
    source = json_result.get("source", "")
    lang = json_result.get("lang", "fr")

    lines.append(f"# Bulletin Officiel {bo_number}")
    if bo_date:
        lines.append(f"**Date :** {bo_date}")
    lines.append(f"**Source :** `{source}`")
    lines.append(f"**Langue :** {lang}")
    lines.append(f"**Nombre d'articles :** {json_result.get('n_articles', 0)}")
    lines.append("")

    # ── Préambule global ─────────────────────────────────────────────────
    preamble = json_result.get("preamble_text", "")
    if preamble:
        lines.append("## Préambule\n")
        lines.append(preamble)
        lines.append("")

    # ── Décrets (préambules par décret) ──────────────────────────────────
    decrees = json_result.get("decrees", [])
    if decrees:
        lines.append("## Décrets\n")
        for d in decrees:
            lines.append(f"### {d.get('title', 'Décret')}\n")
            dp = d.get("preamble", "")
            if dp:
                lines.append(dp)
                lines.append("")
        lines.append("")

    # ── Articles ─────────────────────────────────────────────────────────
    articles = json_result.get("articles", [])
    if articles:
        lines.append("---\n")
        lines.append("## Articles\n")
        for article in articles:
            lines.append(_build_article_md(article))

    # ── Métadonnées enrichies (globales) ─────────────────────────────────
    preamble_entities = json_result.get("preamble_entities", [])
    if preamble_entities:
        lines.append("---\n")
        lines.append("## Entités du préambule\n\n")
        for e in preamble_entities:
            lines.append(f"- `{e.get('text', '')}` → {e.get('label', '?')}")
        lines.append("")

    return "\n".join(lines)
