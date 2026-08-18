"""
src/storage/document_consolidator.py
Agrège les articles enrichis (sortie de etape4_pipeline.enrich_article_json,
un objet par article) en un objet DOCUMENT, avec déduplication des entités
au niveau document.

Règle de déduplication (voir conception) :
  1. Correspondance exacte (texte normalisé : casse + accents ignorés)
     -> une seule entrée canonique, variantes listées.
  2. Correspondance partielle probable (un nom contenu dans un autre,
     ex. "Benali" ⊂ "Ahmed Benali") -> fusion avec flag merged_from,
     jamais silencieuse.
  3. Aucune correspondance -> entrée distincte.
"""
import unicodedata
from collections import defaultdict


def _normalize(text: str) -> str:
    """Casse + accents ignorés, espaces multiples réduits, pour la comparaison."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().split())


def _dedupe_entity_list(entities_with_article: list[tuple[dict, str]]) -> list[dict]:
    """
    entities_with_article : liste de (entity_dict, article_number)
    Retourne une liste d'entrées canoniques :
        {"canonical": ..., "variants": [...], "articles": [...], "merged_from": [...]}
    """
    # Regroupement par correspondance exacte d'abord
    exact_groups = defaultdict(lambda: {"variants": set(), "articles": set()})
    for entity, article_number in entities_with_article:
        key = _normalize(entity.get("text", ""))
        exact_groups[key]["variants"].add(entity["text"])
        exact_groups[key]["articles"].add(article_number)

    canonical_entries = []
    for key, group in exact_groups.items():
        # Texte le plus long comme forme canonique (souvent la plus complète, ex. "Ahmed Benali" > "Benali")
        canonical_text = max(group["variants"], key=len)
        canonical_entries.append({
            "canonical": canonical_text,
            "variants": sorted(group["variants"]),
            "articles": sorted(group["articles"], key=lambda x: (len(x), x)),
            "merged_from": [],
        })

    # Deuxième passe : fusion partielle prudente (un nom contenu dans un autre)
    # Ex: "Benali" (seul) et "Ahmed Benali" -> fusionnés, avec traçabilité.
    canonical_entries.sort(key=lambda e: -len(e["canonical"]))
    merged, absorbed_indices = [], set()

    for i, entry in enumerate(canonical_entries):
        if i in absorbed_indices:
            continue
        norm_i = _normalize(entry["canonical"])
        for j in range(i + 1, len(canonical_entries)):
            if j in absorbed_indices:
                continue
            other = canonical_entries[j]
            norm_j = _normalize(other["canonical"])
            # norm_j doit être un mot entier contenu dans norm_i, pas une sous-chaîne
            # arbitraire (évite "Ali" fusionné dans "Alicante" par exemple).
            if norm_j and norm_j in norm_i.split():
                entry["variants"] = sorted(set(entry["variants"]) | set(other["variants"]))
                entry["articles"] = sorted(set(entry["articles"]) | set(other["articles"]),
                                            key=lambda x: (len(x), x))
                entry["merged_from"].append(other["canonical"])
                absorbed_indices.add(j)
        merged.append(entry)

    return merged


def consolidate_document(doc_metadata: dict, enriched_articles: list[dict]) -> dict:
    """
    doc_metadata : sortie de document_metadata_extractor.extract_document_metadata()
    enriched_articles : liste des JSON par article (sortie étape 4,
                         chacun avec entities/dates/citations/persons/organizations)

    Retourne l'objet DOCUMENT consolidé, prêt pour la persistance (db_connector.py).
    """
    persons_flat, orgs_flat, legal_entities_flat = [], [], []
    citations_graph = []

    for article in enriched_articles:
        number = article.get("number", "?")

        for p in article.get("persons", []):
            persons_flat.append((p, number))
        for o in article.get("organizations", []):
            orgs_flat.append((o, number))
        for e in article.get("entities", []):
            if e.get("label") in ("LOI", "DECRET", "ARRETE", "DAHIR"):
                legal_entities_flat.append((e, number))

        for c in article.get("citations", []):
            citations_graph.append({
                "article": number,
                "citation_text": c.get("text", ""),
                "target_label": c.get("target_label"),
                "target_text": c.get("target_text"),
                "resolved": c.get("resolved", False),
            })

    entities_index = {
        "persons": _dedupe_entity_list(persons_flat),
        "organizations": _dedupe_entity_list(orgs_flat),
        "legal_texts": _dedupe_entity_list(legal_entities_flat),
    }

    return {
        **doc_metadata,
        "num_articles": len(enriched_articles),
        "articles": enriched_articles,
        "entities_index": entities_index,
        "citations_graph": citations_graph,
    }


if __name__ == "__main__":
    import json

    doc_metadata = {
        "doc_id": "BO_7500_Fr", "lang": "fr",
        "bo_number": "7500", "date_publication": "2026-04-16",
    }
    articles = [
        {
            "number": "5", "entities": [{"text": "loi n° 03-25", "label": "LOI", "start": 40, "end": 52}],
            "dates": [], "citations": [{"text": "l'article 5", "target_label": "LOI",
                                         "target_text": "loi n° 03-25", "resolved": True}],
            "persons": [{"text": "Ahmed Benali", "start": 89, "end": 101, "label": "PERSON"}],
            "organizations": [],
        },
        {
            "number": "12", "entities": [], "dates": [], "citations": [],
            "persons": [{"text": "Benali", "start": 10, "end": 16, "label": "PERSON"}],
            "organizations": [],
        },
    ]
    result = consolidate_document(doc_metadata, articles)
    print(json.dumps(result, ensure_ascii=False, indent=2))