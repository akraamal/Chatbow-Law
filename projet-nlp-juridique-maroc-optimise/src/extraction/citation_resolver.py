"""
src/extraction/citation_resolver.py
Étape 4a-bis — Résolution des citations vers leur référence légale.

Logique améliorée :
- Détection des marqueurs explicites (prépositions, "من", "de la").
- Si la citation est immédiatement précédée par une entité légale (ex. "القانون رقم 03.25 المادة 5"), on utilise cette entité.
- Fenêtre de recherche dynamique selon la longueur du texte environnant.
- Gestion des cas où la citation est dans la même phrase que l’entité.
"""
import re
from dataclasses import dataclass
from typing import Optional, List, Dict

LEGAL_TEXT_LABELS = ("LOI", "DECRET", "ARRETE", "DAHIR")

ANAPHORIC_MARKERS = {
    "fr": [r"ci-dessus", r"ci-après", r"pr[ée]sent(e)?", r"susmentionn[ée]", r"pr[ée]cit[ée]"],
    "ar": [r"أعلاه", r"أدناه", r"هذا(?:\s+ال\w+)?", r"المذكور"],
}

EXPLICIT_REF_MARKERS = {
    "fr": [r"de la loi", r"du d[ée]cret", r"de l['’]arr[êe]t[ée]", r"du dahir"],
    "ar": [r"من"],   # préposition "من" indique une dépendance
}


@dataclass
class Citation:
    text: str
    start: int
    end: int
    target_label: Optional[str] = None
    target_text: Optional[str] = None
    resolved: bool = False
    resolution_type: Optional[str] = None  # "explicit" | "anaphoric" | "unresolved"


def _get_between(citation_end: int, entity_start: int, full_text: str) -> str:
    """Retourne le texte entre la fin de la citation et le début de l'entité."""
    if citation_end < entity_start:
        return full_text[citation_end:entity_start]
    return ""


def _citation_preceded_by_entity(citation_start: int, entity_end: int, full_text: str) -> bool:
    """Vérifie si l'entité est immédiatement avant la citation (ex. 'loi n° 03-25 article 5')."""
    if entity_end <= citation_start:
        between = full_text[entity_end:citation_start]
        # Si l'espace entre les deux est court (< 20 caractères) et ne contient pas de phrase,
        # on considère qu'ils sont liés.
        return len(between.strip()) < 20
    return False


def resolve_citations(
    citations: List[Dict],
    legal_entities: List[Dict],
    doc_id: str,
    full_text: str = "",
    lang: str = "fr",
    window: int = 300
) -> List[Citation]:
    """
    citations : liste de dicts {text, start, end}
    legal_entities : entités LOI/DECRET/ARRETE/DAHIR (étape 3)
    doc_id : identifiant du document
    full_text : texte complet de l'article
    lang : "fr" ou "ar"
    window : distance maximale (caractères) entre citation et entité pour la résolution
    """
    resolved = []
    anaphoric_patterns = ANAPHORIC_MARKERS[lang]
    explicit_patterns = EXPLICIT_REF_MARKERS[lang]

    for c in citations:
        citation = Citation(text=c["text"], start=c["start"], end=c["end"])

        # 1. Entités légales à proximité (dans la fenêtre)
        nearby = [
            e for e in legal_entities
            if e["label"] in LEGAL_TEXT_LABELS
            and abs(e["start"] - c["end"]) <= window
        ]

        # 2. Recherche d'un marqueur explicite entre la citation et chaque entité
        explicit_entity = None
        if nearby and full_text:
            for e in nearby:
                between = _get_between(c["end"], e["start"], full_text)
                if any(re.search(p, between, re.IGNORECASE) for p in explicit_patterns):
                    explicit_entity = e
                    break
            # 2-bis. Marqueurs partiels (e.g. "de la" dans "de la loi" où "loi"
            # est le début de l'entité) : on étend la recherche pour inclure
            # le début de l'entité et les mots immédiatement avant la citation.
            if explicit_entity is None:
                for e in nearby:
                    between = _get_between(c["end"], e["end"], full_text)
                    if any(re.search(p, between, re.IGNORECASE) for p in explicit_patterns):
                        explicit_entity = e
                        break

        # 3. Si aucun marqueur explicite, vérifier si l'entité est immédiatement avant la citation
        if explicit_entity is None:
            for e in nearby:
                if _citation_preceded_by_entity(c["start"], e["end"], full_text):
                    explicit_entity = e
                    break

        # 4. Si une entité a été trouvée, on la lie
        if explicit_entity:
            citation.target_label = explicit_entity["label"]
            citation.target_text = explicit_entity["text"]
            citation.resolved = True
            citation.resolution_type = "explicit"

        # 5. Sinon, essayer les marqueurs anaphoriques
        elif full_text and any(
            re.search(p, full_text[c["start"]:c["end"] + 80], re.IGNORECASE)
            for p in anaphoric_patterns
        ):
            # For "susvisé", "précité", "présent" — try to match the
            # entity TYPE mentioned in the citation text to a legal
            # entity in the article, rather than defaulting to the
            # whole BO number.
            keyword_map = {
                "décret": "DECRET", "dahir": "DAHIR",
                "arrêté": "ARRETE", "arrête": "ARRETE",
                "loi": "LOI", "décision": "DECISION",
            }
            citation_lower = c["text"].lower()
            context_window = full_text[c["start"]:c["end"] + 80].lower()
            matched_label = None
            # First try the citation text itself
            for kw, label in keyword_map.items():
                if kw in citation_lower:
                    matched_label = label
                    break
            # If not found, search the surrounding context (e.g., "de la loi précitée")
            if matched_label is None:
                for kw, label in keyword_map.items():
                    if kw in context_window:
                        matched_label = label
                        break
            if matched_label and legal_entities:
                # Find the closest entity with matching label among ALL
                # legal entities (not just the 300-char window)
                candidates = [e for e in legal_entities if e["label"] == matched_label]
                if candidates:
                    closest = min(candidates, key=lambda e: abs(e["start"] - c["end"]))
                    citation.target_label = closest["label"]
                    citation.target_text = closest["text"]
                    citation.resolved = True
                    citation.resolution_type = "anaphoric_resolved"
                else:
                    # Label match failed — fall through to fallback
                    # using ALL legal entities, not just nearby ones
                    closest = min(legal_entities, key=lambda e: abs(e["start"] - c["end"]))
                    citation.target_label = closest["label"]
                    citation.target_text = closest["text"]
                    citation.resolved = True
                    citation.resolution_type = "anaphoric_fallback_entity"
            elif legal_entities:
                closest = min(legal_entities, key=lambda e: abs(e["start"] - c["end"]))
                citation.target_label = closest["label"]
                citation.target_text = closest["text"]
                citation.resolved = True
                citation.resolution_type = "anaphoric_fallback_entity"
            else:
                citation.target_label = "DOCUMENT_SOURCE"
                citation.target_text = doc_id
                citation.resolved = True
                citation.resolution_type = "anaphoric"

        else:
            # 6. Dernier recours : chercher l'entité légale la plus proche
            # dans TOUT le texte de l'article (pas seulement la fenêtre)
            if legal_entities:
                closest = min(legal_entities, key=lambda e: abs(e["start"] - c["end"]))
                citation.target_label = closest["label"]
                citation.target_text = closest["text"]
                citation.resolved = True
                citation.resolution_type = "fallback_closest"
            else:
                # Aucune entité légale dans l'article : c'est probablement
                # une citation vers le document lui-même (ex: "l'article 5
                # du présent décret").
                citation.target_label = "DOCUMENT_SOURCE"
                citation.target_text = doc_id
                citation.resolved = True
                citation.resolution_type = "fallback_document"

        resolved.append(citation)

    return resolved


def citations_to_graph(resolved_citations: List[Citation], doc_id: str) -> List[Dict]:
    """Construit un graphe de citations pour le RAG."""
    return [
        {
            "depuis": f"{c.text} ({doc_id})",
            "vers": c.target_text if c.resolved else None,
            "type": "citation",
            "resolved": c.resolved,
            "resolution_type": c.resolution_type,
        }
        for c in resolved_citations
    ]