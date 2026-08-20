"""
eval/run_routing_eval.py
--------------------------
Évaluation de l'aiguillage des questions (src/rag/query_routing.py) sur les
entrées « routing » d'eval/eval_dataset.json (type == "routing",
expected_route = {"catalog": bool, "scope": "synthesis"|null}).

Compare la précision de l'aiguillage AVANT (routeur lexical seul, repli
neutralisé) et APRÈS (routeur actuel : lexical + repli bas-coût) :
  - sans --embed : seul le repli heuristique de phrases est actif (aucun
    modèle à charger, exécution instantanée) ;
  - avec --embed : le repli par similarité utilise le modèle d'embedding
    déjà chargé par SemanticSearchEngine (chargement du modèle, lent).

Usage:
    python -m eval.run_routing_eval                  # heuristique seule
    python -m eval.run_routing_eval --embed          # + repli par embedding
    python -m eval.run_routing_eval --save results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.rag.query_routing as routing
from src.rag.query_routing import route_query

EVAL_SET_PATH = Path(__file__).parent / "eval_dataset.json"


def load_routing_entries(path: Path = EVAL_SET_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in data["queries"] if e.get("type") == "routing"]


def expected_ok(route: dict, entry: dict) -> bool:
    exp = entry.get("expected_route") or {}
    return (
        route.get("catalog") == exp.get("catalog")
        and route.get("scope") == exp.get("scope")
    )


def kind(entry: dict) -> str:
    exp = entry.get("expected_route") or {}
    if exp.get("scope") == "synthesis":
        return "synthesis"
    if exp.get("catalog"):
        return "aggregation"
    return "semantic"


def run(entries: list[dict], embed_fn) -> tuple[list[dict], dict]:
    rows = []
    for e in entries:
        route = route_query(e["query"], e.get("lang"), embed_fn=embed_fn)
        rows.append(
            {
                "id": e["id"],
                "query": e["query"],
                "expected": e.get("expected_route"),
                "kind": kind(e),
                "route": {k: route.get(k) for k in ("catalog", "type", "scope", "signal")},
                "ok": expected_ok(route, e),
            }
        )
    n = len(rows)
    correct = sum(1 for r in rows if r["ok"])
    by_kind = {}
    for k in ("aggregation", "synthesis", "semantic"):
        sub = [r for r in rows if r["kind"] == k]
        if sub:
            by_kind[k] = {
                "n": len(sub),
                "accuracy": round(sum(1 for r in sub if r["ok"]) / len(sub), 3),
            }
    return rows, {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 3) if n else None,
        "by_kind": by_kind,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embed", action="store_true",
        help="Activer le repli par embedding (charge le modèle de recherche)",
    )
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    entries = load_routing_entries()
    if not entries:
        print("Aucune entrée de type 'routing' dans le jeu d'évaluation.")
        return

    embed_fn = None
    if args.embed:
        print("Chargement de SemanticSearchEngine (modèle d'embedding)...")
        from src.search_engine.search import SemanticSearchEngine

        engine = SemanticSearchEngine()
        embed_fn = engine.embedder.embed_query

    print(f"{len(entries)} requêtes d'aiguillage.\n")

    # AVANT : routeur lexical seul (repli neutralisé).
    original = routing._fallback_classify
    routing._fallback_classify = lambda q, lang, ef: ({}, "none")
    rows_old, summary_old = run(entries, embed_fn)
    routing._fallback_classify = original

    # APRÈS : routeur actuel (lexical + repli).
    rows_new, summary_new = run(entries, embed_fn)

    print("=" * 60)
    print("AVANT (lexical seul)")
    print("=" * 60)
    print(json.dumps(summary_old, indent=2, ensure_ascii=False))
    print()
    print("=" * 60)
    print("APRÈS (lexical + repli)")
    print("=" * 60)
    print(json.dumps(summary_new, indent=2, ensure_ascii=False))

    print("\nDétail par requête (après) :")
    for r in rows_new:
        status = "OK  " if r["ok"] else "FAIL"
        print(f"  [{status}] {r['id']:<16} {r['route']['signal']:<36} {r['query']}")

    if args.save:
        Path(args.save).write_text(
            json.dumps(
                {"before": summary_old, "after": summary_new, "rows": rows_new},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nSauvegardé dans {args.save}")


if __name__ == "__main__":
    main()