"""
test_catalog.py
---------------
Régression pour adli_v2.catalog : le catalogue est trié DÉCRETS EN
PREMIER, chaque entrée porte les métadonnées et les index des articles.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adli_v2.catalog import build_catalog, entry_from_json, load_catalog, save_catalog


def _write_doc(dir_path: Path, name: str, bo_number: str, instruments: list[dict]) -> Path:
    path = dir_path / f"{name}_entities.json"
    path.write_text(json.dumps({
        "doc_id": name,
        "lang": "fr",
        "bo_number": bo_number,
        "bo_date_publication": "2026-04-16",
        "articles": [{"text": "x"}] * 4,
        "instruments": instruments,
        "metadata": {"doc_name": name, "bo_number": bo_number,
                     "date_parution": "2026-04-16"},
    }), encoding="utf-8")
    return path


def test_catalog_sorts_decrets_first(tmp_path):
    _write_doc(tmp_path, "fr_BO_7000_Fr", "7000", [
        {"instrument_type": "ARRETE", "reference": "405-26", "n_articles": 2,
         "article_indices": [0, 1], "keyword_counts": {}},
        {"instrument_type": "DECRET", "reference": "2-20-716", "n_articles": 2,
         "article_indices": [2, 3], "keyword_counts": {}},
    ])
    entries = build_catalog(tmp_path)
    types = [e["instrument_type"] for e in entries]
    assert types[0] == "DECRET", types
    assert types[1] == "ARRETE", types


def test_catalog_bo_number_secondary_sort(tmp_path):
    _write_doc(tmp_path, "fr_BO_7000_Fr", "7000", [
        {"instrument_type": "DECRET", "reference": "2-1", "n_articles": 1,
         "article_indices": [0], "keyword_counts": {}}])
    _write_doc(tmp_path, "fr_BO_7500_Fr", "7500", [
        {"instrument_type": "DECRET", "reference": "2-2", "n_articles": 1,
         "article_indices": [0], "keyword_counts": {}}])
    entries = build_catalog(tmp_path)
    assert entries[0]["bo_number"] == "7500"


def test_entry_carries_metadata_and_article_indices(tmp_path):
    path = _write_doc(tmp_path, "fr_BO_7100_Fr", "7100", [
        {"instrument_type": "DECRET", "reference": "2-20-716", "n_articles": 2,
         "article_indices": [1, 2],
         "title": "Décret n° 2-20-716",
         "decree_date_gregorian": "2026-04-16",
         "keyword_counts": {"per_category": {"Fiscal": 3}}}])
    data = json.loads(path.read_text(encoding="utf-8"))
    (entry,) = entry_from_json(data)
    assert entry["doc_name"] == "fr_BO_7100_Fr"
    assert entry["bo_number"] == "7100"
    assert entry["article_indices"] == [1, 2]
    assert entry["title"] == "Décret n° 2-20-716"
    assert entry["keyword_counts"]["per_category"]["Fiscal"] == 3


def test_catalog_save_load_roundtrip(tmp_path):
    entries = [{"doc_name": "a", "instrument_type": "DECRET", "bo_number": "7100"}]
    path = save_catalog(entries, tmp_path / "catalog.json")
    assert load_catalog(path) == entries
    assert json.loads(path.read_text(encoding="utf-8"))["n_entries"] == 1