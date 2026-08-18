"""
test_llm_citation_fallback.py
-------------------------------
Tests de generate_with_citation_guarantee : le modèle par défaut doit émettre
un bloc [[CITATIONS]] vérifiable. Si le modèle configuré (ex. un "reasoning"
comme qwen/qwen3.6-27b) n'émet jamais ce bloc, on régénère une fois avec
CITATION_CAPABLE_MODEL (groq/compound-mini) — sans double appel inutile
quand le modèle est déjà citant.

Usage :
    python -m pytest tests/test_llm_citation_fallback.py -v
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")

from src.rag.llm_client import (
    CITATION_CAPABLE_MODEL,
    LLMClient,
)


def _make_client(model: str, monkeypatch, responses: list[str]):
    """Client LLM avec generate() mocké ; enregistre les modèles appelés."""
    calls: list[tuple[str, str]] = []
    client = LLMClient(model_name=model, api_key="test-dummy-key")

    def fake_generate(self, system_instruction: str, user_prompt: str) -> str:
        calls.append((self.model_name, user_prompt[:40]))
        return responses.pop(0) if responses else ""

    monkeypatch.setattr(LLMClient, "generate", fake_generate, raising=True)
    return client, calls


def test_citation_capable_model_no_double_call(monkeypatch):
    client, calls = _make_client(CITATION_CAPABLE_MODEL, monkeypatch,
                                 ["réponse [...] sans bloc"])
    out = client.generate_with_citation_guarantee("sys", "q")
    assert out == "réponse [...] sans bloc"
    # modèle déjà citant → un seul appel, pas de fallback
    assert len(calls) == 1 and calls[0][0] == CITATION_CAPABLE_MODEL


def test_reasoning_model_without_citations_forwards_fallback(monkeypatch):
    client, calls = _make_client("qwen/qwen3.6-27b", monkeypatch,
                                 ["réponse valorisée", "réponse [[CITATIONS]] « x » [[END]]"])
    out = client.generate_with_citation_guarantee("sys", "q")
    assert out.startswith("réponse [[CITATIONS]]")
    assert [m for m, _ in calls] == ["qwen/qwen3.6-27b", CITATION_CAPABLE_MODEL]


def test_reasoning_model_already_citing_no_fallback(monkeypatch):
    client, calls = _make_client("qwen/qwen3.6-27b", monkeypatch,
                                 ["déjà cité [[CITATIONS]] « ok » [1] [[END]]"])
    out = client.generate_with_citation_guarantee("sys", "q")
    assert "[[CITATIONS]]" in out
    assert len(calls) == 1 and calls[0][0] == "qwen/qwen3.6-27b"