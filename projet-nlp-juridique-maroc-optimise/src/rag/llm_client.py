"""
src/rag/llm_client.py
-------------------------
Étape RAG (génération) : wrapper isolé autour de l'API Groq.

Isolé du reste du pipeline RAG (chatbot.py, prompt_builder.py) pour
pouvoir changer de fournisseur plus tard sans toucher à l'orchestration
ni à la construction du prompt.

Nécessite la variable d'environnement GROQ_API_KEY.
Clé gratuite : https://console.groq.com/keys
"""
from __future__ import annotations

import os
import time

DEFAULT_MODEL_NAME = "qwen/qwen3.6-27b"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024

# qwen/qwen3.6-27b sur le tier on_demand de Groq est plafonné à 8000 TPM —
# assez bas. Le retry ci-dessous absorbe les 429 transitoires (pic de
# trafic), mais si tu vois des 429 en continu même avec un prompt réduit
# (cf. prompt_builder.MAX_CONTEXT_CHARS), la vraie solution est de changer
# de modèle (llama-3.3-70b a un tier TPM plus élevé sur Groq) ou de passer
# au Dev Tier payant.
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0


class LLMClient:
    # Tentative de chargement du .env si présent (sécurité pour tout consommateur)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    """
    Usage :
        client = LLMClient()
        answer = client.generate(
            system_instruction="Tu es un assistant juridique...",
            user_prompt="Quelles sont les règles de permis de construire ?",
        )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        from groq import Groq

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Clé API Groq introuvable — définis la variable d'environnement "
                "GROQ_API_KEY (clé gratuite sur https://console.groq.com/keys)."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = Groq(api_key=api_key)

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        from groq import RateLimitError

        safe_prompt = user_prompt or "(requête vide)"
        safe_instruction = system_instruction or "Tu es un assistant."

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": safe_instruction},
                        {"role": "user", "content": safe_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                last_error = e
                if attempt == MAX_RETRIES - 1:
                    break
                # Backoff exponentiel simple ; Groq renvoie parfois un
                # Retry-After mais on reste conservateur si absent.
                delay = RETRY_BASE_DELAY_SECONDS * (2**attempt)
                time.sleep(delay)

        raise last_error  # tous les essais ont échoué avec un 429