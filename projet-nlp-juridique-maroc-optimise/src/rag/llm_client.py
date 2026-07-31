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

# qwen/qwen3.6-27b (préversion Groq) — le retry ci-dessous absorbe les 429
# transitoires (pic de trafic) et les erreurs réseau/5xx passagères.
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 60.0


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
        self._client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        from groq import APIConnectionError, APIStatusError, RateLimitError

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
                retryable, last_error = True, e
            except APIConnectionError as e:
                # Réseau, DNS, TLS, timeout serveur — transitoire.
                retryable, last_error = True, e
            except APIStatusError as e:
                # 4xx (clé invalide, contexte trop long) : définitif ;
                # 5xx : transitoire.
                retryable = e.status_code >= 500
                last_error = e
            except Exception as e:
                retryable, last_error = False, e
            if not retryable or attempt == MAX_RETRIES - 1:
                break
            # Backoff exponentiel (429 = rate limit, 5xx/réseau = pic transitoire)
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise last_error  # tous les essais ont échoué