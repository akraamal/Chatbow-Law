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

DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"

# Modèle capable d'émettre le bloc [[CITATIONS]] vérifiable par le chatbot.
# Attention : ne PAS mettre ici un modèle "reasoning" (ex. qwen/qwen3.6-27b)
# — il consomme les tokens sur une chaîne de pensée visible et n'émet JAMAIS
# de bloc [[CITATIONS]], ce qui rendrait aveugle le garde-fou anti-
# hallucination (le chatbot ne pourrait jamais vérifier ce que dit le LLM).
CITATION_CAPABLE_MODEL = "llama-3.3-70b-versatile"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024

# Le retry ci-dessous absorbe les 429 transitoires (pic de trafic) et les
# erreurs réseau/5xx passagères.
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
        self._api_key = api_key
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

    def generate_with_citation_guarantee(
        self, system_instruction: str, user_prompt: str
    ) -> str:
        """
        Génère une réponse en GARANTISSANT l'émission d'un bloc
        [[CITATIONS]] vérifiable.

        Certains modèles configurés (ex. qwen/qwen3.6-27b, un modèle
        "reasoning") n'émettent jamais ce bloc : le garde-fou anti-
        hallucination du chatbot devient aveugle et aucune citation ne peut
        être contrôlée mécaniquement. Si la première génération ne contient
        pas de bloc [[CITATIONS]], on réessaie une fois avec
        CITATION_CAPABLE_MODEL (llama-3.3-70b-versatile) — qui remplit le
        format. Le modèle citant est utilisé directement quand c'est déjà
        celui configuré (pas de double appel).
        """
        answer = self.generate(system_instruction, user_prompt)
        if self.model_name == CITATION_CAPABLE_MODEL or "[[CITATIONS]]" in answer:
            return answer
        fallback = LLMClient(
            model_name=CITATION_CAPABLE_MODEL,
            api_key=self._api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return fallback.generate(system_instruction, user_prompt)