from src.rag.prompt_builder import build_prompt

fake_articles = [{"bo_number": "7500", "doc_id": "BO_7500_Fr", "article_number": 12,
                   "lang": "fr", "date_publication": "2026-01-15", "score": 0.81,
                   "text": "Le permis de construire est délivré par le président du conseil communal..."}]
system, user = build_prompt("dans quelle date l'arrêt a-t-il été publié ?", fake_articles)
print(user)