# ⚖️ ADLI Morocco — Chatbot RAG Juridique Marocain

Assistant juridique basé sur le **Bulletin Officiel du Royaume du Maroc** : un chatbot RAG
(retrieval-augmented generation) qui répond avec **citations à l'appui** (source, article,
numéro de bulletin), propulsé par Groq + embeddings multilingues + FAISS.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask)
![Status](https://img.shields.io/badge/status-production--ready-2ea44f)

---

## ✨ Fonctionnalités

### 💬 Chatbot RAG (`/`)
- Réponses **uniquement** à partir des extraits du BO indexés (anti-hallucination)
- Chaque affirmation cite sa source : `[Source 1]` → bulletin, article, page
- Questions de suivi avec historique (reformulation automatique)
- Bilingue FR/AR (question → réponse dans la même langue)
- Garde-fou par seuil de similarité cosinus (score < 0.82 → pas de réponse inventée)
- Tableaux extraits inclus dans le contexte (liés ou non liés aux articles)
- Téléchargement du PDF source de chaque extrait

### 📄 Analyseur de Bulletins Officiels (`/analyzer`)
- Upload d'un PDF de BO → pipeline complet en temps réel (SSE)
- Détection des instruments : **Dahirs, Lois, Décrets, Arrêtés, Décisions**…
- Segmentation des articles, extraction d'entités (ministères, dates, villes…)
- Extraction des tableaux, export JSON/MD, chat documentaire par règles

### 🧠 Pipeline NLP complet (CLI)
Ingestion (OCR FR/AR) → prétraitement → extraction NER → segmentation →
enrichissement (pages, instruments, tables) → indexation sémantique → RAG.

---

## 🚀 Démarrage rapide

### 1. Prérequis
- Python 3.10+
- Tesseract OCR (pour les BO scannés) :
  `apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara`
- Clé API Groq gratuite : https://console.groq.com/keys

### 2. Installation
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python -m spacy download fr_core_news_md   # modèle NER FR
```

### 3. Configuration
```bash
copy .env.example .env            # puis renseigner GROQ_API_KEY=
```

### 4. Indexer le corpus
```bash
# Option A — à partir des JSON enrichis existants (data/annotated/) :
python -m scripts.run_rag_pipeline --build-index

# Option B — analyser un nouveau BO puis l'indexer :
python -m scripts.run_pipeline_complet --file data/raw/fr/BO_7522_Fr.pdf --enrich --tables
python -m scripts.run_rag_pipeline --build-index   # relit data/annotated/ (pas de base SQLite)
```

### 5. Lancer l'application web
```bash
python lanceur_web.py            # ou : Lancer_Analyseur_BO.bat
```
Puis ouvrir http://localhost:5000

| Route      | Description                                   |
|------------|-----------------------------------------------|
| `/`        | Chatbot RAG juridique (ADLI Morocco)          |
| `/analyzer`| Analyseur de Bulletins Officiels              |
| `/api/chat`| API JSON du chatbot (`POST {query, history}`) |
| `/health`  | Healthcheck                                   |

---

## 🧪 Tests
```bash
python tests/test_pipeline_smoke.py          # smoke test pipeline complet
python tests/test_instrument_detection.py    # détection des instruments
python tests/test_page_mapping.py            # mapping pages PDF → BO
python tests/test_enrichment_schema.py       # schéma des JSON enrichis
```

## 🖥️ CLI utile
```bash
python -m scripts.rag_chat_cli "Qui délivre le permis de construire ?"
python -m scripts.rag_chat_cli --lang ar          # mode interactif AR
python -m scripts.search_cli "licence de télécommunications"
python -m scripts.run_rag_pipeline --chat
python -m scripts.run_rag_pipeline --query "..." --build-index
```

---

## 🗂️ Structure du projet
```
app/                  # Application Flask (chatbot + analyseur BO)
  chat.py             #   Chatbot RAG : / , /api/chat, /download/<doc_id>
  analyzer.py         #   Analyseur BO : /analyzer, /upload, /stream, /result
  main.py             #   Point d'entrée unique (python -m app.main)
  templates/          #   index.html (chat) + analyzer.html (analyse)
  static/             #   CSS + JS du chatbot
src/                  # Bibliothèque cœur du pipeline NLP + RAG
  ingestion/          #   OCR, extraction PDF, split FR/AR, tableaux
  preprocessing/      #   Nettoyage + segmentation en articles
  extraction/         #   NER (règles + statistique), entités, dates
  classification/     #   Classification par domaine
  rag/                #   Chatbot RAG : chatbot.py, prompt_builder, llm_client
  search_engine/      #   Embeddings + index FAISS + recherche sémantique
  storage/            #   SQLite, consolidation des documents
  export/             #   Export Markdown
scripts/              # CLI du pipeline (run_pipeline_complet, build_search_index…)
docs/                 # Diagramme du pipeline
tests/                # Tests
data/                 # (gitignoré) PDFs + sorties du pipeline
models/               # (gitignoré) modèles téléchargés
```

## 🔧 Détails techniques
- **Génération** : Groq, modèle `qwen/qwen3.6-27b` (retry avec backoff sur 429)
- **Embeddings** : `intfloat/multilingual-e5-base`, index FAISS `IndexFlatIP`
- **Seuil anti-hallucination** : score cosinus ≥ 0.82 (calibré sur 24 requêtes
  pertinentes/hors-sujet : recall 11/12, faux positifs 0/12), budget contexte 9000 chars
- **Segmentation** : prises en compte des préambules, sommaires et `Article unique`
- **Nettoyage** : normalisation NFC, correction OCR (dictionnaire FR/AR)

## 📄 Licence
MIT — voir [LICENSE](LICENSE).
