# Rapport de Projet — ADLI Morocco

## Chatbot RAG juridique marocain (Bulletin Officiel du Royaume du Maroc)

**Date du rapport :** 3 août 2026
**Branche :** `main` (historique complet jusqu'au commit `b41f2a0`)
**Dépôt :** https://github.com/akraamal/Chatbow-Law.git

---

## Table des matières

1. [Objectifs du projet](#1-objectifs-du-projet)
2. [Fonctionnement détaillé du système](#2-fonctionnement-détaillé-du-système)
3. [Bugs et erreurs rencontrés](#3-bugs-et-erreurs-rencontrés)
4. [Limitations connues](#4-limitations-connues)
5. [Tests et validation](#5-tests-et-validation)
6. [Corrections de l'audit récent](#6-corrections-de-laudit-récent)
7. [Architecture du dépôt](#7-architecture-du-dépôt)
8. [Pistes d'amélioration](#8-pistes-damélioration)

---

## 1. Objectifs du projet

Le projet **ADLI Morocco** est un assistant juridique marocain construit de bout en bout
autour du **Bulletin Officiel du Royaume du Maroc** (BO), le journal normatif officiel qui
publie les dahirs, lois, décrets et arrêtés.

### 1.1 Objectif métier

Offrir un accès **simple, bilingue (français / arabe)** au droit marocain, sous la forme
d'un chatbot conversationnel capable de répondre à des questions juridiques (ex.
« Qui délivre le permis de construire ? », « نظام التعويض عن حوادث الشغل »)
**uniquement à partir du contenu réellement indexé**, chaque affirmation étant sourcée
(numéro de bulletin, n° d'article, page).

### 1.2 Objectifs techniques

1. **Construire un pipeline NLP complet** depuis le PDF du Bulletin Officiel jusqu'au
   chatbot : ingestion (extraction de texte + OCR) → prétraitement → extraction
   d'entités (NER) → segmentation en articles → enrichissement (pages, instruments,
   tableaux) → indexation sémantique (FAISS) → génération augmentée par récupération (RAG).
2. **Garantir l'intégrité du texte** : ordre de lecture correct des pages à plusieurs
   colonnes (y compris la direction RTL de l'arabe), qualité de l'OCR, réparation des
   artefacts de police corrompue.
3. **Lutter contre l'hallucination** : garde-fou par seuil de similarité cosinus — refuser
   de répondre quand aucun extrait n'est suffisamment pertinent.
4. **Standard GitHub propre** : architecture structurée, code documenté, tests de
   régression pour figer les vérités terrain vérifiées manuellement.

### 1.3 Chiffres clés

- **8 Bulletins Officiels** traités (2 FR + 6 AR), **679 articles** exploitables dans le
  jeu d'entraînement, **725 articles** au total ;
- **1 161 documents** indexés sémantiquement (504 FR + 657 AR) dans `data/index/` ;
- **38 tests unitaires** verts (`python -m pytest tests`) ;
- Corpus étiqueté manuellement : **589/679 lignes (87 %) corrigées** par rapport à
  l'étiquetage automatique par mots-clés ;
- 11 domaines classifiés : Administratif, Environnement, Fiscal, Transport, Civil,
  Commercial, Santé, Justice, Enseignement, Télécommunications, Agriculture.

---

## 2. Fonctionnement détaillé du système

Le système s'articule en **sept étapes** réparties en deux flux : un **flux batch**
(CLI/pipeline) et un **flux temps réel** (application Flask). Schéma global disponible
dans `docs/pipeline_diagram.md`.

```
data/raw/*.pdf (BO)
        │
        ▼
[1] INGESTION      pdf_extractor.py, layout_splitter.py, language_detector.py,
                   table_extractor.py, ocr_extractor_paddle.py
        │
        ▼
data/interim/{fr,ar}/*.txt  +  data/interim/tables/*_tables.json
        │
        ▼
[2] PRÉTRAITEMENT → cleaner_fr.py, cleaner_ar.py, ocr_corrector.py, segmenter.py
        │
        ▼
data/processed/{fr,ar}/*.txt
        │
        ▼
[3] EXTRACTION NLP → entity_ruler_builder_{fr,ar}.py, loi_decrets_patterns{,_ar}.py,
                     dates_patterns.py, ner_statistical(_ar).py, ner_merge.py,
                     ner_filter.py, gazetteer_filter.py, citation_resolver.py,
                     etape4_pipeline.py
        │
        ▼
data/annotated/*_entities.json
        │
        ▼
[4] ENRICHISSEMENT → enrich_json_with_pages.py (pages + instruments + types)
        ▼
data/annotated/ (JSON enrichis)
        │
        ▼
[5] TABLEAUX → enrich_json_with_pages --tables (pdfplumber → articles)
        ▼
[6] INDEXATION → index_builder.py, embedder.py → data/index/faiss.index + metadata.json
        ▼
[7] RAG → chatbot.py, prompt_builder.py, llm_client.py (Groq)
        ▼
        ──→ Interfaces : app/ (Flask) : / (chat), /analyzer, /api/chat ; scripts/ CLI
```

### 2.1 Étape 1 — Ingestion (`src/ingestion`)

**But :** transformer le PDF du Bulletin Officiel en texte brut exploitable + tableaux.

- **`pipeline.py`** : point d'entrée de l'ingestion. Retourne un `IngestionResult`
  (textes FR/AR, texte par page, etc.).
- **`pdf_extractor.py`** (cœur de l'ingestion) :
  - Extraction avec PyMuPDF (`fitz`), puis ordre des blocs **colonne par colonne**
    (fonctions `_group_into_columns()` et `_order_blocks_for_reading()`).
  - Ordre de lecture : **colonne de droite en premier si la page est majoritairement
    arabe (RTL)**, colonne de gauche en premier sinon (FR/LTR).
  - Reconstruction de chaque ligne caractère par caractère avec `_fix_bidi_line()`
    pour résoudre l'ordre logique RTL (mini-algorithme BiDi, cf. Unicode UAX #9).
- **`layout_splitter.py`** : `split_bilingual_columns(document, page)` sépare les zones
  FR et AR d'une même page (les BO sont souvent présentés sur deux colonnes, une par
  langue).
- **`language_detector.py`** : attribue une langue (fr / ar / inconnue) à chaque zone.
- **`table_extractor.py`** : extraction **parallèle** des tableaux avec `pdfplumber`
  (coordonnées des zones de tableau pour le filtrage).
- **`ocr_extractor.py` / `ocr_extractor_paddle.py`** : repli OCR Tesseract/Paddle quand
  le PDF est scanné (le texte natif est préféré à chaque fois que possible).

**Sortie :** `data/interim/{fr,ar}/*.txt` (texte brut par langue) et
`data/interim/tables/*_tables.json`.

### 2.2 Étape 2 — Prétraitement (`src/preprocessing`)

**But :** nettoyer le brut extrait avant segmentation.

- **`cleaner_fr.py` / `cleaner_ar.py`** :
  - normalisation Unicode (NFC) ;
  - suppression des en-têtes/pieds de page répétitifs ;
  - collapse des lignes vides ;
  - correctifs ciblés arabes : `fix_lam_meem_transposition` (répare les séquences de
    lettres inversées dues à des polices corrompues) et
    `fix_hijri_gregorian_paren_placement` (remise en place des parenthèses dans les
    dates « hégirienne (grégorienne) »).
- **`ocr_corrector.py`** : dictionnaire de **150+ corrections OCR** de formes
  fréquemment déformées.
- **`segmenter.py`** : `get_preamble()` (extrait le texte précédant le corps : « Vu la
  loi… », « CONSIDÉRANT… ») et `segment_into_articles()` (découpe en articles
  `Article`/`Article unique`) avec gestion des cas particuliers : préambules, sommaires,
  numérotation reprenant à 1 pour chaque texte.

**Sortie :** `data/processed/{fr,ar}/*.txt` (texte propre, prêt pour la NLP).

### 2.3 Étape 3 — Extraction NLP (`src/extraction`)

**But :** extraire des articles structurés : numéro, texte, entités légales, personnes,
organisations, dates, citations.

- **`entity_ruler_builder_fr.py` / `entity_ruler_builder_ar.py`** : construisent le
  pipeline spaCy et combinent :
  1. **EntityRuler** (patterns littéraux — ministères via
     `patterns/fr/ministeres.jsonl`, `patterns/ar/wizarat.jsonl`) ;
  2. **entités regex** pour les références de textes normatifs (LOI, DECRET, ARRETE,
     DAHIR, BULLETIN_OFFICIEL) et les formats numériques trop irréguliers pour un
     traitement token-par-token ;
  3. **entités de date** (hégiriennes ET grégoriennes), fusionnées avec les entités
     légales avant conversion en spans spaCy.
- **`loi_decrets_patterns.py` / `loi_decrets_patterns_ar.py`** : motifs regex ciblés
  avec capture du numéro et de la date.
- **`ner_statistical.py` (FR) / `ner_statistical_ar.py` (AR)** : NER statistique pour
  les personnes et organisations (spaCy côté FR, camel-tools côté AR — dégradation
  gracieuse si camel-tools est absent).
- **`ner_merge.py`** : fusion des entités des règles et de la NER statistique, priorité
  aux règles.
- **`ner_filter.py`** : filtrage des faux positifs (blacklist, artefacts OCR).
- **`gazetteer_filter.py`** : `gazetteer_filter_persons()` — rejette les entités qui
  ressemblent à des citations (« Op.cit », « Ibid », prépositions…) et les toponymes
  marocains faussement reconnus comme des PERSON.
- **`citation_resolver.py`** : résolution des citations de textes.
- **`etape4_pipeline.py`** : `enrich_article_json()` — filtre les personnes via
  `filter_persons()` (avec `COMMON_TITLES_FR/AR`), extrait les dates de publication,
  et garantit la présence du texte réel de l'article dans `article["text"]`. Résilience
  `try/except ImportError` autour de camel-tools.

**Sortie :** `data/annotated/*_entities.json` avec le schéma :
```json
{ "articles": [ { "number": "...", "text": "...", "entities": [], "persons": [],
                  "dates": [], "citations": [] }, ... ],
  "doc_id": "...", "bo_number": "...", "date_publication": "..." }
```

### 2.4 Étapes 4 et 5 — Enrichissement (pages, instruments, tableaux)

- **`enrich_json_with_pages.py`** :
  - `_backfill_pages()` : **mapping pages PDF → BO** ; cherche les signatures de chaque
    article dans le texte de chaque page et attache `pdf_page` / `printed_page` ;
  - `_group_into_instruments()` : détecte les limites entre instruments juridiques
    (un même Bulletin regroupe plusieurs textes successifs, chacun redémarrant sa
    numérotation) ;
  - `_classify_instrument_type()` : type d'instrument (DAHIR, DECRET, ARRETE,
    ARRETE_CONJOINT) ;
  - mode `--tables` : `enrich_json_with_tables()` — associe les tableaux pdfplumber aux
    articles via `_table_text_overlap()` (le texte des cellules doit apparaître dans
    l'article) et dédoublonne via `_deduplicate_tables()` (hachage page+bbox+rows) ;
    les tableaux non liés sont conservés dans `unlinked_tables`.

**Sortie :** JSON enrichis avec `pdf_page`, `printed_page`, `instruments[]`,
`extracted_tables[]`, `unlinked_tables[]`.

### 2.5 Étape 6 — Indexation sémantique (`src/search_engine`)

**But :** rendre les articles interrogeables par similarité sémantique.

- **`embedder.py`** : encode les textes avec `sentence-transformers`, modèle
  `intfloat/multilingual-e5-base` (multilingue FR/AR natif, préfixes
  `"query: "` / `"passage: "`, jusqu'à 512 tokens) ;
- **`index_builder.py`** : construit un index FAISS `IndexFlatIP` sur embeddings
  normalisés (= similarité cosinus exacte) à partir des articles de
  `data/processed/juridique.db`, puis persiste l'index + métadonnées
  (doc_id, n° d'article, texte, langue, date) dans `data/index/` ;
- **`search.py`** : `SemanticSearchEngine` — recherche par similarité avec **filtre de
  langue optionnel** (une requête FR peut remonter des articles AR sémantiquement
  proches, le modèle étant multilingue).

**Sortie :** `data/index/faiss.index` + `metadata.json` + `model_name.txt`
(1 161 documents : 504 FR + 657 AR).

### 2.6 Étape 7 — RAG (`src/rag`) et interfaces

**But :** répondre à une question en citant ses sources, sans halluciner.

- **`chatbot.py`** : `LegalRAGChatbot` —
  - récupère les `top_k=3` meilleurs extraits via `SemanticSearchEngine` ;
  - applique le **garde-fou anti-hallucination** : tout extrait sous
    `DEFAULT_SCORE_THRESHOLD = 0.82` est écarté ; s'il n'en reste aucun, réponse
    « Je n'ai pas trouvé d'information suffisamment pertinente… » (message FR et AR) ;
  - `_standalone_query()` : reformule une question de suivi (« et pour les décrets ? »)
    en requête autonome via un appel LLM dédié (avec validation de la forme de
    l'historique) ;
  - construit le prompt via `prompt_builder.py` (budget contexte 9 000 caractères,
    règle [INJECTION] : toute instruction contenue dans le contexte non fiable est
    ignorée) et génère la réponse via `llm_client.py` (Groq, `qwen/qwen3.6-27b`,
    retry avec backoff sur 429).
- **`app/chat.py`** : routes `/` (chat), `/api/chat` (POST JSON), `/download/<doc_id>` ;
  re-création du chatbot avec cooldown de 30 s si l'initialisation échoue.
- **`app/analyzer.py`** : analyse d'un BO en temps réel (SSE) : upload → pipeline
  complet → JSON/Markdown exportables ; vérification magic-bytes `%PDF-` à l'upload.
- **`app/main.py`** : point d'entrée unique (`python -m app.main`), Flask sur
  http://localhost:5000.

### 2.7 Flux alternatifs (CLI)

- `python -m scripts.run_rag_pipeline --build-index` : (re)construit l'index depuis
  `data/annotated/` ;
- `python -m scripts.rag_chat_cli "..."` : chatbot en ligne de commande ;
- `python -m scripts.search_cli "..."` : interroge l'index ;
- `python -m scripts.run_pipeline_complet --file <pdf> --enrich --tables` : pipeline
  complet sur un nouveau BO.

---

## 3. Bugs et erreurs rencontrés

Tous les bugs ci-dessous ont été **détectés, corrigés et vérifiés** ; ils sont figés par
des tests de régression là où c'était possible. Historique détaillé dans
`CHANGELOG.md`.

### 3.1 Ingestion (PDF, OCR, ordre de lecture)

| # | Bug | Correctif |
|---|-----|-----------|
| 1 | **Entrelacement des 2 colonnes** : le tri des blocs PyMuPDF par bande horizontale puis x0 lisait les deux colonnes ligne par ligne → texte arabe incohérent (mots et paragraphes mélangés, ex. « 2026 ماي14 »). | `_group_into_columns()` + `_order_blocks_for_reading()` dans `pdf_extractor.py` : lecture colonne par colonne, ordre droite-d'abord si page RTL, gauche-d'abord sinon. |
| 2 | **Ordre des mots dans une ligne RTL** : `page.get_text("blocks")` assemblait le texte d'une ligne dans l'ordre visuel (gauche→droite) au lieu de l'ordre logique RTL (ex. « من23 صادر في943.26 … »). | `_fix_bidi_line()` : reconstruction caractère par caractère avec `get_text("rawdict")` et mini-algorithme BiDi (tronçons homogènes arabes/non-arabes, inversion des tronçons arabes, miroir des parenthèses, chiffres conservés LTR). |
| 3 | **Parenthèses des dates hégiriennes+grégoriennes mal placées** (ex. « ذي القعدة 22) 1447 أبريل (2026 »). | `fix_hijri_gregorian_paren_placement()` dans `cleaner_ar.py` : regex ciblée remettant le motif « date hégirienne (date grégorienne) » en ordre. |
| 4 | **Le mot « المادة » (article) jamais reconnu** → BO_7517 trouvait **0 article**. Cause : le correctif de police corrompue ciblait « امل » mais le motif réel était « مال » (900+ occurrences), et son garde-fou d'idempotence global désactivait le correctif dès que « الم » apparaissait n'importe où. | Regex ciblant « مال » en tête de mot (avec gardes de préfixe و/ف/ب/ك/ل), idempotence par construction, garde-fou global supprimé. Résultat : 117-168 articles trouvés par document AR. |
| 5 | **Chemin Tesseract Windows codé en dur** → cassait l'OCR sous Linux/macOS/CI ; imports inutiles (matplotlib) et dédoublonnés (pytesseract) ; `image.save("debug_page.png")` exécuté à chaque page OCRisée. | `TESSERACT_CMD` via variable d'environnement, imports nettoyés, `debug=False` explicite. |
| 6 | **Cache intermédiaire périmé** : la régénération des JSON à partir d'un texte stale (extracteur modifié sans re-lecture du PDF). | Provenance dans l'interim : hash du PDF + version de l'extracteur, bloquant la régénération depuis un texte obsolète. |

### 3.2 Prétraitement / segmentation

| # | Bug | Correctif |
|---|-----|-----------|
| 7 | **`filter_dates()` ne reconnaissait que des mois arabes**, appliqué aussi aux dates FR → toute date française (« 7 mai 2026 ») était rejetée. | Validation de chaque date contre le vrai dictionnaire de mois de sa langue (`MOIS_GREGORIEN_FR/AR`, `MOIS_HIJRI_FR/AR`). |
| 8 | **Le texte de l'article n'était jamais sauvegardé** : `article` sans clé `text` → classification toujours « Indéterminé », colonne `raw_text` toujours NULL en SQLite, pas de texte pour l'indexation. | `enrich_article_json()` fixe `article["text"] = full_text` ; `db_connector.py` lit la clé harmonisée `text`. |
| 9 | **Contrainte `UNIQUE(doc_id, number)` erronée** sur `articles` : un même BO contient plusieurs textes successifs renumérotant à partir de 1 → `UNIQUE constraint failed` systématique. | Contrainte supprimée ; `id` autoincrement unique. |
| 10 | **Dates en « 1er janvier » non reconnues** : les regex de date n'acceptaient qu'un jour numérique. | « er » optionnel ajouté dans les trois regex de dates. |

### 3.3 Extraction NER

| # | Bug | Correctif |
|---|-----|-----------|
| 11 | **Extraction de dates jamais branchée** : `extract_dates_fr/ar` écrites mais jamais appelées → `article["entities"]` sans DATE_HIJRI/DATE_GREGORIAN ; `extract_dates_from_entities` tournait sur une liste vide. | Fusion des entités de date avec les entités légales avant `entities_to_spacy_doc()` dans les deux `entity_ruler_builder_*.py`. |
| 12 | **Bug de syntaxe silencieux** dans `COMMON_TITLES_AR/FR` : virgule manquante entre deux littéraux → concaténation silencieuse (« املؤهل » + « املكلف » → « املؤهلاملكلف »), filtrage des faux positifs cassé. | Virgule ajoutée. |
| 13 | **Recherche de sous-chaîne au lieu du mot entier** dans le classifieur de domaine : « eau » matchait dans « réseau » ; « رسم » (redevance) matchait dans « الرسمية » (officiel) → ~50/130 étiquettes « Fiscal » arabes artificielles. | Correspondance regex avec limites de mot `\b` (valide aussi pour l'arabe), exception « بناء على ». |
| 14 | **NER statistique FR/AR pas branchée ni protégée** : échec d'import de camel-tools faisait perdre tout l'enrichissement du document. | try/except ImportError dans `enrich_article_json()`, avertissement unique par langue. |
| 15 | **Inversion de chiffres dans les lignes 100 % RTL** et **noms composés entre guillemets arabes** (scan de bornes). | Correction du renversement RTL des chiffres + scan des bornes sensible aux guillemets arabes. |
| 16 | **Déclencheurs ORG à casse mixte entre guillemets** (banque, organisme, site, entreprise, groupe, établissement, université, institution) validés corpus. | Liste de déclencheurs enrichie et validée. |

### 3.4 RAG / API / Web

| # | Bug | Correctif |
|---|-----|-----------|
| 17 | **XSS** dans l'interface (historique chat injectable). | Échappement / sécurisation du rendu. |
| 18 | **Historique de conversation mal formé** → crash de `_standalone_query()`. | Validation de la forme (dict + str) avant reformulation. |
| 19 | **Initialisation du chatbot instable** (échec au premier chargement, sans reprise). | `get_chatbot()` re-tente après un cooldown de 30 s (`_chatbot_error_at`). |
| 20 | **Upload de fichier non-PDF** (fausse extension) analysé par le pipeline. | Vérification magic-bytes `%PDF-` + extension. |
| 21 | **Injection de prompt via le contexte** : une instruction dans un document indexé pouvait détourner le LLM. | Règle [INJECTION] dans le prompt système : tout contenu non fiable du contexte est ignoré. |

### 3.5 NER arabe : toponymes (correction d'audit #6)

- **Symptôme :** les noms de lieux marocains (ex. « بني ملال », « الفقيه بن صالح »,
  « أيت ملول ») étaient reconnus comme des PERSON par la NER statistique arabe.
- **Causes :** spans incluant des ponctuations collées (virgules, crochets) et
  absence de filtre gazetteer pour les toponymes.
- **Correctifs :**
  - `_bio_tags_to_spans()` rogne les ponctuations collées (espaces, virgules, points,
    crochets) ;
  - `_looks_like_ar_toponym()` (règle #11 de `gazetteer_filter_persons()`) rejette les
    noms + préfixes `بني/أيت/الفقيه/...` et le contexte « الفقيه X ».
  - Tests : `tests/test_ner_ar_persons.py` (4 tests).

### 3.6 Pages PDF → BO manquantes (correction d'audit #5)

- **Symptôme :** de nombreux articles arabes (fragments) avaient `page=None`
  (BO_7408 : 74 pages manquantes).
- **Correctif :** passe 2 globale dans `enrich_json_with_pages.py` — rescan de toutes
  les pages avec max global ≥ 0.5 pour les articles `num=None` ; la validation de
  monotonie ne s'applique qu'aux articles numérotés. Résultat : 74 → 3 nulls
  (fragments étrangers).

---

## 4. Limitations connues

Limites **assumées et documentées** (non corrigées volontairement, ou hors périmètre
de l'environnement de développement).

1. **Troncature à 512 tokens (embeddings).** Le modèle E5 plafonne le contexte à
   512 tokens. 29 articles sur 725 dépassent 4 000 caractères, dont un cas extrême à
   202 725 caractères (annexe/tableau avalé faute d'un marqueur suivant). La recherche
   sémantique sur ces articles se base donc sur leur **début seulement**. Non
   corrigeable simplement sans changer de modèle ou découper en chunks.

2. **Saturation des scores cosinus.** Sur ce corpus juridique très homogène, les
   scores de similarité top-1 **saturent autour de 0,78–0,84 même pour des requêtes
   hors-sujet** (mesuré sur 24 requêtes). Un seuil absolu unique ne suffit pas à
   discriminer parfaitement — le seuil calibré 0,82 est un compromis optimal et non
   une frontière nette.

3. **Ordre de lecture des pages de sommaire.** Les pages de sommaire/couverture
   (tables des matières avec pointillés, tarifs d'abonnement en tableau) restent
   parfois imparfaitement ordonnées : mises en page à plus de 2 zones que l'heuristique
   (coupure unique de colonnes) ne modélise pas entièrement. Sans impact sur les
   articles de fond.

4. **Dépendance facultative `camel-tools` (AR).** La NER statistique arabe se
   dégrade gracieusement en son absence.

5. **Résolution BiDi incomplète (UAX #9).** La résolution complète des caractères
   neutres (espaces, parenthèses) n'est pas implémentée en totalité ; le cas résiduel
   principal (parenthèses des dates hégiriennes+grégoriennes) est comblé par un
   correctif regex ciblé.

6. **Traçabilité du jeu d'entraînement.** `build_training_dataset.py` réattribue un
   nouvel `id` à chaque régénération (autoincrement SQLite réinitialisé par
   `run_consolidation`). Les corrections manuelles d'un CSV ne peuvent être fusionnées
   par simple correspondance d'`id` ; la fusion doit se faire par
   `(doc_id, article_number, texte)`.

7. **Couverture des mots-clés insuffisante.** `DOMAIN_KEYWORDS` ne couvre pas tous les
   domaines ; l'étiquetage automatique initial n'était correct que sur ~13 % des lignes
   — la relecture manuelle (87 % corrigées) est indispensable tant qu'aucun modèle
   affiné n'est entraîné.

---

## 5. Tests et validation

### 5.1 Suite de tests (`python -m pytest tests`)

**38 tests verts.** Principaux fichiers :

| Fichier | Couvre |
|---------|--------|
| `test_ar_bidi_integrity.py` | Intégrité BiDi du texte arabe (chiffres, ordre) |
| `test_fr_text_integrity.py` | Intégrité du texte français |
| `test_ner_ar_persons.py` | NER arabe : toponymes rejetés, spans propres (#6) |
| `test_page_mapping.py` | Backfill des pages PDF → BO (#5), monotonie |
| `test_instrument_detection.py` / `test_instrument_boundaries.py` / `test_instrument_type_ar.py` | Détection/limites/type des instruments |
| `test_sommaire_ordering.py` / `test_sommaire_ar.py` | Ordre des sommaires |
| `test_enrichment_schema.py` | Schéma des JSON enrichis |
| `test_entity_offsets.py` | Offsets d'entités |
| `test_interim_provenance.py` | Provenance du cache intermédiaire |
| `test_score_threshold.py` | Seuil anti-hallucination calibré (0.82) |
| `test_pipeline_smoke.py` | Smoke test pipeline complet |

NB : les tests unitaires ne nécessitent ni clé Groq ni index FAISS.

### 5.2 Fixtures de régression (« vérités terrain »)

Plusieurs commits figent des résultats **vérifiés manuellement en rounds de relecture**
(ronds 1 à 5) afin d'empêcher toute régression : tests BO_7480 (frontière
dahir/décret), BO_7492/BO_7510 (frontières d'arrêté), `test_sommaire_*`, et le
rapport `docs/rapport_validation_round5.md`.

### 5.3 Validation du seuil anti-hallucination (audit #6)

Calibration sur **24 requêtes labelisées** (12 pertinentes + 12 hors-sujet, fr/ar) :

```
relevant top-1 : min 0.819 | médiane 0.833 | max 0.844
off-topic top-1 : min 0.777 | médiane 0.801 | max 0.818

seuil 0.80  → recall 12/12, fp 6/12  (F1 0.800)  garde-fou presque inerte
seuil 0.82  → recall 11/12, fp 0/12 (F1 0.957)  compromis retenu
```

Ancien seuil 0.55 : **inerte** (tous les scores top-1, hors-sujet compris, le
dépassaient). Nouveau seuil 0.82 dans `DEFAULT_SCORE_THRESHOLD` (`src/rag/chatbot.py`).

### 5.4 Tests d'acceptation fonctionnelle

- Smoke test pipeline complet sur `data/raw/BO_7500_Fr.pdf` ;
- Parcours de bout en bout : ingestion → prétraitement → extraction → consolidation →
  indexation → chat, sur 8 BO ;
- Tests ad hoc : questions FR/AR sur plusieurs domaines, vérification du format des
  réponses et de la présence des sources ([Source 1] etc.).

---

## 6. Corrections de l'audit récent

Historique des 5 commits récents de correction (voir `git log --oneline`) :

| Commit | Contenu |
|--------|---------|
| `d071349` | Backfill pages AR fragments : passe 2 globale (sans contrainte de monotonie) + validateur exemptant les articles non numérotés (#5) |
| `43894a9` | NER arabe : spans propres + gazetteer toponymes rejetés comme PERSON (#6) |
| `d4bba12` | Durcissement RAG/API : historique validé, re-création chatbot après cooldown, magic-bytes PDF, règle [INJECTION], pytest requis |
| `95ff3f8` | Seuil anti-hallucination calibré à 0.82 (était 0.55, inerte) + test de régression |
| `b41f2a0` | Mise à jour de la documentation (README, pipeline_diagram, CHANGELOG) |

Tous poussés sur `main` (https://github.com/akraamal/Chatbow-Law.git).

---

## 7. Architecture du dépôt

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
  extraction/         #   NER, entités, dates, citations, gazetteer
  classification/     #   Classification par domaine
  rag/                #   Chatbot RAG : chatbot, prompt_builder, llm_client
  search_engine/      #   Embeddings + index FAISS + recherche
  storage/            #   SQLite, consolidation des documents
  export/             #   Export Markdown
scripts/              # CLI du pipeline (run_pipeline_complet, …)
docs/                 # Diagramme + rapports de validation
tests/                # Tests unitaires et de régression
data/                 # (gitignoré) PDFs + sorties du pipeline
models/               # (gitignoré) modèles téléchargés
```

Dépendances clés : Flask, PyMuPDF, pdfplumber, spaCy (+ `fr_core_news_md`), camel-tools
(optionnel), sentence-transformers (E5), FAISS, Groq, pytest.

## 8. Pistes d'amélioration

1. **Découpage en chunks** des articles longs (plafond 512 tokens) avant embedding —
   réduirait la saturation des scores et améliorerait la précision de la recherche.
2. **Fine-tuning** des domaines (le jeu d'entraînement manuel est prêt) et d'une NER
   juridique spécifique.
3. **Fusion par `(doc_id, article_number, texte)`** automatisée dans
   `build_training_dataset.py` pour rendre la relecture manuelle réutilisable.
4. **Mesures d'évaluation RAG** : jeu de validation FR/AR plus large pour ré-estimer
   le seuil et tester le reclassement des hors-sujet.
5. **Gestion des annexes** : segmentation des articles à annexes (cas 202k caractères).
6. **Multilingue étendu** : support de questions en darija/amazighe non couvertes par E5.


