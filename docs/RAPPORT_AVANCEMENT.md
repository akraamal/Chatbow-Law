# Rapport d'état d'avancement — ADLI Morocco

## Chatbot RAG juridique marocain (Bulletin Officiel du Royaume du Maroc)

**Date :** 9 août 2026
**Branche :** `main` — dernier commit `0497c35`
**Dépôt :** https://github.com/akraamal/Chatbow-Law.git
**Statut global :** Fonctionnel de bout en bout, validé par tests de régression et démonstrations E2E.

---

## 1. Résumé exécutif

Le projet **ADLI Morocco** est un assistant juridique bilingue (français / arabe) qui répond
aux questions de droit marocain **uniquement à partir du contenu réellement indexé**
du Bulletin Officiel, avec **citations à l'appui** (bulletin, article, page). Il repose sur
un pipeline NLP complet (PDF → OCR → NER → segmentation → enrichissement → indexation
sémantique → RAG) et une application web Flask combinant **chatbot RAG** et
**analyseur de Bulletins Officiels en temps réel**.

Points clés de l'avancement :
- **Pipeline NLP complet opérationnel** de bout en bout (CLI + application web).
- **Chatbot RAG fonctionnel** : bilingue, sourcé, garde-fou anti-hallucination calibré.
- **Catalogue d'instruments + aiguillage des questions** (nouveau) : le chatbot répond
  désormais aux questions agrégées — « les dahirs les plus importants », « les décrets de
  2024 », « combien d'articles comporte le décret n° 2-25-1080 ? » — sur 814 instruments
  indexés, avec score d'importance et citations vérifiées.
- **Classification par domaine** avec modèle transféré fine-tuné (xlm-roberta).
- **Interface web entoilée** : tous les boutons et parcours utilisateurs sont maintenant
  câblés (audit UI), y compris la **pièce jointe de PDF** depuis le chat.
- **92 tests unitaires verts** (hors smoke test) — 21 suites de régression.
- **Validation manuelle** de la qualité NLP : 5 rounds de revue, figés en tests.

---

## 2. Chiffres clés

| Indicateur | Valeur |
|---|---|
| Corpus brut collecté (`data/raw`, pdf) | 37 PDF de Bulletins Officiels |
| Documents enrichis (JSON `data/annotated`) | 61 |
| Documents indexés sémantiquement (FAISS) | 1 161 (504 FR + 657 AR) |
| **Instruments au catalogue** | **814** (540 Arrêtés, 176 Décrets, 63 Décisions, 25 Dahirs, 3 Lois…) |
| Articles exploitables (jeu d'entraînement) | 679 exploitables / 725 au total |
| Lignes étiquetées manuellement corrigées | 589/679 (87 %) |
| Domaines classifiés | 11 (Administratif, Environnement, Fiscal, Transport, Civil, Commercial…) |
| Tests unitaires | **92 verts** (9 août 2026, hors smoke) / 21 suites |
| Génération LLM | Groq (qwen/llama selon le chemin, retry avec backoff sur 429) |
| Embeddings | `intfloat/multilingual-e5-base`, index `IndexFlatIP` |

---

## 3. Architecture (rappel)

```
data/raw/*.pdf ─ [1] Ingestion ─ (OCR + tables pdfplumber)
     │
     ▼
data/interim/{fr,ar} ─ [2] Prétraitement (nettoyage, correcteur OCR, segmentation)
     │
     ▼
data/processed ─ [3] Extraction NLP : NER (règles + statistique), entités, dates, instruments
     │
     ▼
data/annotated/*.json ─ [4] Enrichissement (pages, instruments, références, tables)
     │
     ├──► [5] Indexation FAISS (embeddings multilingues) ─► articles
     └──► [5'] Catalogue d'instruments (data/index/catalog.json) ─► 814 instruments
     │
     ▼
[6] RAG — chatbot : aiguillage (questions agrégées → catalogue, sinon FAISS)
    + Groq + garde-fou 0.82 + vérification des citations
     │
     ▼
Interfaces : web Flask (`/`, `/analyzer`, `/api/chat`) + CLI
```

→ Schéma détaillé : `docs/pipeline_diagram.md`

---

## 4. Avancement par module

### 4.1 Ingestion (`src/ingestion`)
- Extraction PDF (PyMuPDF), **ordre de lecture colonne par colonne**, direction RTL pour
  l'arabe, reconstruction BiDi ligne par ligne (`_fix_bidi_line`).
- Séparation des zones FR/AR d'une même page (`layout_splitter`), détection de langue.
- Repli OCR Tesseract/Paddle si PDF scanné (texte natif prioritaire).
- Extraction des tableaux en parallèle (pdfplumber).
- **Changement récent** : robustesse des colonnes (gouttière masquée par signatures
  et séparateurs de pied de page — commit `f6c72df`).

### 4.2 Prétraitement (`src/preprocessing`)
- Normalisation NFC, suppression des en-têtes/pieds, collapse des lignes vides.
- Correctifs arabes : transposition Lam/Meem (polices corrompues), parenthèses
  hijri/grégorien.
- Dictionnaire de 150+ corrections OCR, segmentation y compris préambules, sommaires
  et « Article unique ».

### 4.3 Extraction NLP (`src/extraction`)
- NER par règles (ruler FR/AR) + statistique (CRF), fusion, filtrage, gazetteer.
- **Correction AR** : toponymes marocains ne sont plus détectés comme personnes
  (commit `43894a9`) — test de régression dédié.
- Détection des instruments : Dahirs, Lois, Décrets, Arrêtés, Décisions + type/référence
  arabes depuis le préambule (commits `b53264c`, `c6ff5ee`).

### 4.4 Segmentation & enrichissement
- IDs stables par instrument et article, relations `instrument.article_ids` /
  `article.instrument_id` (uniformisation, commit `0b2b284`).
- Enrichissement en pages PDF → BO (passe 1 + passe 2 globale pour l'arabe).
- Schéma « optimal v2 » rétrocompatible (commit `c6ff5ee`).

### 4.5 Classification par domaine (nouveau)
- Baseline par mots-clés : acc 0.132 / F1 0.197 sur 660 exemples.
- **Nouveau** : modèle fine-tuné `xlm-roberta-base` (carnet Colab prêt à l'emploi,
  CV 5-fold, export), chargeur dédié avec repli mots-clés — commit `86b5cd3`.
- Corpus étiqueté : 589/679 lignes.

### 4.6 Recherche sémantique & RAG (`src/search_engine`, `src/rag`)
- Embeddings `multilingual-e5`, index FAISS, recherche sémantique.
- Chatbot RAG : reformulation des questions de suivi (historique), garde-fou par seuil
  de similarité, budget contexte 9000 chars.
- **Garde-fou calibré empiriquement** : seuil minimum de 0.82 (23 requêtes labélisées :
  recall 11/12, faux positifs 0/12, F1 0.957) — commit `95ff3f8`.
- **Vérificateur de citations** + repli LLM sur citations inexistantes (commit `e93c2f7`).
- **Anti-injection** : les instructions contenues dans le contexte non fiable sont ignorées.

### 4.6bis Catalogue d'instruments & aiguillage (NOUVEAU — commit `0497c35`)
- **`src/search_engine/catalog.py`** : catalogue de **814 instruments** (540 Arrêtés,
  176 Décrets, 63 Décisions, 25 Dahirs, 3 Lois…) extrait des JSON enrichis, dédupliqué
  par (langue, bulletin), construit à l'indexation et persisté dans
  `data/index/catalog.json` ; filtres type / année / référence et **score d'importance**
  (taille en articles, modification/abrogation d'un autre texte, statut fondamental —
  loi organique, charte, code —, actualité).
- **`src/rag/query_routing.py`** : aiguillage lexical déterministe — une question qui
  mentionne un instrument (dahir, décret, arrêté, loi… FR/AR) avec un signal
  d'agrégation (liste, « les plus importants », année, référence numérique) part vers le
  catalogue ; sinon le chemin FAISS classique.
- **Prompt dédié** (`build_catalog_prompt`) : réponse sous forme de liste ordonnée
  d'instruments (référence exacte, BO, nb d'articles, résumé), mêmes règles de citations
  verbatim vérifiées mécaniquement.
- **Exemples désormais répondus** : « les dahirs les plus importants », « les décrets de
  2024 », « combien d'articles comporte le décret n° 2-25-1080 ? », « ما هي المراسيم
  المهمة » — sans catalogue, ces questions échouaient sous le seuil anti-hallucination.

### 4.7 Application web (Flask, `app/`)
- `/` chatbot RAG (chat + suggestions + FR↔AR), `/analyzer` analyseur temps réel (SSE),
  `/api/chat` (API JSON), `/health`, `/download/<doc_id>` (PDF source), `/upload`,
  `/stream/<task_id>`, `/result/<task_id>`.
- **Pièce jointe PDF depuis le chat (paperclip)** : upload → pipeline complet en temps
  réel (logs SSE) → panneau document attaché + chat documentaire + téléchargement du
  PDF original. Vérifié de bout en bout (parcours réel via navigateur automatisé,
  zéro exception JS) — commit `6e3669c`.
- **Audit UX** : tous les boutons et liens morts câblés (Exporter, Nouvelle analyse,
  recherche en en-tête, navigations, cache-busting) — commit `e93c2f7`.
- Démarrage : `python lanceur_web.py` ou `Lancer_Analyseur_BO.bat` → http://localhost:5000.

---

## 5. Validation et qualité

- **Suites de régression (21) : 92 tests verts** exécutées le 9 août 2026
  (`pytest tests` hors smoke) — dont 28 nouveaux pour le catalogue d'instruments
  (construction, déduplication, détection FR/AR des types et références, score
  d'importance, recherche par type/année/référence, aiguillage FR/AR).
- Validations manuelles des rounds 1 à 5 figées en tests (intégrité texte FR/AR,
  identification BiDi arabe, offsets d'entités, ordre des sommaires, limites
  d'instruments, provenance temporelle, schéma, seuil, NER arabe, classification).
- Contrôles d'intégrité au niveau du corpus : magic-bytes PDF, provenance du cache
  (indentation + hash), vérification de la cohérence des IDs instruments/articles.
- Démonstrations E2E réelles (navigateur automatisé) : chat RAG répond avec sources,
  analyseur BO total (23 articles), pièce jointe PDF analysée avec OCR en direct.

---

## 6. Travail récent (2 dernières semaines en résumé)

| Récit | Contenu |
|---|---|
| `0b2b284` | IDs stables instruments/articles + régénération des 61 JSON |
| `c6ff5ee` | Schéma optimal v2 rétrocompatible |
| `dbe1cda` | Ajout du corpus complet (PDFs, textes, JSON, index FAISS, données d'entraînement) |
| `0663d88`/`86b5cd3` | Classification de domaine : CV 5-fold, Colab prêt, chargeur xlm-roberta |
| `95ff3f8` | Calibration du seuil 0.82 (anti-hallucination) + test |
| `d4ba12` | Durcissement : forme d'historique, retry chatbot (cooldown 30s), magic-bytes PDF, injection |
| `43894a9` | NER AR : toponymes ≠ personnes |
| `d071578`/`c6878ff` | Backfill pages AR + type d'instrument AR depuis le préambule |
| `e93c2f7` | UI : boutons câblés + vérificateur de citations + repli LLM + backfill pages |
| `741d629` | Rapport d'état d'avancement (`docs/RAPPORT_AVANCEMENT.md`) |
| `6e3669c` | Pièce jointe PDF depuis le chat (paperclip) : upload, SSE temps réel, doc-chat, téléchargement |
| `0497c35` | **(Dernier) Catalogue d'instruments + aiguillage** : questions agrégées (les dahirs importants, décrets de 2024, référence unique), 814 instruments, score d'importance, 28 tests |

---

## 7. Difficultés rencontrées et solutions

1. **Ordre de lecture des colonnes arabes** (BiDi) → re-implémentation mini UAX#9,
   textes figés en régression.
2. **Polices arabes corrompues** (inversion lam/alpe, dates) → correcteurs dédiés +
   dictionnaire OCR 150+ entrées.
3. **Hallucinations** → seuil de similarité calibré (0.82), réponse uniquement sur
   extraits indexés, vérificateur de citations.
4. **OCR des BO scannés** → repli OCR (Tesseract/Paddle) avec provenance documentée.
5. **UI avec boutons passifs** → audit zip des interactions + câblage + tests E2E.
6. **Injections LLM** → règle d'anti-injection sur le contexte.
7. **Stabilité de l'API** (429) → backoff + retries.
8. **Questions agrégées impossibles en RAG dense** (« les dahirs les plus importants ») →
   couche structurée (catalogue d'instruments) + aiguillage pré-retrieval.
9. **Normalisation Unicode cassant l'arabe** (NFD décompose أ → ا + U+0654) → la
   normalisation préserve les textes arabes (lower uniquement), corrigée sur les deux
   modules du catalogue. Figé par tests AR.

---

## 8. Prochaines étapes (suggestions)

1. **Recherche hybride** (dense + lexical BM25/FTS + filtres métadonnées) pour améliorer
   le recall des questions factuelles précises — le catalogue couvrant déjà l'agrégation.
2. Intégrer le **classificateur de domaine fine-tuné** dans le pipeline de production
   (indexation) et le `run_rag_pipeline`.
3. **Évaluation qualitative systématique** : jeu de QA labélisé FR/AR couvrant les deux
   chemins (extraits + catalogue) + métriques RAG (faithfulness, citation accuracy).
4. **Dockeriser** (app + worker pipeline) + déploiement (Railway/Render/VPS).
5. Enrichir le corpus (plus de BO parution après 2026).
6. Tests de charge faille de l'API `/api/chat` + instrumentation (metrics).

---

## 9. Annexe — Démo rapide (5-10 min)

1. `python lanceur_web.py` → http://localhost:5000
2. **Chat RAG factuel** : « Qui délivre le permis de construire ? » → réponse sourcée,
   cliquable sur le PDF source ; suivre la mise à jour de la langue VO ↔ arabisation.
3. **Chat RAG agrégé (nouveau)** : « les dahirs les plus importants », « les décrets de
   2024 », « combien d'articles comporte le décret n° 2-25-1080 ? » → liste ordonnée
   d'instruments sourcée (type, référence, BO, nb articles, importance), cartes
   « INSTRUMENT JURIDIQUE ».
4. **Pièce jointe** : cliquez sur le nerudopaperclip, choisissez un PDF de BO → pipeline
   temps réel, articles détectés, poser une question sur le document.
5. **Analyseur** (`/analyzer`) : upload PDF, suivi SSE, export JSON/MD, liste des
   instruments vs articles.
6. **API** : `POST /api/chat {query}` et `GET /health`.

Commandes de test (CLI) :
```bash
python -m pytest tests -q                              # 92 tests (hors smoke)
python -m scripts.run_rag_pipeline --query "les dahirs les plus importants"
python -m scripts.rag_chat_cli "Qui délivre le permis de construire ?"
python -m scripts.search_cli "licence de télécommunications"
```