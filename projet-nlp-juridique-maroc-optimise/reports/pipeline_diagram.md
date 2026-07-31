# Diagramme du Pipeline — ADLI Morocco NLP Juridique

```
                         ┌─────────────────────────────────────────────────────────────────────┐
                         │                        DONNÉES BRUTES                              │
                         │                    data/raw/*.pdf (BO)                              │
                         └────────────────────────┬────────────────────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────────────────┐
                    │  ÉTAPE 1 : INGESTION        │                                         │
                    │  pipeline.py                │                                         │
                    │  pdf_extractor.py           │  PyMuPDF (fitz) + OCR paddle fallback   │
                    │  ocr_extractor_paddle.py    │  Détection layout (colonnes bilingues)  │
                    │  layout_splitter.py         │  Séparation FR/AR/Unknown                │
                    │  language_detector.py       │  TableExtractor (pdfplumber) parallèle   │
                    │  table_extractor.py         │                                         │
                    └─────────────────────────────┼─────────────────────────────────────────┘
                                                  │
                                                  v
                    ┌─────────────────────────────────────────────────────────────────────┐
                    │                    data/interim/{fr,ar}/*.txt                        │
                    │                    data/interim/tables/*_tables.json                │
                    └────────────────────────────────┬────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼──────────────────────────────────┐
                    │  ÉTAPE 2 : PRÉTRAITEMENT       │                                  │
                    │  cleaner_fr.py / cleaner_ar.py │  Normalisation Unicode/NFC        │
                    │  ocr_corrector.py ★            │  Suppression en-têtes/pieds page  │
                    │  segmenter.py                  │  Nettoyage artefacts arabes        │
                    │                                │  OCR corrector (dictionnaire 150+) │
                    │                                │  Collapse lignes vides             │
                    └────────────────────────────────┼──────────────────────────────────┘
                                                     │
                                                     v
                    ┌─────────────────────────────────────────────────────────────────────┐
                    │                    data/processed/{fr,ar}/*.txt                      │
                    └────────────────────────────────┬────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼──────────────────────────────────┐
                    │  ÉTAPE 3 : EXTRACTION NLP      │  segment_into_articles()          │
                    │  entity_ruler_builder_fr/ar.py │  → entités regex (LOI, DECRET...) │
                    │  loi_decrets_patterns.py       │  → EntityRuler (MINISTERE)        │
                    │  dates_patterns.py             │  → dates hégiriennes/grégoriennes  │
                    │  ner_statistical.py            │  → NER statistique (PERSON, ORG)   │
                    │  ner_merge.py                  │  → Fusion règles > statistique     │
                    │  ner_filter.py                 │  → Filtrage (blacklist, OCR)       │
                    │  gazetteer_filter.py ★         │  → Filtre gazetteer (Op.cit...)    │
                    │  citation_resolver.py          │  → Résolution citations            │
                    │  etape4_pipeline.py            │  → enrich_articles_batch()          │
                    └────────────────────────────────┼──────────────────────────────────┘
                                                     │
                                                     v
                    ┌─────────────────────────────────────────────────────────────────────┐
                    │               data/annotated/{lang}_{stem}_entities.json             │
                    │  {articles: [{number, text, entities, persons, dates, citations}]   │
                    └────────────────────────────────┬────────────────────────────────────┘
                                                     │
              ┌──────────────────────────────────────┼──────────────────────────────────────┐
              │  ÉTAPE 4 : ENRICHISSEMENT            │                                      │
              │  enrich_json_with_pages.py           │                                      │
              │  └─ _backfill_pages()                │  Page mapping (PDF text matching)    │
              │  └─ _group_into_instruments()        │  Détection limites instruments       │
              │  └─ _classify_instrument_type()      │  Type: DECRET/ARRETE/DAHIR            │
              │                                      │                                      │
              │  ÉTAPE 5 : TABLEAUX                  │                                      │
              │  enrich_json_with_pages.py --tables  │                                      │
              │  └─ enrich_json_with_tables()        │  pdfplumber table → article (text    │
              │  └─ _table_text_overlap() ★          │    overlap filter)                   │
              │  └─ _deduplicate_tables() ★          │  Dédoublement tables identiques      │
              └──────────────────────────────────────┼──────────────────────────────────────┘
                                                     │
                                                     v
                    ┌─────────────────────────────────────────────────────────────────────┐
                    │               data/annotated/ (enriched JSONs)                       │
                    │  + pdf_page, printed_page, instruments[], extracted_tables[]         │
                    └────────────────────────────────┬────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼──────────────────────────────────┐
                    │  ÉTAPE 6 : INDEXATION          │  run_rag_pipeline --build-index   │
                    │  index_builder.py              │  build_search_index.py             │
                    │  embedder.py                   │  Embeddings E5 → FAISS index       │
                    │  semantic_search.py            │  Métadonnées → metadata.json       │
                    └────────────────────────────────┼──────────────────────────────────┘
                                                     │
                                                     v
                    ┌─────────────────────────────────────────────────────────────────────┐
                    │                    data/index/faiss.index + metadata.json            │
                    └────────────────────────────────┬────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼──────────────────────────────────┐
                    │  ÉTAPE 7 : RAG                 │                                  │
                    │  chatbot.py → LegalRAGChatbot  │  Reformulation historique         │
                    │  ├─ _standalone_query()        │  Retrieval (top_k=5, seuil 0.55) │
                    │  ├─ SemanticSearchEngine.search│  Prompt → SYSTEM + context        │
                    │  ├─ build_prompt()             │  Génération Groq (llama-3.3-70b) │
                    │  └─ LLMClient.generate()       │  Réponse + sources citées         │
                    └────────────────────────────────┼──────────────────────────────────┘
                                                     │
                          ┌──────────────────────────┼──────────────────────────┐
                          │                          │                          │
                          v                          v                          v
              ┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
              │  app.py (Streamlit)  │   │  app/flask_app.py   │   │  scripts/             │
              │  Interface web       │   │  Interface Flask     │   │  └─ run_rag_pipeline  │
              │  Chat + upload docs  │   │  Chat + panel doc    │   │  └─ rag_chat_cli     │
              │                      │   │  Téléchargement PDF  │   │  └─ search_cli       │
              └─────────────────────┘   └─────────────────────┘   └──────────────────────┘



                    LÉGENDE
                    ───────

                    ★   =   Ajouté lors de la session de post-processing (juillet 2026)
                    ──── =   Flux principal (pipeline batch)
                    ──── =   Flux secondaire (interfaces utilisateur)
                    BO   =   Bulletin Officiel du Maroc
                    RAG  =   Retrieval-Augmented Generation
                    NER  =   Named Entity Recognition
                    OCR  =   Optical Character Recognition
```

### Flux alternatif (Post-Processing Cleanup)

```
  cleaned_text (from cleaner_fr.py)
         │
         ▼
  ocr_corrector.correct_ocr()     ← dictionnaire 150+ corrections
         │
         ▼
  segment_into_articles() → extraction NLP

  persons (from ner_statistical + filter_persons())
         │
         ▼
  gazetteer_filter_persons()      ← rejette Op.cit, Ibid, prépositions...
         │
         ▼
  store dans JSON

  table liée à article (by page)
         │
         ▼
  _table_text_overlap()           ← ne lie que si texte cellule dans article
         │
         ▼
  _deduplicate_tables()           ← hash (page, bbox, rows)
         │
         ▼
  deduplicated_tables (document-level)
```
