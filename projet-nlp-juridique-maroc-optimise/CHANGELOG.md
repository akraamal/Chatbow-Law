# Corrections apportées à cette version

## Audit GitHub (commits récents) — résolution des anomalies relevées

1. **Pages manquantes côté arabe (#5)** (`d071349`) : `scripts/enrich_json_with_pages.py`
   effectue une passe 2 globale (rescan de toutes les pages, max global ≥ 0.5) pour
   les articles non `None` et `num=None`, avec validation de monotonie limitée aux
   articles numérotés (`tests/test_page_mapping.py`). Résultat BO_7408 : 74 → 3 nulls
   (fragments étrangers).

2. **NER arabe : toponymes marocains détectés comme personnes (#6)** (`43894a9`) :
   `src/extraction/ner_statistical_ar.py` rogne les ponctuations collées aux spans
   (espaces, virgules, points, crochets) dans `_bio_tags_to_spans` ;
   `src/extraction/gazetteer_filter.py` ajoute la règle #11 `_looks_like_ar_toponym`
   (noms + préfixes `بني/أيت/الفقيه/...` + contexte « الفقيه X ») pour rejeter les
   toponymes marocains des personnes. Tests : `tests/test_ner_ar_persons.py`.

3. **Durcissement RAG/API** (`d4bba12`) :
   - `src/rag/chatbot.py` : `_standalone_query` valide la forme de l'historique
     (dict + str) avant reformulation.
   - `app/chat.py` : `get_chatbot()` re-essaie l'init du chatbot après un cooldown
     de 30 s (`_chatbot_error_at`, `_CHATBOT_RETRY_COOLDOWN_SECONDS=30.0`).
   - `src/rag/prompt_builder.py` : règle [INJECTION] — toute instruction contenue
     dans le « contenu non fiable » (contexte) est ignorée.
   - `app/analyzer.py` : vérification magic-bytes `%PDF-` avant traitement (avec
     vérification d'extension).
   - `requirements.txt` : ajout de `pytest>=7.0`.

4. **Seuil anti-hallucination calibré empiriquement (#6)** (`95ff3f8`) :
   `DEFAULT_SCORE_THRESHOLD` passe de 0.55 → 0.82 dans `src/rag/chatbot.py`.
   L'ancien seuil 0.55 était inerte : tous les scores top-1, y compris hors-sujet,
   dépassaient 0.77. Calibration sur 24 requêtes labelisées (12 pertinentes + 12
   hors-sujet, fr/ar, index 1161 docs E5) : pertinentes min 0.819 / médiane 0.833,
   hors-sujet max 0.818. Seuil 0.82 → recall 11/12, faux positifs 0/12, F1 0.957.
   Test de régression : `tests/test_score_threshold.py`.

0. **Réorganisation du dépôt (v5 → template GitHub centré chatbot)** :
   - Les deux interfaces Flask fusionnées dans un package unique `app/`
     (point d'entrée `python -m app.main`) :
     `/` = chatbot RAG (`app/chat.py`), `/analyzer` = analyseur de BO
     (`app/analyzer.py`), `lanceur_web.py` mis à jour en conséquence.
   - UIs Streamlit (`app.py`) et Chainlit (`chainlit/`) supprimées — le
     projet est désormais centré uniquement sur le chatbot LLM.
   - Fichiers supprimés : `scripts/web_app.py`, `scripts/templates/`,
     `app/flask_app.py`, scripts d'audit ponctuels, fichiers de test jetables.
   - Ajouts : `README.md`, `LICENSE` (MIT), `.env.example`,
     `src/rag/__init__.py`, `src/export/__init__.py`.
   - `reports/pipeline_diagram.md` déplacé vers `docs/` ;
     `CHANGELOG_OPTIMISATION.md` renommé en `CHANGELOG.md`.
   - `requirements.txt` complété (flask, groq, python-dotenv, numpy, pandas).

1. **scripts/run_ingestion_batch.py** : import cassé vers un module
   `pipeline_ingestion.py` qui n'existait plus (seul le `.pyc` traînait).
   `run_ingestion_pipeline` et `IngestionResult` ont été réimplémentés dans
   `src/ingestion/pipeline.py`, en s'appuyant sur les modules déjà présents
   (`layout_splitter`, `table_extractor`, `language_detector`).

2. **scripts/run_preprocess.py** : le calcul du chemin de sortie utilisait
   `"data\\interim"` (séparateur Windows) — sous Linux/macOS le
   `.replace()` ne matchait rien et le fichier nettoyé était réécrit
   silencieusement dans `data/interim/` au lieu de `data/processed/`.
   Remplacé par une manipulation `Path` indépendante de l'OS.

3. **src/ingestion/ocr_extractor.py** :
   - chemin Tesseract Windows codé en dur → rendu conditionnel via la
     variable d'environnement `TESSERACT_CMD` (ne casse plus sous
     Linux/macOS/CI).
   - import inutile de `matplotlib` (non déclaré dans requirements.txt, et
     masquait les noms locaux `image`/`text`) supprimé.
   - import dupliqué de `pytesseract` supprimé.
   - `image.save("debug_page.png")` s'exécutait à chaque page OCRisée →
     déplacé derrière un flag `debug=False` explicite.

4. **requirements.txt** : ajout de `pdfplumber` (utilisé par
   `table_extractor.py` mais absent du fichier).

5. **src/extraction/loi_decrets_patterns.py** : docstring qui référençait
   un nom de fichier différent (`lois_decrets_patterns.py`) corrigé.

6. **src/ingestion/layout_splitter.py** : paramètre `page` renommé en
   `document` dans `split_bilingual_columns` — la fonction reçoit en
   réalité le document entier, pas une page isolée (comportement
   inchangé, juste plus clair).

7. Nettoyage : `__pycache__/` et `debug_page.png` (résidu de debug) retirés
   de l'archive.

## Étape 3 — Extraction NLP (ajoutée)

Nouveaux fichiers :
- `src/extraction/entities.py` : dataclass `LegalEntity` + conversion des
  correspondances regex en spans spaCy (`doc.char_span`, fusion sans
  chevauchement avec les entités déjà posées par un EntityRuler).
- `src/extraction/loi_decrets_patterns_ar.py` : équivalent arabe de
  `loi_decrets_patterns.py` (ظهير/قانون/مرسوم/قرار/الجريدة الرسمية).
- `src/extraction/entity_ruler_builder_fr.py` / `entity_ruler_builder_ar.py` :
  pipeline spaCy blank (fr/ar) combinant EntityRuler (MINISTERE, patterns
  littéraux) + entités regex (références légales, format numérique trop
  irrégulier pour un EntityRuler token-par-token).
- `src/extraction/patterns/fr/ministeres.jsonl`,
  `src/extraction/patterns/ar/wizarat.jsonl` : patterns EntityRuler pour
  les mentions de ministères.
- `scripts/run_extraction.py` : CLI qui segmente les fichiers de
  `data/processed/{fr,ar}/` en articles (segmenter.py), extrait les
  entités de chaque article + du préambule, et sauvegarde en JSON dans
  `data/annotated/`.

Bugs trouvés et corrigés **en testant ce nouveau code sur des exemples
réels** (avant d'être intégrés au pipeline) :
- `DAHIR_PATTERN`/`BULLETIN_OFFICIEL_PATTERN` (FR, fichier existant) : la
  capture de date après "du" utilisait une classe de caractères trop
  large (`[^,;.\n]{4,40}`), qui avalait toute la phrase suivante (et même
  la parenthèse de date grégorienne, cassant sa fermeture). Remplacée par
  un motif de date précis (jour + mois + année).
- Même bug reproduit en écrivant `loi_decrets_patterns_ar.py` — corrigé
  dès l'écriture avec un motif de date hégirien précis.
- L'EntityRuler des ministères ne matchait que "ministère" et pas
  "ministre" (very fréquent dans les arrêtés) — pattern élargi avec
  `{"LOWER": {"IN": ["ministère", "ministre"]}}`.

Limite connue (comportement voulu, pas un bug) : quand une entité
ARRETE/DAHIR regex chevauche une entité MINISTERE de l'EntityRuler (ex :
"arrêté du ministre de X" contient "ministre de X"), c'est l'EntityRuler
(MINISTERE, posé en premier par le pipeline) qui gagne et l'entité regex
plus large (ARRETE) est ignorée — à ajuster dans `entities.py` si tu
préfères l'inverse (garder la référence légale complète plutôt que le
seul nom du ministère).

## Étape 4 — Enrichissement (corrections)

Bugs trouvés en exécutant le pipeline bout en bout sur des exemples réels
(la plupart des modules ci-dessous se compilaient et s'importaient sans
erreur, mais ne faisaient pas ce que leur docstring annonçait) :

1. **Extraction de dates jamais branchée** (`dates_patterns.py` /
   `dates_patterns_ar.py`) : `extract_dates_fr` / `extract_dates_ar`
   étaient écrites, testées en `__main__`, mais n'étaient appelées nulle
   part dans le pipeline réel — `entity_ruler_builder_fr.py` /
   `entity_ruler_builder_ar.py` ne posaient que les entités LOI/DECRET/
   ARRETE/DAHIR/BULLETIN_OFFICIEL/MINISTERE. Résultat : `article["entities"]`
   ne contenait jamais de DATE_HIJRI/DATE_GREGORIAN, donc
   `extract_dates_from_entities` (étape 4c) tournait sur une liste vide en
   permanence — la fonctionnalité "extraction des dates" était entièrement
   morte malgré tout le code écrit pour elle. Corrigé en fusionnant les
   entités de date avec les entités légales avant `entities_to_spacy_doc()`
   dans les deux fichiers `entity_ruler_builder_*.py`.

2. **`filter_dates()` ne reconnaissait que des mois arabes** (liste
   partielle codée en dur), appliquée indifféremment aux dates FR et AR :
   toute date française ("7 mai 2026") était donc systématiquement rejetée
   par le filtre, même une fois le bug n°1 corrigé. Réécrit pour valider
   chaque date contre le vrai dictionnaire de mois (grégorien + hégirien)
   de sa langue (`MOIS_GREGORIEN_FR/AR`, `MOIS_HIJRI_FR/AR`), au lieu d'une
   liste de mots-clés arabes partielle et non paramétrée par langue.

3. **Le texte de l'article n'était jamais sauvegardé** : `run_extraction.py`
   construisait `article = {"number":..., "raw_header":..., "entities":...}`
   sans jamais y mettre le texte (`art.text` du `segmenter.Article`), et
   `enrich_article_json()` ne le rajoutait pas non plus. Conséquence en
   cascade :
   - `keyword_classifier.classify_document()` lit `article.get("text", "")`
     pour reconstituer le texte à classifier → toujours une chaîne vide →
     domaine toujours "Indéterminé".
   - `db_connector.save_document()` insérait `article.get("raw_text")`
     (clé différente, jamais présente) dans `articles.raw_text` → colonne
     toujours NULL en base.
   - Plus généralement, le JSON annoté ne contenait jamais le texte source
     de l'article, pourtant indispensable pour l'indexation FAISS/RAG en
     aval.
   Corrigé : `enrich_article_json()` fixe maintenant `article["text"] =
   full_text` ; `db_connector.py` lit `article.get("text")` (clé
   harmonisée avec `segmenter.Article.text` et `keyword_classifier.py`).

4. **`document_metadata_extractor.py` jamais appelé** : le module
   (numéro de BO, date de publication, libellé d'édition) existait,
   testé isolément, mais aucun script ne l'invoquait. `run_extraction.py`
   calcule maintenant ces métadonnées par document et les ajoute au JSON
   produit dans `data/annotated/`.

5. **Bug de syntaxe silencieux** dans `etape4_pipeline.py` :
   `COMMON_TITLES_AR` et `COMMON_TITLES_FR` avaient une virgule manquante
   entre deux littéraux de chaîne consécutifs — Python les concatène alors
   silencieusement en une seule chaîne (`"املؤهل" "املكلف\nاملتعلق"` →
   `"املؤهلاملكلف\nاملتعلق"`), sans erreur à l'exécution, mais le filtrage
   de faux positifs qui en dépend était cassé (l'entrée fusionnée ne
   correspond plus jamais à un texte réel).

6. **Code mort supprimé** : `filter_french_persons()` (une version plus
   ancienne de `filter_persons()`, jamais appelée nulle part) et une ligne
   dans `classify_document()` qui assignait `full_text` à partir de
   `preamble_entities` avant d'être immédiatement écrasée.

7. **`requirements.txt`** : le commentaire sur `camel-tools` disait qu'il
   était "laissé en commentaire" alors qu'il était en fait actif et requis
   par `ner_statistical_ar.py` (étape 4b arabe, appelé par
   `etape4_pipeline.py`) — commentaire corrigé. Ajout d'une note sur le
   téléchargement du modèle `fr_core_news_md`, requis par
   `ner_statistical.py` mais non documenté auparavant.

Vérifications faites après ces corrections : compilation de tous les
fichiers `.py` du projet, puis test bout en bout FR (installation réelle
de `fr_core_news_md`) confirmant que les dates grégoriennes/hégiriennes,
les personnes et le texte de l'article apparaissent maintenant
correctement dans le JSON produit par `enrich_article_json()`, en français
et en arabe.

## Étape 4-bis — Consolidation + persistance SQLite (nouveau)

Nouveau fichier : `scripts/run_consolidation.py`. Parcourt les JSON
produits par `run_extraction.py` (`data/annotated/*_entities.json`),
appelle `document_consolidator.consolidate_document()` (dédoublonnage des
entités au niveau document — code déjà écrit à l'étape 4 mais jamais
branché à un script) puis `db_connector.DBConnector.save_document()` pour
persister le tout dans `data/processed/juridique.db`.

Bugs trouvés en exécutant ce nouveau script sur les documents réels du
projet (BO_7470_Fr, BO_7500_Fr) :

1. **Contrainte `UNIQUE(doc_id, number)` incorrecte sur `articles`**
   (`db_models.py`) : elle supposait qu'un numéro d'article est unique au
   sein d'un document. Faux en pratique — un même Bulletin Officiel
   regroupe souvent plusieurs textes juridiques distincts (dahir, décret,
   arrêté...) publiés à la suite, chacun recommençant sa propre
   numérotation à partir de 1. Résultat : la sauvegarde en base échouait
   systématiquement (`UNIQUE constraint failed`) dès qu'un document
   contenait plus d'un texte. Contrainte supprimée ; `id` (autoincrement)
   reste l'identifiant unique de chaque ligne.
   Limite connue (non corrigée ici, documentée pour référence future) :
   comme le découpage en articles ne garde pas trace de la frontière entre
   deux textes juridiques d'un même document, `entities_index.articles`
   peut, dans de rares cas, mélanger des mentions d'un même numéro
   d'article provenant de deux textes différents du même Bulletin.

2. **Dates de publication non reconnues quand le jour est écrit "1er"**
   (ex. "1er janvier 2026", forme très fréquente en français pour le
   premier jour du mois) : `GREGORIAN_DATE_PATTERN_FR`
   (`document_metadata_extractor.py`) et `DATE_GREGORIAN_PATTERN` /
   `DATE_HIJRI_PATTERN` (`dates_patterns.py`) n'acceptaient qu'un jour
   purement numérique (`\d{1,2}`). Corrigé pour accepter un "er" optionnel
   après le chiffre dans les trois regex.

Vérifié en conditions réelles : `python -m scripts.run_extraction` puis
`python -m scripts.run_consolidation` sur les 2 documents FR déjà traités
du projet — consolidation et sauvegarde SQLite réussies, avec
`bo_number`, `date_publication` (y compris pour les en-têtes en "1er ..."),
personnes/organisations/textes légaux dédupliqués et citations résolues
tous correctement enregistrés et interrogeables via `DBConnector`.



## Étape 1 — Ingestion : bug majeur d'ordre de lecture (2 colonnes) — corrigé

En relisant le texte arabe brut pour préparer un jeu d'entraînement
(étape 5), le texte de plusieurs articles s'est révélé incohérent : mots
et paragraphes dans le désordre, dates écrites "2026 ماي14" au lieu de
"14 ماي 2026". Vérifié directement dans `data/processed/ar/BO_7506_Ar.txt` :
le problème est présent dans le texte extrait lui-même, avant toute
étape de nettoyage ou de segmentation — donc un bug d'**ingestion**, pas
de classification.

**Cause identifiée** (`src/ingestion/pdf_extractor.py`) : le Bulletin
Officiel marocain est très souvent mis en page sur **2 colonnes**
(confirmé en inspectant les coordonnées brutes des blocs PyMuPDF sur
`data/raw/ar/BO_7506_Ar.pdf`, page 11 : une colonne de droite x0≈300-570
contenant les articles 36-37, une colonne de gauche x0≈23-210 contenant
les articles 39-40, séparées par un espace horizontal d'environ 30pt).
L'ancien tri `sorted(blocks, key=lambda b: (int(b[1]*10), b[0]))` groupe
les blocs par bande horizontale (0.1px près) puis les trie par x0
croissant — ce qui **entrelace les deux colonnes ligne par ligne** dès
qu'elles partagent une bande horizontale, au lieu de lire une colonne
en entier avant l'autre. Résultat : article 36 (colonne droite) suivi
d'un fragment de l'article 39 (colonne gauche), puis retour à l'article
36, etc. — exactement le mélange observé.

**Corrigé** : deux nouvelles fonctions dans `pdf_extractor.py` :
- `_group_into_columns()` : détecte automatiquement les colonnes d'une
  page en cherchant le plus grand espace horizontal entre deux blocs
  triés par x0 (seuil : 20pt) ; renvoie une seule "colonne" si la page
  n'a pas de vraie coupure (évite de casser les pages à une seule
  colonne).
- `_order_blocks_for_reading()` : ordonne les blocs colonne par colonne
  (chaque colonne triée de haut en bas), et décide l'ordre de lecture
  des colonnes selon la langue dominante de la page — colonne de
  **droite** en premier si le texte est majoritairement arabe (RTL),
  colonne de **gauche** en premier sinon (FR/LTR). La détection RTL
  (`_is_rtl_text()`) compte simplement les caractères dans les plages
  Unicode arabes (couvre lettres de base + formes de présentation).

Vérifié avant/après sur `data/raw/ar/BO_7506_Ar.pdf` page 11 : le texte
extrait suit maintenant l'ordre logique correct (article 36 → 37 → 38 →
39 → 40, chaque phrase grammaticalement cohérente), là où l'ancien code
donnait un mélange incompréhensible. Vérifié aussi sur un PDF français
(`BO_7470_Fr.pdf`) qu'aucune régression n'est introduite sur les pages
mono-colonne ou multi-colonnes LTR.

Limite connue : les pages de sommaire/couverture (table des matières
avec pointillés "....", tarifs d'abonnement en tableau) restent parfois
imparfaitement ordonnées — ce sont des mises en page à plus de 2 zones
que l'heuristique actuelle (coupure unique) ne modélise pas complètement.
Sans impact sur les articles de fond (ce que consomment segmenter.py et
la classification), qui sont la cible réelle de cette correction.

À refaire après cette correction : régénérer `data/processed/ar/*.txt`
pour tous les documents arabes déjà traités (la commande
`python -m scripts.run_ingestion_batch` régénère `data/interim/`, puis
`python -m scripts.run_preprocess` régénère `data/processed/` à partir
du texte corrigé), puis relancer `run_extraction` /
`run_consolidation` / `build_training_dataset` pour que le jeu
d'entraînement bénéficie du texte arabe corrigé.

## Étape 1 — Ingestion : suite de la correction arabe (bugs supplémentaires trouvés en creusant)

En creusant plus loin le bug d'ordre de lecture arabe (2 colonnes, corrigé
plus haut), trois bugs supplémentaires — plus profonds, à un niveau de
granularité plus fin — ont été trouvés et corrigés :

### 1. Ordre des mots à l'intérieur d'une ligne (nouveau bug, distinct du précédent)

M�me après la correction de l'entrelacement des 2 colonnes, certaines
lignes RTL (typiquement les lignes de référence "dahir/décret/arrêté n°
X du [date hégirienne] ([date grégorienne])") ressortaient toujours dans
le mauvais ordre : `"من23 صادر في943.26 والتجارة رقمةقرار لوزير الصناع"`
au lieu de `"قرار لوزير الصناعة والتجارة رقم 943.26 صادر في 23 من"`.
Confirmé sur `data/raw/ar/BO_7515_Ar.pdf`, page 9 : `page.get_text
("blocks")` assemble parfois le texte d'UNE SEULE ligne/span dans l'ordre
VISUEL (gauche → droite) plutôt que dans l'ordre logique de lecture RTL —
un bug distinct de l'entrelacement de colonnes, à un niveau plus fin
(dans le texte déjà assemblé d'un bloc, pas dans l'ordre des blocs entre
eux).

**Corrigé** : `pdf_extractor.py` utilise maintenant `page.get_text
("rawdict")` pour reconstruire chaque ligne caractère par caractère
(`_fix_bidi_line()`) plutôt que de faire confiance au texte déjà assemblé
par `"blocks"`. Principe (mini algorithme BiDi, cf. Unicode UAX #9) : trie
les caractères par position physique gauche→droite, découpe en tronçons
homogènes (arabe vs "autre" : chiffres/latin/ponctuation), inverse
l'ordre des tronçons ET le contenu des tronçons arabes, mais PAS celui
des tronçons numériques (les nombres restent lisibles gauche→droite même
en contexte RTL), et permute les parenthèses/crochets (miroir BiDi
standard). Vérifié : le texte se lit maintenant correctement sur les 6
PDF arabes du projet, avec l'intégrité des références de decrets/lois et
des nombres préservée.

Limite assumée : la résolution complète des caractères neutres (espaces,
parenthèses) d'UAX #9 n'est pas implémentée en totalité — voir point 2
ci-dessous pour le correctif ciblé qui comble le principal cas résiduel.

### 2. Placement des parenthèses dans les dates hégiriennes+grégoriennes

Conséquence directe de la limite du point 1 : le motif très fréquent
"date hégirienne (date grégorienne)" ressortait avec les parenthèses mal
placées, ex. `"...ذي القعدة 22) 1447 أبريل (2026"` au lieu de `"...ذي
القعدة 1447 (22 أبريل 2026)"`.

**Corrigé** : nouvelle fonction `fix_hijri_gregorian_paren_placement()`
dans `cleaner_ar.py`, avec une regex ciblée (`HIJRI_GREGORIAN_PAREN_
SWAP`) reconnaissant ce motif précis et remettant les parenthèses à leur
place. Gère les deux variantes observées (année hégirienne collée juste
après la parenthèse mal placée, ou déjà bien positionnée avant tout le
groupe) et les séparateurs parasites observés entre le mois et la
parenthèse (`:`, `.`, `-`). Résultat : passage de ~40 occurrences
cassées par document à 0-1 (le seul résidu restant est dans une page de
sommaire/index à la mise en page plus complexe, cf. limite déjà notée
pour la colonne unique du point précédent).

### 3. Bug majeur découvert en conséquence : le mot "المادة" (article) n'était jamais reconnu

En vérifiant le résultat sur les 6 PDF arabes, `BO_7517_Ar` ne trouvait
**aucun article** (`Nombre d'articles : 0`) alors que le texte semblait
correct par ailleurs. Cause trouvée : `cleaner_ar.py` contenait déjà un
correctif pour un bug de police corrompue (`fix_lam_meem_transposition`,
ciblant un motif "امل" → "الم"), mais deux problèmes cumulés le
rendaient inopérant :

- **Garde-fou d'idempotence cassé** : la fonction commençait par `if
  "الم" in text: return text` — cette condition est presque toujours
  vraie quelque part dans un texte de plusieurs milliers de caractères,
  ce qui désactivait silencieusement le correctif sur la quasi-totalité
  des documents réels (jamais détecté auparavant faute de test sur un
  texte complet).
- **Motif de corruption obsolète** : le correctif ciblait "امل" (motif
  documenté comme confirmé sur `BO_7506_Ar.pdf`), mais la correction
  BiDi caractère par caractère introduite au point 1 change la façon
  dont cette même corruption de police sous-jacente se manifeste — le
  motif réellement présent dans TOUS les documents est maintenant "مال"
  (ex. "مالادة" au lieu de "المادة", "مالالية" au lieu de "المالية"),
  confirmé avec plus de 900 occurrences de "مالادة" à travers le corpus.

**Corrigé** : la regex cible maintenant "مال" en tête de mot (avec le
même garde de préfixe court و/ف/ب/ك/ل qu'avant), uniquement quand une
lettre arabe suit (pour ne jamais toucher un éventuel "مال" isolé — "l'
argent" à part entière ; vérifié : ce mot n'apparaît jamais isolé dans ce
corpus). Le garde-fou d'idempotence global est supprimé : la regex est
idempotente par construction (une occurrence corrigée ne peut plus
matcher ensuite). `COMMON_TITLES_AR` dans `etape4_pipeline.py` (liste de
filtrage utilisée à l'étape 4 pour éliminer les faux positifs de
personnes) mise à jour en conséquence, des formes "امل..." obsolètes vers
les formes "الم..." désormais correctes.

Vérifié : les 6 documents arabes trouvent maintenant chacun des dizaines
à une centaine d'articles (117 à 168 selon le document), là où
`BO_7517_Ar` n'en trouvait aucun avant ce correctif.

**Corpus arabe regénéré** : `data/processed/ar/*.txt` pour les 6 PDF
disponibles (`BO_7505`, `BO_7506`, `BO_7511`, `BO_7513`, `BO_7515`,
`BO_7517`), avec l'intégralité des correctifs ci-dessus appliqués.

## Étape 4 — Résilience : dégradation gracieuse en l'absence de camel-tools

Bug trouvé en essayant de valider le pipeline complet sur les 6 documents
arabes après les corrections d'ingestion ci-dessus : `enrich_article_json()`
appelait `_NER_MODULES[lang].extract_persons_orgs(full_text)` sans
gestion d'erreur — si `camel-tools` n'est pas installé (cas de cet
environnement de développement, mais aussi de tout poste où cette
dépendance optionnelle n'a pas encore été installée), l'`ImportError`
remontait et faisait perdre l'enrichissement **entier** du document (dates,
citations, institutions compris), alors que seule la NER statistique
personnes/organisations en dépend réellement.

**Corrigé** : `try/except ImportError` autour de cet appel dans
`etape4_pipeline.enrich_article_json()` — en cas d'échec, poursuit avec
des listes vides pour personnes/organisations statistiques plutôt que
d'abandonner tout le document. Un avertissement s'affiche, mais une seule
fois par langue et par exécution (pas une fois par article — la première
version du correctif produisait plusieurs centaines de lignes de log
identiques sur un document de 168 articles).

Grâce à cette résilience, le pipeline complet (`run_extraction` →
`run_consolidation` → `build_training_dataset`) a pu tourner de bout en
bout sur les 8 documents du projet (2 FR + 6 AR) dans cet environnement
de développement sans `camel-tools` installé, ce qui a permis de valider
concrètement toutes les corrections d'ingestion arabe ci-dessus sur
l'ensemble du corpus disponible plutôt que sur des échantillons isolés :
**679 articles exploitables** exportés dans `data/training/domain_dataset_v2.csv`
(134 FR + 545 AR), contre 456 lignes dans la version précédente du
dataset (2 FR seulement + 168 AR au texte alors corrompu et inutilisable).

Sur un poste avec `camel-tools` installé, ce correctif n'a aucun effet
observable pour l'arabe (le bloc `try` réussit normalement) ; il ne
change quoi que ce soit pour le français, qui n'a jamais dépendu de
`camel-tools`.

## Étape 5 (préparation) — Relecture manuelle du dataset arabe

M�me démarche que pour le français : lecture du texte réel de chaque
article arabe (pas seulement les scores de mots-clés) et correction du
jeu de données `domain_dataset_final.csv` (679 lignes : 134 FR + 545 AR).

**Bug racine corrigé dans `keyword_classifier.py`** avant la relecture :
`classify_text_with_scores()` utilisait une simple recherche de
sous-chaîne (`text.count(word)`) au lieu d'une correspondance sur mot
entier, pour le français ET l'arabe — un commentaire affirmait que les
limites de mot (`\b`) "ne sont pas fiables pour l'arabe", ce qui s'est
révélé faux (vérifié). Deux conséquences concrètes :
- FR : "eau"/"air" (mots-clés Environnement) matchaient à l'intérieur de
  "réseau"/"nécessaire" (déjà connu, cf. étape 5 précédente).
- AR : "رسم" (redevance, mot-clé Fiscal) matchait à l'intérieur de
  "الرسمية" ("officiel", comme dans "الجريدة الرسمية" = Journal Officiel,
  présent dans la quasi-totalité des articles juridiques) — ce qui
  gonflait artificiellement "Fiscal" sur des articles sans rapport.
  Confirmé : au moins 50/130 étiquettes "Fiscal" arabes provenaient de ce
  seul artefact.

Corrigé : correspondance par expression régulière avec limites de mot
(`\b`), qui fonctionne correctement avec les caractères arabes en
Python. Exception ajoutée pour "بناء على" (locution "vu que"/"conformément
à", qui contient "بناء"/construction sans jamais en parler) afin de ne
pas polluer le domaine "Urbain".

Après cette correction, le taux d'étiquettes "Indéterminé" côté arabe
est passé de 214/679 à 466/679 avant relecture manuelle — confirmant,
comme pour le français, que `DOMAIN_KEYWORDS` a une couverture insuffisante
et que la relecture manuelle est indispensable (pas seulement corriger
des bugs de correspondance).

**Trois nouveaux domaines créés** (absents des 8 initiaux, sur le même
principe que "Télécommunications" ajouté côté français) :
- **Santé** : autorisations de recherche biomédicale, statut des internes/
  résidents en médecine-pharmacie-dentaire, équivalences de diplômes
  médicaux (~185 articles, un des blocs les plus importants du corpus).
- **Justice** : réglementation des professions judiciaires — huissiers de
  justice (concours, formation, tarifs, élections de l'instance
  professionnelle), réforme de la profession des adouls (notaires),
  juges de liaison et coopération judiciaire internationale (~176
  articles).
- **Enseignement** : gouvernance des établissements d'enseignement
  supérieur (conseils scientifiques, départements, laboratoires de
  recherche), équivalences de diplômes non médicales, nominations de
  l'administration scolaire (~220 articles).

**Autres corrections notables identifiées en lisant le texte** :
- Expropriations et classification de terres collectives → **Civil**
  (droit de propriété), distinct des simples clauses d'exécution de
  décret qui restent **Administratif**.
- Décisions du Conseil de la concurrence sur les opérations de
  concentration économique (fusions-acquisitions) → **Commercial**
  (~65 articles, motif très récurrent, non détecté par les mots-clés
  faute de correspondance directe avec "société"/"commerce").
- Location de droits de pêche en eaux continentales → **Environnement**,
  cohérent avec la convention déjà adoptée côté français pour
  l'aquaculture/pêche.

**589/679 lignes (87%) corrigées** au total par rapport à l'étiquetage
automatique (mots-clés). Fichier livré : `data/training/domain_dataset_final.csv`.

Point de vigilance signalé pour la suite : `scripts/build_training_dataset.py`
réattribue un nouvel `id` à chaque régénération (autoincrement SQLite
réinitialisé à chaque `run_consolidation`), donc une relecture manuelle
faite sur une version du CSV ne peut pas être fusionnée automatiquement
dans une régénération ultérieure par simple correspondance d'`id` — la
fusion doit se faire par `(doc_id, article_number, texte)`, comme fait
ici pour récupérer les corrections françaises précédentes après la
régénération du corpus.

## Étape 6 — Recherche sémantique (nouveau)

Nouveaux fichiers :
- `src/search_engine/embedder.py` : encodage des textes via
  `sentence-transformers`, modèle par défaut `intfloat/multilingual-e5-base`
  (choisi plutôt qu'un modèle "paraphrase-multilingual-*" plus généraliste
  car E5 est entraîné spécifiquement pour la recherche — embeddings
  asymétriques requête/passage via les préfixes `"query: "`/`"passage: "` —
  et supporte jusqu'à 512 tokens de contexte, utile vu la longueur de
  certains articles). Couvre le français et l'arabe nativement.
- `src/search_engine/index_builder.py` : construit un index FAISS
  (`IndexFlatIP` sur embeddings normalisés = similarité cosinus exacte)
  à partir des articles de `data/processed/juridique.db`, persiste
  l'index + les métadonnées (doc_id, numéro d'article, texte, langue,
  date) dans `data/index/`.
- `src/search_engine/search.py` : `SemanticSearchEngine`, charge l'index
  et effectue une recherche par similarité, avec filtre de langue
  optionnel (une requête en français peut légitimement remonter des
  articles arabes sémantiquement proches, le modèle étant multilingue —
  le filtre permet de restreindre si besoin).
- `scripts/build_search_index.py` : construit l'index (`python -m
  scripts.build_search_index`).
- `scripts/search_cli.py` : interroge l'index en ligne de commande, mode
  argument unique ou interactif (`python -m scripts.search_cli "ta requête"`).

Ajout à `src/storage/db_connector.py` : méthode `list_articles()`
(manquante jusqu'ici — aucune méthode ne renvoyait tous les articles
avec le texte et les métadonnées du document parent, nécessaire pour
construire l'index).

**Limite de cet environnement de développement** : pas d'accès réseau à
`huggingface.co` depuis ce bac à sable (seuls PyPI/npm/GitHub sont
autorisés), donc impossible d'y télécharger et tester le vrai modèle
`multilingual-e5-base`. Tout le reste de la logique (construction de
l'index FAISS, sauvegarde/chargement, recherche par similarité, filtre
de langue, formatage CLI) a été vérifié de bout en bout avec un embedder
factice (vecteurs aléatoires normalisés, même interface que le vrai) sur
les 683 articles réels de la base — la mécanique fonctionne, seul le
téléchargement du modèle n'a pas pu être testé ici.

**À faire de ton côté** (accès réseau normal) :
```
pip install -r requirements.txt
python -m scripts.build_search_index      # télécharge multilingual-e5-base (~1.1 Go) puis indexe
python -m scripts.search_cli "ta requête"  # ou sans argument pour le mode interactif
```

Limite connue (assumée, pas un bug) : les articles très longs (29 sur
725 dépassent 4000 caractères, dont le cas extrême à 202 725 caractères
déjà documenté à l'étape 3 — annexe/tableau avalé faute de marqueur
suivant) sont tronqués aux ~512 premiers tokens par le modèle — la
recherche sur ces articles se base donc surtout sur leur début.
