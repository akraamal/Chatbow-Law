# ADLI v2 — extraction centrée décret

Nouvelle version de l'application ADLI Morocco, tournée vers un outil
**d'extraction** : mots-clés, décrets en premier, articles complets,
métadonnées de bulletin.  L'ancienne version (chatbot RAG + analyseur)
reste **intacte** à la racine du dépôt — v2 la réutilise **en lecture
seule** (aucun fichier v1 n'est modifié).

## Différences avec la v1

| | v1 (racine) | v2 (adli-v2/) |
|---|---|---|
| Classification de domaine | xlm-roberta fine-tuné (exploratoire) | mots-clés statiques, déterministes |
| Retour d'articles | contexte RAG découpé/budgété | articles **complets** par instrument |
| Tri des instruments | ordre du document | **décrets en premier** (décret-first) |
| Métadonnées | dispersées | bloc `metadata` par document |
| OCR | priorité de dev | fallback ; priorité au texte natif |

Le chatbot RAG v1 reste disponible sur `/` (réutilisé tel quel) ;
l'analyseur v2 est sur `/analyzer`.

## Structure

```
adli-v2/
├── adli_v2/
│   ├── config/            # listes statiques de mots-clés (FR/AR, 8 catégories)
│   ├── keyword_counter.py # comptage regex, aucun modèle
│   ├── metadata.py        # bloc metadata + keyword_counts (post-enrichissement)
│   ├── pipeline.py        # orchestration : pipeline v1 + enrichissement + étape v2
│   ├── catalog.py         # catalogue décret-first
│   ├── app/               # Flask : /analyzer (v2) ; / = chat v1
│   └── scripts/           # CLI : run_extraction, build_catalog
├── tests/                 # pytest (unitaires + smoke pipeline sur PDF réel)
└── README.md
```

Sorties (gitignorées) : `adli-v2/data/{uploads,interim,processed,annotated,annotated-MD,catalog.json}`.

## Installation

Mêmes dépendances que v1 (Flask, PyMuPDF, spaCy + camel_tools pour le
pipeline).  Aucune installation supplémentaire.

## Usage

Pipeline CLI (depuis la racine du dépôt — `adli-v2/` doit être sur le
`PYTHONPATH` pour que le paquet `adli_v2` soit importable) :

```bash
# PowerShell
$env:PYTHONPATH = "adli-v2"
# bash
export PYTHONPATH=adli-v2

python -m adli_v2.scripts.run_extraction --file chemin/vers/document.pdf
python -m adli_v2.scripts.run_extraction --dir chemin/vers/dossier
python -m adli_v2.scripts.build_catalog        # → adli-v2/data/catalog.json
```

Application web (depuis la racine du dépôt) :

```bash
$env:PYTHONPATH = "adli-v2"      # export PYTHONPATH=adli-v2 en bash
python -m adli_v2.app.main
# → http://localhost:5001   (/ = chat v1, /analyzer = analyseur v2)
```

Analyseur v2 : upload d'un PDF du BO → pipeline en arrière-plan (logs
streamés) → liste des documents (métadonnées) → vue décret-first avec
articles complets et compteurs de mots-clés par instrument → agrégats de
mots-clés sur tout le corpus.

## Mots-clés

Listes **statiques** dans `adli_v2/config/` : 8 catégories héritées du
classifieur v1 (Fiscal, Social, Administratif, Civil, Pénal, Commercial,
Environnement, Urbain).  Comptage FR borné par mot + pluriel et casse
insensibles ; comptage AR par sous-chaîne.  Éditez les JSON pour ajouter
des termes — aucun modèle à réentraîner.

## Notes d'architecture

- Le pipeline v2 appelle `scripts.run_pipeline_complet` et
  `scripts.enrich_json_with_pages` (v1) avec des constantes de module
  redirigées vers `adli-v2/data/` puis restaurées : les données v1 et le
  code v1 ne sont pas modifiés.
- Le classifieur de domaine fine-tuné est **désactivé** dans le pipeline
  v2 (`classify_domain=False`) : les compteurs de mots-clés le remplacent.
- `date_parution` (en-tête du bulletin) et `decree_date_*` (signature des
  instruments) sont distincts, conformément à la revue de périmètre.

## Tests

```bash
python -m pytest adli-v2/tests -q          # unitaires (rapides)
python -m pytest adli-v2/tests/test_pipeline_smoke.py -q   # smoke PDF réel (~2 min)
```