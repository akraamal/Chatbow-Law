# Rapport de validation — Round 5 : extension ORG + gel des régressions

**Date :** 1 août 2026 · **Document :** BO_7522 (édition n° 7522 du Bulletin Officiel)

## 1. Contexte

Les rounds 1 à 4 ont corrigé des bugs d'intégrité du texte (citations arabes
silencieusement supprimées, marqueurs d'élision législative perdus, inversion
des chiffres dans les lignes RTL). Le round 5 a deux objectifs :

1. **Étendre `ORG_PATTERN`** aux dénominations entre guillemets à casse
   mixte (`site « Analysis and Control Laboratory (ACLAB) »`, banques,
   organismes…), en validant la liste des déclencheurs sur **tout le corpus**
   et pas seulement sur BO_7522.
2. **Geler la vérité terrain** vérifiée à la main contre les PDF dans une
   suite de tests de régression permanente.

## 2. Extension ORG — validation corpus-wide

### Scan des déclencheurs avant « ... » sur tout le corpus (interim + processed)

| Déclencheur | Occurrences | Verdict |
|---|---|---|
| `société` | 2 167 | déjà couvert |
| `dit` / `dite` | 494 | **bruit** — permis de recherche (`dit « RISSANA OFFSHORE 1 »`), concessions, taxes, pêche « No Kill », quotas |
| `dénommé(e)(s)` | 280 | **bruit 100 %** — pôles, commissions, permis (`« MOGADOR OFFSHORE 1 à 6 »`, `« GUERCIF ONSHORE I à IV »`, `« GHARB DEEP OFFSHORE I à X »`) |
| `pétrolier` / `hydrocarbures` / `réassurance` | 209 / 120 / 742 | **bruit** — secteurs, pas des entités |
| `creuse` / `espèces` | 94 / 34 | **bruit** — noms d'espèces (`l'huître creuse « Crassostrea Gigas »`) |
| `banque` | 18 | **vrai** — `« CDG Capital »`, `« Bank of Africa »`, `« Attijariwafa Bank »` (BO_7470) |
| `site` | 11 | **vrai** — `« Analysis and Control Laboratory (ACLAB) »` (BO_7500 + BO_7522) |
| `entreprise` | 4 | **vrai** — `«SMAËX»` (BO_6754) |
| `organisme` | 4 | **vrai** — `« Bureau Veritas »`, `« TÜV RHEINLAND »` (BO_7480) |
| `groupe` | 4 | **vrai** — `« SANLAM »` (BO_6758) |
| `institution` / `université` / `établissement` | — | **vrai** — `State Higher Educational Institution « Prydniprovska State Academy … »` (BO_7360_Ar) |

### Règle retenue

- Déclencheurs inclus : `société` (inchangé) + `banque`, `organisme`, `site`,
  `entreprise`, `groupe`, `établissement`, `université`, `institution`.
- Contenu entre guillemets : casse mixte, **1ʳᵉ lettre majuscule obligatoire**
  (`[A-ZÀ-Ü0-9]`) — élimine le seul faux positif trouvé
  (`entreprise « d'assurances concernée …… »`, phrase de formulaire BO_7506)
  sans perdre de vrai nom : `« chada radio s.a. »` en minuscule n'existe que
  derrière `société`, branche laissée sans contrainte.
- Exclus : `dit/dite/dénommé` (noms de permis, taxes, pôles) et noms
  d'espèces — bruit confirmé corpus-wide.

### Résultat sur BO_7522 (pipeline canonique)

- **138 articles, 191 entités** (190 avant) : **+1 ORG** —
  `site « Analysis and Control Laboratory (ACLAB) »` dans l'article n° 2
  (agrément antidumping), le miss identifié au round 4 est fermé.
- **0 régression** sur tous les autres labels :
  LOI 23 · DAHIR 5 · BULLETIN_OFFICIEL 13 · ARRETE 47 · DECRET 2 ·
  DATE_HIJRI 29 · DATE_GREGORIAN 42 · MONEY 2 · MINISTERE 3 · ORG 12.
- `possible_embedded_arabic` inchangé : `["؛", "؛"]`.

## 3. Références internes d'articles — état vérifié

Question : les articles zéro-entité contiennent-ils des références internes
(`l'article 22`, `articles 3 et 4`) non suivies ? **Elles sont déjà suivies** :
`ARTICLE_CITATION_PATTERNS` / `find_article_citations()`
(`src/extraction/article_citation_patterns.py`), champ `citations` du JSON
+ résolution via `resolve_citations`.

| Mesure (BO_7522) | Valeur |
|---|---|
| Articles zéro-entité | 50 (51 − 1 fermé par ACLAB) |
| … dont avec citations déjà suivies | **7** — exactement n° 19, 23, 27, 34, 36, 60, 72 |
| … restants sans mention « article » non capturée | **0** — pure prose procédurale |

Conclusion : pas de nouveau pattern nécessaire ; l'état « 43 articles sans
entité ni citation » est un état **correct et honnête** pour l'extracteur.

## 4. Tests de régression gelés (round 5)

Nouveaux fichiers :

| Fichier | Test | Ce qui est verrouillé |
|---|---|---|
| `tests/test_ar_bidi_integrity.py` | `test_ar_digits_in_logical_order` | BO_7515_Ar.pdf réel : `943.26`, `855.26`, `2.26.324`, `165.009,80`, `(11 ماي 2026)`, `NM 01.4.510` présents ; formes inversées `62.349`, `62.558`, `08,900.561` interdites |
| `tests/test_fr_text_integrity.py` | `test_elision_dots_preserved` | Points d'élision 4+ → marqueur `[…texte non modifié…]` ; ligne de sommaire à points retirée (pur texte, sans PDF) |
| `tests/test_fr_text_integrity.py` | `test_bo7522_pipeline_integrity` | Pipeline canonique complet sur BO_7522_Fr.pdf (sorties en `tmp_path`) : 138 articles, `arabic_runs == ["؛","؛"]`, clause SNRT `تمتنع الشركة` conservée dans le préambule du décret, marqueur d'élision présent, ACLAB capturé, 50 zéro-entité |
| `tests/test_sommaire_ordering.py` | `test_org_money_patterns` (étendu) | 12 cas positifs (dont 7 nouveaux déclencheurs) + 5 cas négatifs (permis, pôles, taxes, espèces, phrase de formulaire) |

Les tests PDF utilisent `pytest.skipif` si le PDF gitignoré est absent
(même convention que l'existant).

### Exécution

```
$ python -m pytest tests -q
11 passed in 123.42s
```

## 5. État final — à quoi s'attendre

- **Texte (embedding / LLM)** : fidèle au PDF pour ce document — le blocage
  réel est levé.
- **Entités** : index utilisable, avec des lacunes connues et documentées
  (noms d'institutions sans forme regex, variantes d'espacement, sparsité
  légitime).
- **Non couvert** : validation sur échantillon large (autres éditions) —
  étape suivante recommandée.
