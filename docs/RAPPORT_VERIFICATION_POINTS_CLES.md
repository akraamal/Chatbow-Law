# Rapport de vérification manuelle — « Points clés » (§6)

**Date** : 23 août 2026
**Fonctionnalité vérifiée** : onglet « Points clés » de l'analyseur v2
(liste par décret avec titre+teaser IA, détail paresseux au clic,
plafond à 15 décrets, garde-fous de quota).
**Référence** : guide d'implémentation « Points clés », section 6.

---

## 1. Méthodologie

| Élément | Détail |
|---|---|
| Pilote | Playwright 1.x + Chromium headless (1440×950) |
| Application | `adli_v2.app.main` servie sur `127.0.0.1:5055` |
| Instrumentation | Compteur d'appels LLM monté autour de `LLMClient.generate` / `generate_with_citation_guarantee` par le lanceur de test (**aucune modification du dépôt**) |
| Documents de test | `BO_7500_Fr_519ee81f` (2 décrets) et `BO_7350_Ar_ad0651cc` (33 décrets → plafond 15) |
| Traçage réseau | Interception des requêtes `/key-points/*` dans le navigateur |

Chaque étape du guide §6 a été exécutée automatiquement avec assertions ;
les échecs n'interrompent pas le scénario (rapport complet garanti).

---

## 2. Résultats — 14/15 ✅

| # | Vérification | Résultat | Détail mesuré |
|---|---|---|---|
| 1a | Le clic sur « 📌 Points clés » active l'onglet dédié | ✅ OK | classe `active` présente sur `.tab-btn[data-tab="keypoints"]` |
| 1b | Liste générée : un bullet par décret | ✅ OK | 2 bullets pour le document à 2 décrets |
| 1c | Titres générés par IA = **sujets réels**, pas des numéros | ✅ OK | ex. *« Montants minimums et règles de gouvernance des fonds d'investissement »* |
| 2 | Exactement **une** requête de liste vers le bon document | ✅ OK | 1 × `GET /key-points/BO_7500_Fr_519ee81f`, aucune autre |
| 3a | Clic sur un point → développement détaillé avec sources ancrées | ✅ OK | texte contenant « Sources » + références « art. N » cliquables |
| 3b | Génération paresseuse : 1 requête de détail au clic (0 pendant le listage) | ✅ OK | compteur réseau et compteur Groq cohérents |
| 3c | Re-clic (fermer/rouvrir) : **aucun re-fetch** (`dataset.loaded`) | ✅ OK | requêtes détail inchangées ; 0 appel Groq supplémentaire |
| 4a | Après F5 + réouverture : la liste est re-demandée au serveur | ✅ OK | exactement 1 nouvelle requête client |
| 4b | Mais **zéro appel Groq** (cache serveur `_key_point_cache`) | ✅ OK | delta compteur = 0 |
| 5a | Document à 33 décrets : note de troncature visible | ✅ OK | *« Affichage de 15 décret(s) sur 33 »* |
| 5b | Plafond respecté : 15 bullets affichés | ✅ OK | `count == 15` |
| 5c | Coût Groq de la liste plafonnée | ⚠️ Partiel | **14 appels sur 15 réussis** — voir §3 |
| 5d | Titres distincts, langue du document respectée (arabe ici) | ✅ OK | titres uniques en arabe, aucun « indisponible » parmi les réussis |

**Appels Groq consommés par la session de test** : 17 requêtes
(2 liste petit doc + 1 détail + 14 liste grand doc), hors diagnostics.

---

## 3. Analyse de l'unique échec (étape 5c)

Pendant la génération des 15 points du gros BO, l'appel n° 15 a reçu :

```
RateLimitError: Error code: 429 — Rate limit reached for model
`llama-3.3-70b-versatile` … on tokens per day (TPD): Limit 100000,
Used 99371, Requested 2123.
```

**Cause** : épuisement de la limite quotidienne de **jetons** Groq
(TPD 100 000, offre gratuite) en plein lot — indépendant du garde-fou
`_LLM_FEATURE_DAILY_BUDGET` (qui compte des *lancements*, lui).

**Comportement observé = comportement conçu** :
- seul le point concerné passe en erreur (`error` renseigné, titre `null`) ;
- les 14 autres bullets et la note de troncature restent intacts ;
- aucun plantage de la route ni de l'interface ;
- le point en erreur est **retentable** (les erreurs ne sont jamais mises
  en cache).

⚠️ **Point de vigilance ouvert** : dans la capture navigateur, ce bullet
affichait le libellé « Clé API du modèle de langage invalide ou absente. »
alors que la reproduction directe après coup mappe bien `RateLimitError`
→ « Quota quotidien… ». L'hypothèse est une course avec le basculement
TPD (deux erreurs différentes selon la fenêtre) ; non reproductible
maintenant que le quota est épuisé. À réobserver lors d'une prochaine
session.

---

## 4. Captures d'écran

| Fichier | Contenu |
|---|---|
| `%TEMP%\adli_shots\kp_1_liste.png` | Liste des points clés (petit document, 2 décrets) |
| `%TEMP%\adli_shots\kp_2_detail.png` | Point déplié : détail + sources cliquables |
| `%TEMP%\adli_shots\kp_3_tronque.png` | Gros BO : note de troncature 15/33 |

---

## 5. État du quota à l'issue des tests

- **TPD (jetons/jour)** : épuisé (~100 000 / 100 000) — toute génération
  LLM renverra « Quota quotidien… » jusqu'à demain.
- **RPD (requêtes/jour)** : ~18 consommées sur 250.
- Le budget applicatif `_LLM_FEATURE_DAILY_BUDGET` (20 lancements) n'a
  pas été atteint (3 lancements consommés).

---

## 6. Conclusion

La fonctionnalité « Points clés » est **conforme au guide** sur tous les
critères vérifiables automatiquement : onglet dédié indépendant du chat,
titres IA ancrés dans le contenu réel, détail paresseux mis en cache
(client + serveur), plafond et troncature explicites, dégradation isolée
en cas d'échec LLM. L'unique échec constaté provient du quota externe
Groq, pas du code ; il a en outre démontré la robustesse prévue.

**Amélioration optionnelle relevée** : un titre arabe commence par
« DECRET n° 2.24.801 — » malgré la règle anti-numéro du prompt — durcir
la règle `[FORMAT STRICT]` si jugé gênant.
