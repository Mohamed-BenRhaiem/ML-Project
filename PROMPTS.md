# Journal des interactions avec les outils IA

Ce document liste les interactions significatives avec des assistants IA (Claude / ChatGPT) pendant le projet, en respectant la **Charte IA** du cahier des charges :

- 🔴 **IA interdite** : tests unitaires, fonctions EDA de base, première implémentation du prétraitement NLP
- 🟠 **IA structuration seulement** : configuration DVC/MLflow, débogage de code existant
- 🟢 **IA libre** : optimisation, Dockerfile, CI/CD, monitoring, API

Chaque entrée précise : **contexte → demande à l'IA → réponse de l'IA → décision finale (et pourquoi)**.

---

## 1. Migration JSON → FastAPI 🟢

**Contexte.** La version initiale exportait le modèle Ridge en JSON pour une inférence en JavaScript pur dans le navigateur. Cette approche limitait le projet à des modèles linéaires.

**Demande.** « On peut intégrer FastAPI au lieu de JSON ? »

**Réponse de l'IA.** L'IA a proposé une comparaison des trade-offs (statique vs serveur, modèles supportés, sécurité, latence) et a recommandé FastAPI dès qu'on souhaite tester des modèles non linéaires (RandomForest, GradientBoosting). L'IA a ensuite généré le squelette : `main.py` avec endpoints `/metadata`, `/predict`, `/`, `requirements.txt`, et l'adaptation du frontend pour `fetch` au lieu de `FileReader`.

**Décision.** Acceptée. Justification : sans backend, on ne pouvait pas exploiter les modèles non linéaires nécessaires pour atteindre des performances correctes. Le coût (un serveur Python à héberger) est acceptable pour le scope du projet.

---

## 2. Réduction du RMSE de la régression 🟢

**Contexte.** La RMSE initiale avec Ridge α=1 était de 0.85 € (R²=0.92). On voulait l'améliorer.

**Demande.** « RMSE est élevé (0.8), essayer de réduire. »

**Réponse de l'IA.** L'IA a proposé trois axes : feature engineering (densité, interactions), TF-IDF enrichi (300 features + bigrams + sublinear_tf), et comparaison Ridge vs HistGradientBoosting via cross-validation. Elle a aussi anticipé le besoin d'adapter `main.py` pour gérer les modèles non linéaires (contributions par ablation au lieu de produit scalaire).

**Décision.** Acceptée. Résultat : RMSE divisée par 2 (0.85 → 0.40). L'ajout d'interactions physiques (densité = poids/volume) et de bigrams a apporté l'essentiel du gain. Choix critique : on a accepté le passage à un modèle non linéaire en sachant que ça complique l'explicabilité, mais le gain de précision justifie.

---

## 3. Ajout SVR et RandomForest avec GridSearchCV 🟢

**Demande.** « Ajouter SVM et Random Forest et avec GridSearch, sélectionne le meilleur. »

**Réponse de l'IA.** L'IA a alerté que SVR avec kernel RBF est lent sur ~6500 échantillons × 300 features (O(n²)) et a proposé une grille minimale pour SVR (C=10, ε=0.2, γ=scale) afin de garder le temps total acceptable. Pour RandomForest et HistGradientBoosting, des grilles raisonnables (n_estimators, max_depth, min_samples_leaf, learning_rate, max_iter).

**Décision.** Acceptée. Le RandomForest est ressorti vainqueur (RMSE CV = 0.27) légèrement devant HistGradientBoosting (0.28). L'IA a logué les temps de chaque modèle : RF = 250 s, HistGB = 30 min — c'est un trade-off à considérer pour de futures itérations.

---

## 4. Identification d'un bug architectural critique 🟠

**Contexte.** Lors de l'analyse projet vs cahier des charges, l'IA a confronté la spec avec le code.

**Demande.** « Analyser le projet et trouver des améliorations », puis fourniture du cahier des charges.

**Réponse de l'IA.** L'IA a immédiatement identifié que `Categorie` était utilisée comme **feature** dans `cat_cols = ['Source', 'Categorie']` (ligne 22 de `nettoyage.py`), alors que le cahier des charges la définit explicitement comme **« la variable cible (Categorie) »**. C'était une erreur méthodologique : `Categorie` est censée être prédite par la classification, pas servir d'entrée pour la régression. Cela créait un *target leakage* indirect car `Categorie` corrèle fortement avec `Prix_Revente`.

**Décision.** Acceptée immédiatement. Refactor complet : `cat_cols = ['Source']` uniquement, ajout d'un pipeline classification séparé (RandomForest), split 70:15:15 stratifié sur `Categorie`. Justification : sans cette correction, les performances étaient en partie illusoires, et le projet ne pouvait pas répondre au Module 2 du cahier des charges (classification de Categorie).

**Apprentissage.** L'IA est précieuse pour confronter le code à un document de spécification. Sans cette confrontation, le bug serait passé inaperçu — il était cohérent en interne mais incompatible avec l'objectif déclaré.

---

## 5. Choix K=4 pour KNN Imputer 🟠

**Demande.** « Corriger les problèmes (avec K=4 en KNN). »

**Réponse de l'IA.** L'IA a documenté la justification dans le code : « K=4 offre un compromis entre sensibilité au bruit (K=1) et lissage excessif (K≥10) ».

**Décision.** Acceptée — c'était une consigne explicite du tuteur/utilisateur. Justification ajoutée par l'IA dans le rapport : avec ~9 500 lignes après drop des cibles, K=4 garantit une estimation locale sans surlissage.

---

## 6. Light mode interface 🟢

**Demande.** « Changer l'interface en light mode, liste roulante de Source est vide. »

**Réponse de l'IA.** L'IA a d'abord diagnostiqué la cause de la liste vide (uvicorn non redémarré après mise à jour des artefacts), puis a proposé une nouvelle palette light mode (off-white + vert forêt) en remplaçant les variables CSS.

**Décision.** Acceptée. Le diagnostic sur la liste vide était juste : il fallait redémarrer uvicorn. La palette est cohérente avec un thème éco-responsable.

---

## 7. Critique honnête du 100 % accuracy classification 🟠

**Contexte.** Le RandomForest atteint 100 % d'accuracy sur le test set. L'utilisateur a demandé si une prédiction concrète était correcte.

**Réponse de l'IA.** Au lieu de simplement valider, l'IA a noté que :
- Sur des inputs incohérents (texte « plastique » + conductivité 0.85 typique du métal), la confiance descend à ~60 %
- Le 100 % pourrait être dû au fait que les rapports textuels mentionnent souvent le matériau directement
- C'est une limite à investiguer (suppression des stopwords, comparaison avec d'autres vectoriseurs)

**Décision.** Acceptée. L'esprit critique a été ajouté à la section 11.1 du rapport. C'est exactement le type de discours que le cahier des charges valorise (« Un modèle imparfait mais rigoureusement analysé vaut plus qu'un modèle performant sans justification »).

---

## 8. Adaptation du rapport au projet réel 🟠

**Demande.** « Adapter le rapport selon notre projet et modification. »

**Réponse de l'IA.** Réécriture complète du rapport en Markdown reflétant :
- Le passage de Ridge → HistGB / RandomForest
- La correction du bug Categorie
- La migration JSON → FastAPI
- L'ajout du module classification
- L'esprit critique sur les 100 % d'accuracy

**Décision.** Acceptée. Le rapport contient maintenant une section §11.3 « Conformité au cahier des charges — gaps restants » qui liste honnêtement ce qui manque (DVC, MLflow, Docker, tests, monitoring, etc.).

---

## 9. Plan de redressement Phase 1 🟢

**Contexte.** Estimation honnête de la note actuelle : ~36/100. Le projet ne couvre que 2 modules sur 6 du cahier des charges.

**Demande.** « Commencer par Phase 1. »

**Réponse de l'IA.** Plan en 6 livrables (PROMPTS.md, stopwords FR + stemmer, comparaison imputation, module Clustering, Dockerfile, tests pytest), priorisé par ratio points/temps.

**Décision.** En cours d'exécution.

---

## Synthèse de notre relation à l'IA

Trois principes appliqués pendant tout le projet :

1. **Confronter avant d'accepter.** Une suggestion IA est validée par un test ou une lecture critique du cahier des charges, jamais sur la confiance.

2. **Documenter le « pourquoi », pas seulement le « quoi ».** Les commentaires `# FIX Bug 2` dans le code et les sections « Justification » dans le rapport tracent les décisions.

3. **Limiter le périmètre déléguable à l'IA selon la charte.**
   - Tests unitaires : rédigés sans IA (charte rouge)
   - Première implémentation du prétraitement NLP : version initiale faite sans IA, l'IA a juste optimisé après
   - Optimisation, Docker, CI/CD : zone verte, IA exploitée pleinement

L'IA a particulièrement aidé sur :
- La confrontation cahier des charges ↔ code (détection du bug Categorie)
- L'arbitrage technique (JSON vs FastAPI, Ridge vs HistGB)
- L'esprit critique sur les résultats (100 % accuracy = signal à creuser)

L'IA n'a **pas** suffi pour :
- Choisir K=4 (consigne externe)
- Comprendre le contexte métier (déchets, usines)
- Valider la pertinence des features physiques
