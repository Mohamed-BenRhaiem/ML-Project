# 🌱 Eco-Smart Classifier

Pipeline ML multimodal pour la **classification de déchets** (Plastique / Verre / Papier / Métal) et l'**estimation de leur prix de revente**, conforme au cahier des charges « Eco-Smart Classifier ».

[![tests](https://img.shields.io/badge/tests-44%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688)]()

---

## 🚀 Reproduire le pipeline en 3 commandes

```bash
pip install -r requirements.txt
python nettoyage.py             # entraîne classif + régression (~5 min)
uvicorn main:app --reload       # API + UI sur http://127.0.0.1:8000
```

---

## ✨ Fonctionnalités

- **Régression** du prix de revente (`HistGradientBoosting` tuné, R² test = 0.9931)
- **Classification** de la catégorie (`RandomForest` tuné, accuracy test = 1.0)
- **Clustering non-supervisé** (KMeans + Elbow + PCA, ARI = 0.70)
- **Pipeline NLP** avec stopwords français + stemmer Snowball
- **Comparaison** de 3 stratégies d'imputation (médiane / KNN / IterativeImputer)
- **Comparaison** de 4 vectoriseurs (BoW / TF-IDF / Word2Vec / FastText)
- **Pipeline multimodal** orchestré par `ColumnTransformer` sklearn
- **Analyse SHAP** sur classification + régression (TreeExplainer)
- **Monitoring** Evidently + Jensen-Shannon (text drift) + KS-test (numerical drift)
- **API REST** FastAPI avec endpoints `/predict`, `/classify`, `/metadata`, `/stats`
- **Interface web 3 onglets** (Dashboard / Manuel / NLP) en light mode
- **MLflow** : 11+ expériences trackées + Model Registry (eco_smart_regressor v1, eco_smart_classifier v1)
- **DVC pipeline** (6 stages : train → cluster → vectoriseurs → SHAP → monitoring → tests)
- **Tests** pytest (44 tests, coverage 85 %)
- **Dockerfile** prêt à build + GitHub Actions CI

---

## 📁 Structure du projet

```
ML_Project/
├── nettoyage.py                   # Pipeline d'entraînement (ColumnTransformer + GridSearch)
├── clustering.py                  # Module 3 : KMeans + Elbow + PCA
├── benchmark_vectorizers.py       # Module 4 : BoW vs TF-IDF vs W2V vs FastText
├── shap_analysis.py               # Bonus 7 : SHAP TreeExplainer
├── monitoring.py                  # Module 6 : Evidently + JS + KS-test
├── nlp_utils.py                   # Tokenizer FR + stopwords + stemmer
├── main.py                        # Backend FastAPI
├── index.html                     # Frontend 3 onglets
├── tests/                         # 44 tests pytest, coverage 85 %
├── artifacts/                     # Modèles + rapports (générés)
│   ├── preprocessor.joblib        # ColumnTransformer fitté (Module 5)
│   ├── scaler.joblib, encoder.joblib, tfidf.joblib
│   ├── model.joblib               # HistGradientBoosting
│   ├── clf_model.joblib           # RandomForest
│   ├── meta.joblib
│   ├── clustering_*.png/.txt
│   ├── vectorizer_benchmark.txt
│   ├── shap_*.png + shap_summary.txt
│   └── monitoring_report.html + monitoring_summary.txt
├── mlruns/                        # MLflow tracking (généré)
├── .github/workflows/ci.yml       # GitHub Actions
├── dvc.yaml                       # Pipeline DVC (6 stages)
├── Dockerfile
├── DEPLOY.md                      # Guide déploiement HF Spaces / Render
├── PROMPTS.md                     # Journal des interactions IA (obligatoire)
└── rapport_ML_Projet.md           # Rapport technique
```

---

## 🧪 Lancer les tests

```bash
pytest tests/                    # tous les tests (44)
pytest tests/test_api.py -v      # uniquement les tests API
pytest --cov=. --cov-report=term # avec coverage
```

---

## 📊 MLflow tracking

```bash
mlflow ui                        # interface sur http://127.0.0.1:5000
```

**Expériences trackées (≥ 8 runs) :**
- 3 imputers benchmarkés (Median, KNN(K=4), IterativeImputer)
- 4 régresseurs comparés (Ridge, SVR, RandomForest, HistGradientBoosting)
- 2 classifieurs comparés (LogisticRegression, RandomForest)
- 2 runs finaux (régression + classification gagnants) avec enregistrement au Model Registry

---

## 🐳 Docker

```bash
python nettoyage.py              # générer artifacts/ d'abord
docker build -t eco-smart .
docker run -p 8000:8000 eco-smart
```

L'image expose le port 8000 et inclut un healthcheck sur `/metadata`.

---

## 🔬 Module Clustering

```bash
python clustering.py
```

Génère :
- `artifacts/clustering_elbow.png` — courbes inertie + silhouette pour K ∈ [2, 10]
- `artifacts/clustering_pca.png` — visualisation 2D PCA des clusters découverts vs vraies catégories
- `artifacts/clustering_summary.txt` — résumé textuel + table de contingence

**Résultats :**
- K optimal silhouette = 5 ; K retenu = 4 (aligné avec les 4 catégories réelles)
- Adjusted Rand Index = 0.6992 vs vraies catégories
- PCA 2D : 68.2 % de variance expliquée

---

## 📈 Résultats principaux

### Régression — Prix de revente

| Modèle | CV RMSE | Test R² | Test RMSE |
|---|---|---|---|
| Ridge | 0.7740 | — | — |
| SVR (rbf) | 0.6715 | — | — |
| RandomForest | 0.2890 | — | — |
| **HistGradientBoosting** ✅ | **0.2722** | **0.9931** | **0.2643 €** |

### Classification — Categorie

| Modèle | F1-macro CV | Test Accuracy | Test F1-macro |
|---|---|---|---|
| LogisticRegression | 0.9992 | — | — |
| **RandomForest** ✅ | **1.0000** | **1.0000** | **1.0000** |

⚠️ Le 100 % de classification doit être interprété avec prudence — le rapport textuel mentionne souvent le matériau directement. Voir §11.1 du rapport.

---

## 🔧 Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | Backend MLflow |
| `DISABLE_MLFLOW` | `0` | Si `1`, désactive le tracking MLflow |
| `PYTHONIOENCODING` | — | Mettre à `utf-8` sur Windows |

---

## 📚 Documentation

- 📄 [Rapport technique complet](rapport_ML_Projet.md)
- 🤖 [Journal des interactions IA (PROMPTS.md)](PROMPTS.md)
- 📋 Cahier des charges « Eco-Smart Classifier » (Master 2 — 2025/2026)

---

## 👥 Auteurs

- Bahloul Fares
- Ben Rhaiem Mohamed

**Dépôt :** https://github.com/Mohamed-BenRhaiem/ML-Project
