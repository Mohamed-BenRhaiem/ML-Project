---
title: Eco-Smart Classifier
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Eco-Smart Classifier

Application de classification de déchets et estimation du prix de revente.

- **Régression** : HistGradientBoosting (R² test = 0.99)
- **Classification** : RandomForest (Accuracy test = 1.0)
- **Stack** : FastAPI + scikit-learn + frontend HTML/JS

## Routes

- `GET /` — Interface web (3 onglets : Dashboard / Manuel / NLP)
- `GET /metadata` — métadonnées du pipeline
- `GET /stats` — distribution du dataset
- `POST /predict` — classification + régression
- `POST /classify` — classification seule
- `GET /docs` — Swagger UI

## Déploiement

Hugging Face Spaces avec SDK Docker. Les artefacts modèles (`artifacts/*.joblib`)
sont copiés dans l'image au build.

Le code source est sur [GitHub](https://github.com/Mohamed-BenRhaiem/ML-Project).
