# Déploiement — Eco-Smart Classifier

Trois cibles de déploiement supportées : **Hugging Face Spaces** (recommandé, gratuit), **Render** (gratuit, plus de RAM), ou **Docker** quelconque.

---

## Option 1 — Hugging Face Spaces (recommandé)

### Pré-requis
- Un compte HF : https://huggingface.co
- `git lfs` (pour pousser les artefacts > 10 Mo)

### Étapes

```bash
# 1. Génère les artefacts si pas déjà fait
python nettoyage.py
python clustering.py

# 2. Crée un Space (SDK = Docker, port = 8000)
#    via https://huggingface.co/new-space

# 3. Clone le Space localement
git clone https://huggingface.co/spaces/<TON-USER>/eco-smart-classifier hf-space
cd hf-space

# 4. Copie les fichiers nécessaires
cp ../main.py ../nlp_utils.py ../index.html ../requirements.txt ../Dockerfile .
cp ../hf_space/README.md .            # frontmatter HF spécifique
cp -r ../artifacts ./artifacts

# 5. git lfs pour les .joblib (peuvent dépasser 10 Mo)
git lfs install
git lfs track "artifacts/*.joblib"

# 6. Push
git add .
git commit -m "Initial deploy"
git push
```

L'URL sera : `https://huggingface.co/spaces/<TON-USER>/eco-smart-classifier`

---

## Option 2 — Render

Render détecte le `Dockerfile` automatiquement.

```bash
# 1. Push sur GitHub (déjà fait)

# 2. Sur https://render.com :
#    New > Web Service > GitHub repo
#    Environment: Docker
#    Plan: Free
#    Auto-deploy: oui

# 3. Vérifie que les artefacts/ sont commités OU :
#    Build Command: python nettoyage.py
#    (cette option régénère à chaque build, plus long mais self-contained)
```

---

## Option 3 — Docker local / VPS

```bash
docker build -t eco-smart .
docker run -d -p 8000:8000 --name eco-smart eco-smart

# vérification
curl http://localhost:8000/metadata
```

L'image utilise `python:3.11-slim` + healthcheck sur `/metadata`.

---

## Validation post-déploiement

```bash
curl https://<URL>/metadata          # doit renvoyer 200 + JSON
curl -X POST https://<URL>/predict \
  -H "Content-Type: application/json" \
  -d '{"Source":"Centre_Tri","Poids":2.5,"Volume":5,"Conductivite":0.05,"Opacite":0.3,"Rigidite":3,"Rapport_Collecte":"plastique"}'
```

Tu devrais voir `predicted_category: "Plastique"` et un prix > 0.

---

## Notes de prod

- **Logs** : `docker logs eco-smart` ou onglet Logs du dashboard HF / Render
- **CORS** : actuellement ouvert à `*` dans `main.py` — à restreindre en prod
- **Modèles dans l'image** : ~10 Mo total, OK pour HF Spaces (limite 10 Go)
- **Drift monitoring** : nécessite Evidently AI (cf. Phase 4 si à implémenter)
