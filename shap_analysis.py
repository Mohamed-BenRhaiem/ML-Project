"""
shap_analysis.py
----------------
Bonus 7 du cahier des charges : analyse SHAP pour l'explicabilité du modèle multimodal.

  - TreeExplainer sur le RandomForest (classification de Categorie)
  - TreeExplainer sur le HistGradientBoosting (régression de Prix_Revente)
  - Summary plots (top features par impact moyen sur la sortie)
  - Save : artifacts/shap_classification.png + shap_regression.png + shap_summary.txt
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from sklearn.impute import KNNImputer

ARTIFACTS_DIR = Path("artifacts")

# ─── Chargement des artefacts ─────────────────────────────────────
scaler    = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
encoder   = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
tfidf     = joblib.load(ARTIFACTS_DIR / "tfidf.joblib")
reg_model = joblib.load(ARTIFACTS_DIR / "model.joblib")
clf_model = joblib.load(ARTIFACTS_DIR / "clf_model.joblib")
meta      = joblib.load(ARTIFACTS_DIR / "meta.joblib")

FEATURE_NUM_BASE = meta["feature_num_base"]
FEATURE_NUM      = meta["feature_num"]
CAT_COLS         = meta["cat_cols"]


# ─── Reconstruction d'un échantillon de test ──────────────────────
df = pd.read_csv("dataset_ProjetML_2026.csv")
df = df.dropna(subset=["Prix_Revente", "Categorie"]).reset_index(drop=True)

# Mêmes traitements que nettoyage.py
imp = KNNImputer(n_neighbors=4)
df[FEATURE_NUM_BASE] = imp.fit_transform(df[FEATURE_NUM_BASE])
df["Source"] = df["Source"].fillna(df["Source"].mode()[0])
df["Rapport_Collecte"] = df["Rapport_Collecte"].fillna("")

df["Densite"]           = df["Poids"] / (df["Volume"] + 1e-6)
df["Cond_x_Rigidite"]   = df["Conductivite"] * df["Rigidite"]
df["Opacite_x_Cond"]    = df["Opacite"] * df["Conductivite"]
df["Poids_x_Rigidite"]  = df["Poids"] * df["Rigidite"]

# IQR outliers
y_all = df["Prix_Revente"].values
Q1, Q3 = np.percentile(y_all, 25), np.percentile(y_all, 75)
mask = (y_all >= Q1 - 1.5 * (Q3 - Q1)) & (y_all <= Q3 + 1.5 * (Q3 - Q1))
df = df[mask].reset_index(drop=True)

# Échantillon (SHAP est lent → 200 lignes suffisent pour un summary plot)
SAMPLE = min(200, len(df))
df_sample = df.sample(SAMPLE, random_state=42).reset_index(drop=True)

num   = scaler.transform(df_sample[FEATURE_NUM])
cat   = encoder.transform(df_sample[CAT_COLS])
text  = tfidf.transform(df_sample["Rapport_Collecte"]).toarray()

cat_names   = list(encoder.get_feature_names_out(CAT_COLS))
text_names  = list(tfidf.get_feature_names_out())
all_names   = list(FEATURE_NUM) + cat_names + text_names

X = pd.DataFrame(np.concatenate([num, cat, text], axis=1), columns=all_names)
print(f"Échantillon SHAP : {X.shape}")


# ─── SHAP — Classification (RandomForest) ─────────────────────────
print("\n─── SHAP : classification ───")
explainer_clf = shap.TreeExplainer(clf_model)
shap_values_clf = explainer_clf.shap_values(X)
# RF multiclasse → liste / tenseur 3D selon version SHAP
if isinstance(shap_values_clf, list):
    print(f"  {len(shap_values_clf)} matrices SHAP (une par classe)")
    abs_means = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_clf], axis=0)
elif shap_values_clf.ndim == 3:
    # (n_samples, n_features, n_classes)
    print(f"  Tenseur 3D : {shap_values_clf.shape}")
    abs_means = np.abs(shap_values_clf).mean(axis=(0, 2))
else:
    abs_means = np.abs(shap_values_clf).mean(axis=0)

top_clf = sorted(zip(all_names, abs_means), key=lambda x: -x[1])[:15]

fig, ax = plt.subplots(figsize=(8, 6))
labels = [n for n, _ in top_clf][::-1]
values = [v for _, v in top_clf][::-1]
ax.barh(range(len(labels)), values, color="#2f7a2f")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("|SHAP value| moyen")
ax.set_title("Top 15 features SHAP — Classification (RandomForest)")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "shap_classification.png", dpi=120)
plt.close()
print(f"  ✓ {ARTIFACTS_DIR / 'shap_classification.png'}")


# ─── SHAP — Régression (HistGradientBoosting) ─────────────────────
print("\n─── SHAP : régression ───")
explainer_reg = shap.TreeExplainer(reg_model)
shap_values_reg = explainer_reg.shap_values(X)
abs_means_reg = np.abs(shap_values_reg).mean(axis=0)
top_reg = sorted(zip(all_names, abs_means_reg), key=lambda x: -x[1])[:15]

fig, ax = plt.subplots(figsize=(8, 6))
labels = [n for n, _ in top_reg][::-1]
values = [v for _, v in top_reg][::-1]
ax.barh(range(len(labels)), values, color="#246124")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("|SHAP value| moyen (€)")
ax.set_title("Top 15 features SHAP — Régression (HistGradientBoosting)")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "shap_regression.png", dpi=120)
plt.close()
print(f"  ✓ {ARTIFACTS_DIR / 'shap_regression.png'}")


# ─── Résumé texte ─────────────────────────────────────────────────
out_path = ARTIFACTS_DIR / "shap_summary.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("Analyse SHAP — Bonus 7 du cahier des charges\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Échantillon : {SAMPLE} lignes\n\n")
    f.write("─── Top 15 features — Classification (RandomForest) ───\n")
    for name, val in top_clf:
        f.write(f"  {val:8.4f}  {name}\n")
    f.write("\n─── Top 15 features — Régression (HistGradientBoosting) ───\n")
    for name, val in top_reg:
        f.write(f"  {val:8.4f}  {name}\n")

print(f"  ✓ {out_path}")
print("\nDone.")
