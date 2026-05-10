"""
monitoring.py
-------------
Module 6 (MLOps) du cahier des charges : monitoring & data drift.

  - Génère un rapport HTML Evidently (data drift + data summary)
  - Calcule la divergence Jensen-Shannon sur le rapport textuel (text drift)
  - Sauvegarde dans artifacts/

Simulation de drift : on prend deux moitiés du dataset comme reference (entraînement)
vs current (production). En production réelle, `current` serait alimenté
par les requêtes /predict reçues.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset

from nlp_utils import french_tokenizer

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ─── 1. Chargement et split reference / current ───────────────────
df = pd.read_csv("dataset_ProjetML_2026.csv")
df = df.dropna(subset=["Categorie", "Prix_Revente"]).reset_index(drop=True)

# Pour simuler du drift : 50/50 mais shuffle pour éviter biais d'ordre
df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
n_ref = len(df_shuffled) // 2
ref     = df_shuffled.iloc[:n_ref].copy()
current = df_shuffled.iloc[n_ref:].copy()

print(f"Reference : {len(ref)} | Current : {len(current)}")


# ─── 2. Rapport Evidently — features numériques + cibles ─────────
num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
cat_cols = ["Source", "Categorie"]

monitored_cols = num_cols + cat_cols

ref_eval = ref[monitored_cols].copy()
cur_eval = current[monitored_cols].copy()

# DataDefinition : précise quelles colonnes sont quoi
data_def = DataDefinition(
    numerical_columns=num_cols,
    categorical_columns=cat_cols,
)
ref_ds = Dataset.from_pandas(ref_eval, data_definition=data_def)
cur_ds = Dataset.from_pandas(cur_eval, data_definition=data_def)

print("\n─── Génération du rapport Evidently ───")
report = Report(metrics=[
    DataDriftPreset(),
    DataSummaryPreset(),
])
result = report.run(reference_data=ref_ds, current_data=cur_ds)

html_path = ARTIFACTS_DIR / "monitoring_report.html"
result.save_html(str(html_path))
print(f"  ✓ {html_path}")


# ─── 3. Jensen-Shannon sur le texte (Module 6 — text drift) ──────
print("\n─── Jensen-Shannon divergence sur Rapport_Collecte ───")

# Vocabulaire commun fitté sur les deux corpus
all_text = pd.concat([ref["Rapport_Collecte"].fillna(""),
                       current["Rapport_Collecte"].fillna("")]).tolist()
cv = CountVectorizer(
    tokenizer=french_tokenizer, token_pattern=None,
    lowercase=False, min_df=2, max_features=500,
)
cv.fit(all_text)

def text_dist(texts):
    """Distribution des termes (probabilités normalisées)."""
    counts = cv.transform(texts.fillna("")).sum(axis=0)
    arr = np.asarray(counts).flatten().astype(float)
    total = arr.sum()
    return arr / total if total > 0 else arr

p = text_dist(ref["Rapport_Collecte"])
q = text_dist(current["Rapport_Collecte"])

# scipy.spatial.distance.jensenshannon retourne la racine carrée de la divergence
js_distance  = jensenshannon(p, q, base=2)  # entre 0 et 1
js_divergence = js_distance ** 2

# Seuil empirique : <0.05 = drift négligeable, >0.1 = drift significatif
status = "négligeable" if js_divergence < 0.05 else "modéré" if js_divergence < 0.1 else "SIGNIFICATIF"
print(f"  JS distance   : {js_distance:.4f}")
print(f"  JS divergence : {js_divergence:.4f}")
print(f"  Statut        : drift {status}")


# ─── 4. Détection drift par feature numérique (KS-like simple) ────
from scipy.stats import ks_2samp
print("\n─── Test Kolmogorov-Smirnov par feature numérique ───")
ks_results = {}
for col in ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]:
    ref_vals = ref[col].dropna().values
    cur_vals = current[col].dropna().values
    stat, pval = ks_2samp(ref_vals, cur_vals)
    drift = "❗ drift" if pval < 0.05 else "OK"
    ks_results[col] = {"stat": float(stat), "pvalue": float(pval), "drift": drift}
    print(f"  {col:14s}  KS stat = {stat:.4f}  p = {pval:.4f}  {drift}")


# ─── 5. Résumé texte ─────────────────────────────────────────────
out_path = ARTIFACTS_DIR / "monitoring_summary.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("Module 6 — Monitoring & Drift Detection\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Reference : {len(ref)} lignes\n")
    f.write(f"Current   : {len(current)} lignes\n\n")
    f.write(f"Rapport HTML Evidently : {html_path}\n\n")
    f.write("─── Text drift (Jensen-Shannon) ───\n")
    f.write(f"  JS distance   : {js_distance:.4f}\n")
    f.write(f"  JS divergence : {js_divergence:.4f}\n")
    f.write(f"  Statut        : drift {status}\n\n")
    f.write("─── Test KS par feature numérique ───\n")
    for col, r in ks_results.items():
        f.write(f"  {col:14s}  KS = {r['stat']:.4f}  p = {r['pvalue']:.4f}  {r['drift']}\n")
print(f"\n  ✓ {out_path}")
print("\nDone.")
