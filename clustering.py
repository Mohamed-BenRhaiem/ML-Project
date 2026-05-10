"""
clustering.py
-------------
Module 3 du cahier des charges : analyse non-supervisée.

  - Ignore les targets (Categorie, Prix_Revente)
  - KMeans avec sélection de K via la méthode du coude (Elbow)
  - Visualisation 2D via PCA
  - Comparaison clusters trouvés vs Categorie réelle (à titre indicatif)

Sauvegarde :
  - artifacts/clustering_elbow.png
  - artifacts/clustering_pca.png
  - artifacts/clustering_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (utile pour CI / serveur)
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ─── 1. Chargement et préparation ─────────────────────────────────
df = pd.read_csv("dataset_ProjetML_2026.csv")
print(f"Dataset brut : {df.shape}")

feature_num = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']

# On peut clusteriser SANS les cibles : on les retire
# mais on garde Categorie pour comparaison a posteriori
df_work = df.dropna(subset=feature_num + ['Categorie']).copy()
print(f"Après drop NaN sur features + Categorie : {len(df_work)}")

# Imputation KNN (cohérent avec le reste du projet)
imputer = KNNImputer(n_neighbors=4)
X = imputer.fit_transform(df_work[feature_num])

# Standardisation (KMeans est sensible aux échelles)
X_scaled = StandardScaler().fit_transform(X)


# ─── 2. Méthode du coude (Elbow) ──────────────────────────────────
print("\n─── Méthode du coude (Elbow) ───")
K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels_k)
    silhouettes.append(sil)
    print(f"  K={k:2d} | inertie = {km.inertia_:8.1f} | silhouette = {sil:.4f}")

# Plot Elbow + Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(K_range), inertias, 'o-', color='#2f7a2f', linewidth=2)
ax1.set_xlabel('Nombre de clusters (K)')
ax1.set_ylabel('Inertie intra-clusters')
ax1.set_title("Méthode du coude")
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_range), silhouettes, 'o-', color='#246124', linewidth=2)
ax2.set_xlabel('Nombre de clusters (K)')
ax2.set_ylabel('Silhouette score')
ax2.set_title("Silhouette score par K")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "clustering_elbow.png", dpi=120)
plt.close()
print(f"  ✓ Sauvegardé : {ARTIFACTS_DIR / 'clustering_elbow.png'}")

# Choix de K : on prend K=4 par défaut (cohérent avec les 4 catégories réelles)
# Le K optimal selon silhouette peut différer ; on documente les deux.
optimal_k_silhouette = list(K_range)[int(np.argmax(silhouettes))]
chosen_k = 4
print(f"\n  K optimal (silhouette) : {optimal_k_silhouette}")
print(f"  K retenu (aligné avec les 4 catégories réelles) : {chosen_k}")


# ─── 3. KMeans final + visualisation PCA 2D ───────────────────────
kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
print(f"\n  PCA : variance expliquée par les 2 axes = {explained.sum()*100:.1f} % "
      f"({explained[0]*100:.1f} % + {explained[1]*100:.1f} %)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1 : clusters trouvés
scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                        cmap='viridis', s=12, alpha=0.6)
ax1.set_xlabel(f"PC1 ({explained[0]*100:.1f} %)")
ax1.set_ylabel(f"PC2 ({explained[1]*100:.1f} %)")
ax1.set_title(f"KMeans (K={chosen_k}) — clusters découverts")
plt.colorbar(scatter1, ax=ax1, label='cluster')

# Plot 2 : vraies catégories pour comparaison
cat_labels = df_work['Categorie'].astype('category').cat.codes
cat_names  = df_work['Categorie'].astype('category').cat.categories
scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=cat_labels,
                        cmap='tab10', s=12, alpha=0.6)
ax2.set_xlabel(f"PC1 ({explained[0]*100:.1f} %)")
ax2.set_ylabel(f"PC2 ({explained[1]*100:.1f} %)")
ax2.set_title("Vraies catégories (référence)")
cbar = plt.colorbar(scatter2, ax=ax2, ticks=range(len(cat_names)))
cbar.ax.set_yticklabels(cat_names)

plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "clustering_pca.png", dpi=120)
plt.close()
print(f"  ✓ Sauvegardé : {ARTIFACTS_DIR / 'clustering_pca.png'}")


# ─── 4. Adéquation clusters ↔ vraies catégories ───────────────────
# Adjusted Rand Index : 1 = parfait, 0 = aléatoire, négatif = pire qu'aléatoire
ari = adjusted_rand_score(cat_labels, cluster_labels)
print(f"\n  Adjusted Rand Index (clusters vs Categorie) : {ari:.4f}")
print("  Interprétation : 1 = adéquation parfaite, 0 = aléatoire")

# Table de contingence
contingency = pd.crosstab(
    pd.Series(cluster_labels, name='Cluster'),
    pd.Series(df_work['Categorie'].values, name='Categorie'),
)
print("\n  Table de contingence (Cluster × Categorie) :")
print(contingency.to_string())


# ─── 5. Sauvegarde du résumé ──────────────────────────────────────
summary_path = ARTIFACTS_DIR / "clustering_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Module 3 — Clustering non supervisé\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Échantillons : {len(X_scaled)}\n")
    f.write(f"Features : {feature_num}\n\n")
    f.write("Inertie / Silhouette par K :\n")
    for k, inert, sil in zip(K_range, inertias, silhouettes):
        f.write(f"  K={k}  inertie={inert:.1f}  silhouette={sil:.4f}\n")
    f.write(f"\nK optimal (silhouette) : {optimal_k_silhouette}\n")
    f.write(f"K retenu : {chosen_k}\n")
    f.write(f"Variance PCA 2D : {explained.sum()*100:.1f} %\n")
    f.write(f"Adjusted Rand Index : {ari:.4f}\n\n")
    f.write("Table de contingence :\n")
    f.write(contingency.to_string())
print(f"\n  ✓ Sauvegardé : {summary_path}")
print("\nDone.")
