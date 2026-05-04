"""
export_model.py
---------------
Lance ce script APRES ton training.
Il exporte le TF-IDF vectorizer + un modele Ridge en JSON
pour que l'app navigateur puisse les charger directement.

Fichiers produits :
  - tfidf_vectorizer.json
  - ridge_model.json

Usage : python export_model.py
"""

import pandas as pd
import numpy as np
import json

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ─── 1. CHARGEMENT ───────────────────────────────────────────────
df = pd.read_csv("dataset_ProjetML_2026.csv")

# FIX Bug 2 : num_cols ne contient PAS la cible (évite le target leakage)
feature_num = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
cat_cols    = ['Source', 'Categorie']   # Categorie ajoutée (Plastique/Verre/Papier/Métal)
text_col    = 'Rapport_Collecte'
target      = 'Prix_Revente'

# ─── 2. NETTOYAGE ────────────────────────────────────────────────
df = df.dropna(subset=[target])

# FIX Bug 2 : imputation KNN uniquement sur les features, PAS sur la cible
knn_imputer_num = KNNImputer(n_neighbors=5)
df[feature_num] = knn_imputer_num.fit_transform(df[feature_num])

# Categorical KNN imputation via ordinal encoding
ord_enc = OrdinalEncoder()
cat_encoded = ord_enc.fit_transform(df[cat_cols])
knn_imputer_cat = KNNImputer(n_neighbors=5)
cat_imputed = knn_imputer_cat.fit_transform(cat_encoded)
df[cat_cols] = ord_enc.inverse_transform(np.round(cat_imputed).astype(int))

df[text_col] = df[text_col].fillna("")

# ─── 3. SUPPRESSION OUTLIERS SUR LA CIBLE ────────────────────────
# Fait AVANT le split mais APRES nettoyage, sur y uniquement
y_all = df[target].values
Q1, Q3 = np.percentile(y_all, 25), np.percentile(y_all, 75)
IQR = Q3 - Q1
mask = (y_all >= Q1 - 1.5 * IQR) & (y_all <= Q3 + 1.5 * IQR)
df = df[mask].reset_index(drop=True)
print(f"Lignes après suppression outliers : {len(df)}")

# ─── 4. TRAIN / TEST SPLIT (avant tout fit de transformateurs) ───
# FIX Bug 1 : on split les index bruts AVANT de fitter scaler/tfidf/encoder
from sklearn.model_selection import train_test_split as tts

idx = np.arange(len(df))
idx_train, idx_test = tts(idx, test_size=0.2, random_state=42)

df_train = df.iloc[idx_train].reset_index(drop=True)
df_test  = df.iloc[idx_test].reset_index(drop=True)

# ─── 5. TF-IDF — fit sur train uniquement ────────────────────────
tfidf = TfidfVectorizer(max_features=100)
text_train = tfidf.fit_transform(df_train[text_col]).toarray()
text_test  = tfidf.transform(df_test[text_col]).toarray()

text_train_df = pd.DataFrame(text_train, columns=tfidf.get_feature_names_out())
text_test_df  = pd.DataFrame(text_test,  columns=tfidf.get_feature_names_out())

# ─── 6. ENCODAGE CATEGORIEL — fit sur train uniquement ───────────
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_train = encoder.fit_transform(df_train[cat_cols])
cat_test  = encoder.transform(df_test[cat_cols])

cat_train_df = pd.DataFrame(cat_train, columns=encoder.get_feature_names_out(cat_cols))
cat_test_df  = pd.DataFrame(cat_test,  columns=encoder.get_feature_names_out(cat_cols))

# ─── 7. STANDARDISATION — fit sur train uniquement ───────────────
scaler = StandardScaler()
num_train = pd.DataFrame(scaler.fit_transform(df_train[feature_num]), columns=feature_num)
num_test  = pd.DataFrame(scaler.transform(df_test[feature_num]),      columns=feature_num)

# ─── 8. ASSEMBLAGE ───────────────────────────────────────────────
X_train = pd.concat([num_train, cat_train_df, text_train_df], axis=1)
X_test  = pd.concat([num_test,  cat_test_df,  text_test_df],  axis=1)

y_train = df_train[target].values
y_test  = df_test[target].values

# ─── 9. ENTRAINEMENT Ridge ───────────────────────────────────────
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"R²  : {r2_score(y_test, preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")

# ─── 10. EXPORT TF-IDF ───────────────────────────────────────────
vocab = {str(k): int(v) for k, v in tfidf.vocabulary_.items()}
idf   = [float(x) for x in tfidf.idf_]

tfidf_export = {
    "vocabulary": vocab,
    "idf": idf,
    "max_features": 100
}

with open("tfidf_vectorizer.json", "w", encoding="utf-8") as f:
    json.dump(tfidf_export, f, ensure_ascii=False)
print("✓ tfidf_vectorizer.json sauvegardé")

# ─── 11. EXPORT MODELE Ridge ─────────────────────────────────────
cat_feature_names  = [str(x) for x in encoder.get_feature_names_out(cat_cols)]
tfidf_feature_names = [str(x) for x in tfidf.get_feature_names_out()]

# FIX Bug 3 : on exporte aussi les catégories réelles pour que le HTML soit cohérent
model_export = {
    "intercept": float(model.intercept_),
    "coefficients": [float(x) for x in model.coef_],
    "feature_names": feature_num + cat_feature_names + tfidf_feature_names,

    "scaler_mean":  [float(x) for x in scaler.mean_],
    "scaler_scale": [float(x) for x in scaler.scale_],
    "scaler_features": feature_num,

    "encoder_categories": {
        cat_cols[i]: [str(x) for x in encoder.categories_[i]]
        for i in range(len(cat_cols))
    }
}

with open("ridge_model.json", "w", encoding="utf-8") as f:
    json.dump(model_export, f, ensure_ascii=False)
print("✓ ridge_model.json sauvegardé")
print("\nCatégories Source dans le modèle :", model_export["encoder_categories"]["Source"])
print("Place ces 2 fichiers dans le même dossier que index.html")
