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

from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

# ─── 1. CHARGEMENT ───────────────────────────────────────────────
df = pd.read_csv("dataset_ProjetML_2026.csv")

num_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Prix_Revente']
cat_cols = ['Source']
text_col  = 'Rapport_Collecte'
target    = 'Prix_Revente'

# ─── 2. NETTOYAGE ────────────────────────────────────────────────
df = df.dropna(subset=[target])

# Numeric imputation (KNN replaces IterativeImputer)
knn_imputer_num = KNNImputer(n_neighbors=5)
df[num_cols] = knn_imputer_num.fit_transform(df[num_cols])

# Categorical KNN imputation via ordinal encoding
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
cat_encoded = ord_enc.fit_transform(df[cat_cols])
knn_imputer_cat = KNNImputer(n_neighbors=5)
cat_imputed = knn_imputer_cat.fit_transform(cat_encoded)
df[cat_cols] = ord_enc.inverse_transform(np.round(cat_imputed).astype(int))

df[text_col] = df[text_col].fillna("")

# ─── 3. TF-IDF sur Rapport_Collecte ──────────────────────────────
tfidf = TfidfVectorizer(max_features=100)
text_features = tfidf.fit_transform(df[text_col]).toarray()
text_df = pd.DataFrame(text_features, columns=tfidf.get_feature_names_out())

# ─── 4. ENCODAGE CATEGORIEL ──────────────────────────────────────
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(df[cat_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))

# ─── 5. ASSEMBLAGE ───────────────────────────────────────────────
feature_num = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[feature_num]),
    columns=feature_num
)

X = pd.concat([
    df_scaled.reset_index(drop=True),
    encoded_cat_df.reset_index(drop=True),
    text_df.reset_index(drop=True)
], axis=1)

y = df[target].values

# ─── Suppression outliers sur la cible ───────────────────────────
Q1, Q3 = np.percentile(y, 25), np.percentile(y, 75)
IQR = Q3 - Q1
mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
X, y = X[mask], y[mask]
print(f"Lignes après suppression outliers : {len(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── 6. ENTRAINEMENT Ridge ───────────────────────────────────────
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"R²  : {r2_score(y_test, preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")

# ─── 7. EXPORT TF-IDF ────────────────────────────────────────────
# Convertir les clés numpy int64 → int Python natif
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

# ─── 8. EXPORT MODELE Ridge ──────────────────────────────────────
cat_feature_names = [str(x) for x in encoder.get_feature_names_out(cat_cols)]
tfidf_feature_names = [str(x) for x in tfidf.get_feature_names_out()]

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
print("\nPlace ces 2 fichiers dans le même dossier que index.html")