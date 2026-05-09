"""
nettoyage.py
------------
Pipeline conforme au cahier des charges :
  - Cibles : Categorie (classification), Prix_Revente (régression)
  - Features : Poids, Volume, Conductivite, Opacite, Rigidite, Source, Rapport_Collecte
  - Imputation KNN (K=4)
  - Split 70:15:15 stratifié sur Categorie
  - GridSearchCV sur 2 problèmes : régression + classification

Sauvegarde dans artifacts/ : scaler, encoder, tfidf, model (régression),
clf_model (classification), meta.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix,
)

# ─── 1. CHARGEMENT ───────────────────────────────────────────────
df = pd.read_csv("dataset_ProjetML_2026.csv")
print(f"Dataset brut : {df.shape}")

feature_num_base = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
cat_cols   = ['Source']            # FIX : Categorie n'est PAS une feature
text_col   = 'Rapport_Collecte'
target_reg = 'Prix_Revente'
target_clf = 'Categorie'

# ─── 2. NETTOYAGE — drop NaN sur les cibles ──────────────────────
df = df.dropna(subset=[target_reg, target_clf]).reset_index(drop=True)
print(f"Après drop NaN sur cibles ({target_reg}, {target_clf}) : {len(df)}")

# ─── 2bis. IMPUTATION KNN (K=4) sur features numériques ──────────
knn_imputer = KNNImputer(n_neighbors=4)
df[feature_num_base] = knn_imputer.fit_transform(df[feature_num_base])

# Source manquant → imputation par mode (catégoriel simple)
df['Source'] = df['Source'].fillna(df['Source'].mode()[0])

df[text_col] = df[text_col].fillna("")

# ─── 2ter. FEATURE ENGINEERING ───────────────────────────────────
df['Densite']           = df['Poids'] / (df['Volume'] + 1e-6)
df['Cond_x_Rigidite']   = df['Conductivite'] * df['Rigidite']
df['Opacite_x_Cond']    = df['Opacite'] * df['Conductivite']
df['Poids_x_Rigidite']  = df['Poids'] * df['Rigidite']

feature_num = feature_num_base + [
    'Densite', 'Cond_x_Rigidite', 'Opacite_x_Cond', 'Poids_x_Rigidite'
]

# ─── 3. OUTLIERS sur la cible de régression ──────────────────────
y_all = df[target_reg].values
Q1, Q3 = np.percentile(y_all, 25), np.percentile(y_all, 75)
IQR = Q3 - Q1
mask = (y_all >= Q1 - 1.5 * IQR) & (y_all <= Q3 + 1.5 * IQR)
df = df[mask].reset_index(drop=True)
print(f"Après suppression outliers Prix_Revente : {len(df)}")

# ─── 4. SPLIT 70:15:15 stratifié sur Categorie ───────────────────
df_trainval, df_test = train_test_split(
    df, test_size=0.15, stratify=df[target_clf], random_state=42
)
df_train, df_val = train_test_split(
    df_trainval, test_size=0.15 / 0.85,
    stratify=df_trainval[target_clf], random_state=42,
)
print(f"\nSplit  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
print(f"Ratios  Train: {len(df_train)/len(df):.0%} | Val: {len(df_val)/len(df):.0%} | Test: {len(df_test)/len(df):.0%}")

# ─── 5. TF-IDF — fit train uniquement ────────────────────────────
tfidf = TfidfVectorizer(
    max_features=300, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
)
text_train = tfidf.fit_transform(df_train[text_col]).toarray()
text_val   = tfidf.transform(df_val[text_col]).toarray()
text_test  = tfidf.transform(df_test[text_col]).toarray()

text_feat_names = list(tfidf.get_feature_names_out())
text_train_df = pd.DataFrame(text_train, columns=text_feat_names)
text_val_df   = pd.DataFrame(text_val,   columns=text_feat_names)
text_test_df  = pd.DataFrame(text_test,  columns=text_feat_names)

# ─── 6. ENCODAGE One-Hot sur Source uniquement ───────────────────
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_train = encoder.fit_transform(df_train[cat_cols])
cat_val   = encoder.transform(df_val[cat_cols])
cat_test  = encoder.transform(df_test[cat_cols])

cat_feat_names = list(encoder.get_feature_names_out(cat_cols))
cat_train_df = pd.DataFrame(cat_train, columns=cat_feat_names)
cat_val_df   = pd.DataFrame(cat_val,   columns=cat_feat_names)
cat_test_df  = pd.DataFrame(cat_test,  columns=cat_feat_names)

# ─── 7. STANDARDISATION numérique ────────────────────────────────
scaler = StandardScaler()
num_train = pd.DataFrame(scaler.fit_transform(df_train[feature_num]), columns=feature_num)
num_val   = pd.DataFrame(scaler.transform(df_val[feature_num]),       columns=feature_num)
num_test  = pd.DataFrame(scaler.transform(df_test[feature_num]),      columns=feature_num)

# ─── 8. ASSEMBLAGE ───────────────────────────────────────────────
def assemble(num, cat, txt):
    return pd.concat(
        [num.reset_index(drop=True), cat.reset_index(drop=True), txt.reset_index(drop=True)],
        axis=1,
    )

X_train = assemble(num_train, cat_train_df, text_train_df)
X_val   = assemble(num_val,   cat_val_df,   text_val_df)
X_test  = assemble(num_test,  cat_test_df,  text_test_df)

y_train_reg = df_train[target_reg].values
y_val_reg   = df_val[target_reg].values
y_test_reg  = df_test[target_reg].values

y_train_clf = df_train[target_clf].values
y_val_clf   = df_val[target_clf].values
y_test_clf  = df_test[target_clf].values


# ─── 9. HELPER GRIDSEARCH ────────────────────────────────────────
def run_gridsearch(candidates, X, y, scoring, metric_name, lower_is_better):
    print(f"\n─── GridSearch ({metric_name}, 3-fold CV) ───")
    results = {}
    for name, cfg in candidates.items():
        n_combos = int(np.prod([len(v) for v in cfg['params'].values()]))
        print(f"\n[{name}] {n_combos} combos × 3 folds = {n_combos*3} fits")
        t0 = time.time()
        gs = GridSearchCV(
            cfg['estimator'], cfg['params'],
            cv=3, scoring=scoring, n_jobs=-1,
        )
        gs.fit(X, y)
        elapsed = time.time() - t0
        score = -gs.best_score_ if lower_is_better else gs.best_score_
        results[name] = {
            'score': score,
            'best_params': gs.best_params_,
            'estimator': gs.best_estimator_,
        }
        print(f"  best params : {gs.best_params_}")
        print(f"  CV {metric_name} : {score:.4f}")
        print(f"  temps       : {elapsed:.1f}s")
    return results


# ─── 9a. RÉGRESSION : Prix_Revente ───────────────────────────────
print("\n" + "═" * 60)
print("RÉGRESSION (cible : Prix_Revente)")
print("═" * 60)

reg_candidates = {
    'Ridge': {
        'estimator': Ridge(),
        'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
    },
    'SVR': {
        'estimator': SVR(kernel='rbf'),
        'params': {'C': [10.0], 'gamma': ['scale'], 'epsilon': [0.2]},
    },
    'RandomForest': {
        'estimator': RandomForestRegressor(random_state=42, n_jobs=1),
        'params': {
            'n_estimators': [200],
            'max_depth': [None, 20],
            'min_samples_leaf': [2, 5],
        },
    },
    'HistGradientBoosting': {
        'estimator': HistGradientBoostingRegressor(random_state=42),
        'params': {
            'learning_rate': [0.05, 0.1],
            'max_depth': [6, 8],
            'max_iter': [300],
            'min_samples_leaf': [20],
            'l2_regularization': [1.0],
        },
    },
}

reg_results = run_gridsearch(
    reg_candidates, X_train, y_train_reg,
    scoring='neg_root_mean_squared_error',
    metric_name='RMSE',
    lower_is_better=True,
)
best_reg_name = min(reg_results, key=lambda n: reg_results[n]['score'])
reg_model = reg_results[best_reg_name]['estimator']

print(f"\n✓ Meilleur régresseur : {best_reg_name}")
preds_val  = reg_model.predict(X_val)
preds_test = reg_model.predict(X_test)
print(f"  Val   R²={r2_score(y_val_reg, preds_val):.4f} | RMSE={np.sqrt(mean_squared_error(y_val_reg, preds_val)):.4f}")
print(f"  Test  R²={r2_score(y_test_reg, preds_test):.4f} | RMSE={np.sqrt(mean_squared_error(y_test_reg, preds_test)):.4f}")


# ─── 9b. CLASSIFICATION : Categorie ──────────────────────────────
print("\n" + "═" * 60)
print("CLASSIFICATION (cible : Categorie)")
print("═" * 60)

clf_candidates = {
    'LogisticRegression': {
        'estimator': LogisticRegression(max_iter=2000, random_state=42),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs'],
        },
    },
    'RandomForest': {
        'estimator': RandomForestClassifier(random_state=42, n_jobs=1),
        'params': {
            'n_estimators': [200],
            'max_depth': [None, 20],
            'min_samples_leaf': [2, 5],
        },
    },
}

clf_results = run_gridsearch(
    clf_candidates, X_train, y_train_clf,
    scoring='f1_macro',
    metric_name='F1-macro',
    lower_is_better=False,
)
best_clf_name = max(clf_results, key=lambda n: clf_results[n]['score'])
clf_model = clf_results[best_clf_name]['estimator']

print(f"\n✓ Meilleur classifieur : {best_clf_name}")
preds_val  = clf_model.predict(X_val)
preds_test = clf_model.predict(X_test)
print(f"  Val   Acc={accuracy_score(y_val_clf, preds_val):.4f} | F1-macro={f1_score(y_val_clf, preds_val, average='macro'):.4f}")
print(f"  Test  Acc={accuracy_score(y_test_clf, preds_test):.4f} | F1-macro={f1_score(y_test_clf, preds_test, average='macro'):.4f}")
print(f"\nMatrice de confusion (test) — classes {list(clf_model.classes_)}:")
print(confusion_matrix(y_test_clf, preds_test, labels=clf_model.classes_))


# ─── 10. EXPORT JOBLIB POUR FASTAPI ──────────────────────────────
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

joblib.dump(scaler,     ARTIFACTS_DIR / "scaler.joblib")
joblib.dump(encoder,    ARTIFACTS_DIR / "encoder.joblib")
joblib.dump(tfidf,      ARTIFACTS_DIR / "tfidf.joblib")
joblib.dump(reg_model,  ARTIFACTS_DIR / "model.joblib")
joblib.dump(clf_model,  ARTIFACTS_DIR / "clf_model.joblib")

joblib.dump(
    {
        "feature_num_base": feature_num_base,
        "feature_num": feature_num,
        "cat_cols": cat_cols,
        "text_col": text_col,
        "target_reg": target_reg,
        "target_clf": target_clf,
        "model_name": best_reg_name,
        "is_linear": hasattr(reg_model, 'coef_'),
        "clf_model_name": best_clf_name,
        "clf_classes": [str(c) for c in clf_model.classes_],
        "knn_neighbors": 4,
        "split_ratios": "70/15/15 stratifié sur Categorie",
    },
    ARTIFACTS_DIR / "meta.joblib",
)

print("\n✓ Artefacts sauvegardés dans artifacts/")
print("  scaler.joblib, encoder.joblib, tfidf.joblib,")
print("  model.joblib (régression), clf_model.joblib (classification), meta.joblib")
print(f"\nSource possibles : {list(encoder.categories_[cat_cols.index('Source')])}")
print(f"Categorie possibles : {list(clf_model.classes_)}")
print("\nLance maintenant : uvicorn main:app --reload")
