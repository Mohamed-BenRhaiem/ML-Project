"""
nettoyage.py
------------
Pipeline conforme au cahier des charges :
  - Cibles : Categorie (classification), Prix_Revente (régression)
  - Features : Poids, Volume, Conductivite, Opacite, Rigidite, Source, Rapport_Collecte
  - Comparaison imputation (médiane / KNN / IterativeImputer) — choix K=4
  - NLP : stopwords FR + stemmer Snowball + TF-IDF uni+bigrams
  - Split 70:15:15 stratifié sur Categorie
  - GridSearchCV sur 2 problèmes : régression + classification

Sauvegarde dans artifacts/ : scaler, encoder, tfidf, model (régression),
clf_model (classification), meta.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import mlflow
import mlflow.sklearn

# Activer IterativeImputer (toujours expérimental dans sklearn)
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix,
)

# NLP : stopwords + stemmer Snowball partagés via nlp_utils.py
from nlp_utils import french_tokenizer

# ─── MLFLOW : tracking local + experiment ─────────────────────────
# Sauvegarde les runs dans ./mlruns/ (file backend)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("eco-smart-classifier")
USE_MLFLOW = os.environ.get("DISABLE_MLFLOW", "0") != "1"

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


def compare_imputers(df_raw, num_cols, target):
    print("\n─── Comparaison stratégies d'imputation ───")
    candidates = {
        'Median':           SimpleImputer(strategy='median'),
        'KNN(K=4)':         KNNImputer(n_neighbors=4),
        'IterativeImputer': IterativeImputer(max_iter=10, random_state=42),
    }
    results = {}
    y = df_raw[target].values
    for name, imp in candidates.items():
        t0 = time.time()
        X = imp.fit_transform(df_raw[num_cols])
        scaler_tmp = StandardScaler()
        X_scaled = scaler_tmp.fit_transform(X)
        scores = cross_val_score(
            Ridge(alpha=1.0), X_scaled, y,
            scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1,
        )
        rmse = -scores.mean()
        elapsed = time.time() - t0
        results[name] = rmse
        print(f"  {name:18s} RMSE Ridge-CV3 = {rmse:.4f}  ({elapsed:.1f}s)")

        if USE_MLFLOW:
            with mlflow.start_run(run_name=f"imputer_{name.replace('(', '_').replace(')', '')}"):
                mlflow.set_tag("task", "imputation_benchmark")
                mlflow.set_tag("imputer", name)
                mlflow.log_metric("ridge_cv_rmse_pre_outliers", rmse)
                mlflow.log_metric("imputation_time_seconds", elapsed)
    best = min(results, key=results.get)
    print(f"  → meilleur imputeur d'après ce benchmark : {best}")
    return results

imputation_benchmark = compare_imputers(df.copy(), feature_num_base, target_reg)


# ─── 2ter. IMPUTATION KNN (K=4) — choix retenu ───────────────────

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

# ─── 5-8. PIPELINE MULTIMODAL via ColumnTransformer ──────────────
# Module 5 du cahier des charges : fusion numérique + catégoriel + texte
# orchestrée par ColumnTransformer pour la reproductibilité.
#
# Note : TfidfVectorizer attend une 1D (Series), pas un DataFrame —
# on passe donc le NOM de colonne text_col (string) au lieu d'une liste.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_num),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('text', TfidfVectorizer(
            tokenizer=french_tokenizer,
            token_pattern=None,
            max_features=300,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            lowercase=False,
        ), text_col),
    ],
    remainder='drop',
    verbose_feature_names_out=False,
)

# Fit sur le train uniquement
preprocessor.fit(df_train)

# Extraction des transformateurs fittés (pour compat main.py + tests)
scaler  = preprocessor.named_transformers_['num']
encoder = preprocessor.named_transformers_['cat']
tfidf   = preprocessor.named_transformers_['text']

# Construction des matrices X via le ColumnTransformer
# Densification car TfidfVectorizer renvoie une matrice sparse
def _ct_transform(ct, df):
    out = ct.transform(df)
    if hasattr(out, 'toarray'):
        out = out.toarray()
    return out

feat_names = list(preprocessor.get_feature_names_out())
X_train = pd.DataFrame(_ct_transform(preprocessor, df_train), columns=feat_names)
X_val   = pd.DataFrame(_ct_transform(preprocessor, df_val),   columns=feat_names)
X_test  = pd.DataFrame(_ct_transform(preprocessor, df_test),  columns=feat_names)
print(f"\nVecteur final assemblé : {X_train.shape[1]} features ({len(feature_num)} num + "
      f"{len(encoder.get_feature_names_out(cat_cols))} cat + "
      f"{len(tfidf.get_feature_names_out())} TF-IDF)")

y_train_reg = df_train[target_reg].values
y_val_reg   = df_val[target_reg].values
y_test_reg  = df_test[target_reg].values

y_train_clf = df_train[target_clf].values
y_val_clf   = df_val[target_clf].values
y_test_clf  = df_test[target_clf].values


# ─── 9. HELPER GRIDSEARCH (avec logging MLflow par candidat) ─────
def run_gridsearch(candidates, X, y, scoring, metric_name, lower_is_better, task_tag):
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

        # ─── MLflow : un run par candidat ────────────────────────
        if USE_MLFLOW:
            with mlflow.start_run(run_name=f"{task_tag}_{name}"):
                mlflow.set_tag("task", task_tag)
                mlflow.set_tag("model_family", name)
                mlflow.log_params({f"hp_{k}": v for k, v in gs.best_params_.items()})
                mlflow.log_param("n_combos_tested", n_combos)
                mlflow.log_param("cv_folds", 3)
                mlflow.log_param("scoring", scoring)
                mlflow.log_metric(f"cv_{metric_name.lower().replace('-', '_')}", score)
                mlflow.log_metric("cv_time_seconds", elapsed)
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
    task_tag='regression',
)
best_reg_name = min(reg_results, key=lambda n: reg_results[n]['score'])
reg_model = reg_results[best_reg_name]['estimator']

print(f"\n✓ Meilleur régresseur : {best_reg_name}")
val_preds_reg  = reg_model.predict(X_val)
test_preds_reg = reg_model.predict(X_test)
val_r2_reg   = r2_score(y_val_reg, val_preds_reg)
val_rmse_reg = np.sqrt(mean_squared_error(y_val_reg, val_preds_reg))
test_r2_reg   = r2_score(y_test_reg, test_preds_reg)
test_rmse_reg = np.sqrt(mean_squared_error(y_test_reg, test_preds_reg))
print(f"  Val   R²={val_r2_reg:.4f} | RMSE={val_rmse_reg:.4f}")
print(f"  Test  R²={test_r2_reg:.4f} | RMSE={test_rmse_reg:.4f}")

# ─── MLflow : run final régression + Model Registry ──────────────
if USE_MLFLOW:
    with mlflow.start_run(run_name=f"FINAL_regression_{best_reg_name}"):
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("stage", "final")
        mlflow.set_tag("winner", best_reg_name)
        mlflow.log_metric("val_r2", val_r2_reg)
        mlflow.log_metric("val_rmse", val_rmse_reg)
        mlflow.log_metric("test_r2", test_r2_reg)
        mlflow.log_metric("test_rmse", test_rmse_reg)
        mlflow.log_param("test_size", len(X_test))
        # Enregistre le modèle dans le Model Registry
        mlflow.sklearn.log_model(
            reg_model, name="model",
            registered_model_name="eco_smart_regressor",
        )


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
    task_tag='classification',
)
best_clf_name = max(clf_results, key=lambda n: clf_results[n]['score'])
clf_model = clf_results[best_clf_name]['estimator']

print(f"\n✓ Meilleur classifieur : {best_clf_name}")
val_preds_clf  = clf_model.predict(X_val)
test_preds_clf = clf_model.predict(X_test)
val_acc_clf  = accuracy_score(y_val_clf, val_preds_clf)
val_f1_clf   = f1_score(y_val_clf, val_preds_clf, average='macro')
test_acc_clf = accuracy_score(y_test_clf, test_preds_clf)
test_f1_clf  = f1_score(y_test_clf, test_preds_clf, average='macro')
test_cm = confusion_matrix(y_test_clf, test_preds_clf, labels=clf_model.classes_)
print(f"  Val   Acc={val_acc_clf:.4f} | F1-macro={val_f1_clf:.4f}")
print(f"  Test  Acc={test_acc_clf:.4f} | F1-macro={test_f1_clf:.4f}")
print(f"\nMatrice de confusion (test) — classes {list(clf_model.classes_)}:")
print(test_cm)

# ─── MLflow : run final classification + Model Registry ──────────
if USE_MLFLOW:
    with mlflow.start_run(run_name=f"FINAL_classification_{best_clf_name}"):
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("stage", "final")
        mlflow.set_tag("winner", best_clf_name)
        mlflow.log_metric("val_accuracy", val_acc_clf)
        mlflow.log_metric("val_f1_macro", val_f1_clf)
        mlflow.log_metric("test_accuracy", test_acc_clf)
        mlflow.log_metric("test_f1_macro", test_f1_clf)
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_classes", len(clf_model.classes_))
        mlflow.log_param("classes", list(clf_model.classes_))
        # Matrice de confusion sauvegardée comme artefact texte
        cm_path = Path("artifacts") / "confusion_matrix.txt"
        cm_path.parent.mkdir(exist_ok=True)
        with open(cm_path, "w", encoding="utf-8") as f:
            f.write(f"Classes: {list(clf_model.classes_)}\n")
            f.write(np.array2string(test_cm))
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(
            clf_model, name="model",
            registered_model_name="eco_smart_classifier",
        )


# ─── 10. EXPORT JOBLIB POUR FASTAPI ──────────────────────────────
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")  # ColumnTransformer fitté
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
