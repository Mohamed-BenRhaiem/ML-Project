"""
Tests des artefacts produits par nettoyage.py.
Vérifient la qualité post-imputation, la cohérence des modèles sauvegardés
et le seuil de performance minimal exigé par le cahier des charges (≥ 0.70 accuracy).
Catégorie 🔴 IA interdite — rédigés à la main.
"""

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def artifacts(artifacts_dir):
    if not artifacts_dir.exists():
        pytest.skip("artifacts/ absent — lance d'abord python nettoyage.py")
    needed = ["scaler", "encoder", "tfidf", "model", "clf_model", "meta"]
    out = {}
    for name in needed:
        path = artifacts_dir / f"{name}.joblib"
        if not path.exists():
            pytest.skip(f"{path} absent")
        out[name] = joblib.load(path)
    return out


def test_meta_contains_required_fields(artifacts):
    meta = artifacts["meta"]
    for key in ("feature_num", "cat_cols", "target_reg", "target_clf",
                "model_name", "clf_model_name", "knn_neighbors"):
        assert key in meta, f"Champ manquant dans meta : {key}"


def test_knn_neighbors_is_4(artifacts):
    """Le cahier des charges + consigne utilisateur exigent K=4."""
    assert artifacts["meta"]["knn_neighbors"] == 4


def test_categorie_not_in_features(artifacts):
    """Bug fix : Categorie doit être la cible, pas une feature."""
    cat_cols = artifacts["meta"]["cat_cols"]
    assert "Categorie" not in cat_cols, "Categorie ne doit PAS être une feature"
    assert artifacts["meta"]["target_clf"] == "Categorie"


def test_scaler_dimensions_match_features(artifacts):
    scaler = artifacts["scaler"]
    feature_num = artifacts["meta"]["feature_num"]
    assert len(scaler.mean_) == len(feature_num)


def test_encoder_handles_unknown(artifacts):
    encoder = artifacts["encoder"]
    # OneHotEncoder doit avoir handle_unknown='ignore' pour la prod
    assert encoder.handle_unknown == "ignore"


def test_tfidf_uses_french_tokenizer(artifacts):
    """Vérifie que le TF-IDF a bien le tokenizer français custom."""
    tfidf = artifacts["tfidf"]
    # Le tokenizer doit être une fonction (pas None / pas le default)
    assert tfidf.tokenizer is not None
    assert callable(tfidf.tokenizer)


def test_classifier_classes_match_target(artifacts):
    clf = artifacts["clf_model"]
    expected = {"Plastique", "Verre", "Papier", "Métal"}
    assert set(clf.classes_) == expected


def _filter_outliers_iqr(df, col):
    """Reproduit le filtrage IQR appliqué dans nettoyage.py."""
    y = df[col].values
    Q1, Q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = Q3 - Q1
    mask = (y >= Q1 - 1.5 * iqr) & (y <= Q3 + 1.5 * iqr)
    return df[mask]


def test_regression_threshold_on_test(dataset_path, artifacts):
    """
    Seuil de performance minimal : R² > 0.7 sur un échantillon du dataset
    (après filtrage IQR cohérent avec le pipeline d'entraînement).
    """
    from sklearn.metrics import r2_score

    df = pd.read_csv(dataset_path).dropna(subset=["Prix_Revente", "Categorie"])
    df = _filter_outliers_iqr(df, "Prix_Revente")
    df = df.sample(min(500, len(df)), random_state=123).copy()

    # Imputation rapide pour ce test seul (cohérent : KNN K=4)
    from sklearn.impute import KNNImputer
    feature_num_base = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    imp = KNNImputer(n_neighbors=4)
    df[feature_num_base] = imp.fit_transform(df[feature_num_base])
    df['Source'] = df['Source'].fillna(df['Source'].mode()[0])
    df['Rapport_Collecte'] = df['Rapport_Collecte'].fillna("")

    # Feature engineering
    df['Densite']           = df['Poids'] / (df['Volume'] + 1e-6)
    df['Cond_x_Rigidite']   = df['Conductivite'] * df['Rigidite']
    df['Opacite_x_Cond']    = df['Opacite'] * df['Conductivite']
    df['Poids_x_Rigidite']  = df['Poids'] * df['Rigidite']

    feature_num = artifacts["meta"]["feature_num"]
    cat_cols    = artifacts["meta"]["cat_cols"]

    num_scaled  = artifacts["scaler"].transform(df[feature_num])
    cat_encoded = artifacts["encoder"].transform(df[cat_cols])
    text_vec    = artifacts["tfidf"].transform(df['Rapport_Collecte']).toarray()

    feat_names = (list(feature_num)
                  + list(artifacts["encoder"].get_feature_names_out(cat_cols))
                  + list(artifacts["tfidf"].get_feature_names_out()))
    X = pd.DataFrame(np.concatenate([num_scaled, cat_encoded, text_vec], axis=1),
                     columns=feat_names)

    preds = artifacts["model"].predict(X)
    r2 = r2_score(df['Prix_Revente'].values, preds)
    assert r2 > 0.70, f"R² = {r2:.3f}, sous le seuil de 0.70"


def test_classification_threshold_on_test(dataset_path, artifacts):
    """Seuil minimal de classification : accuracy > 0.70 (cahier des charges)."""
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(dataset_path).dropna(subset=["Prix_Revente", "Categorie"])
    df = df.sample(min(500, len(df)), random_state=123).copy()

    from sklearn.impute import KNNImputer
    feature_num_base = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    imp = KNNImputer(n_neighbors=4)
    df[feature_num_base] = imp.fit_transform(df[feature_num_base])
    df['Source'] = df['Source'].fillna(df['Source'].mode()[0])
    df['Rapport_Collecte'] = df['Rapport_Collecte'].fillna("")

    df['Densite']           = df['Poids'] / (df['Volume'] + 1e-6)
    df['Cond_x_Rigidite']   = df['Conductivite'] * df['Rigidite']
    df['Opacite_x_Cond']    = df['Opacite'] * df['Conductivite']
    df['Poids_x_Rigidite']  = df['Poids'] * df['Rigidite']

    feature_num = artifacts["meta"]["feature_num"]
    cat_cols    = artifacts["meta"]["cat_cols"]

    num_scaled  = artifacts["scaler"].transform(df[feature_num])
    cat_encoded = artifacts["encoder"].transform(df[cat_cols])
    text_vec    = artifacts["tfidf"].transform(df['Rapport_Collecte']).toarray()

    feat_names = (list(feature_num)
                  + list(artifacts["encoder"].get_feature_names_out(cat_cols))
                  + list(artifacts["tfidf"].get_feature_names_out()))
    X = pd.DataFrame(np.concatenate([num_scaled, cat_encoded, text_vec], axis=1),
                     columns=feat_names)

    preds = artifacts["clf_model"].predict(X)
    acc = accuracy_score(df['Categorie'].values, preds)
    assert acc > 0.70, f"Accuracy = {acc:.3f}, sous le seuil de 0.70"
