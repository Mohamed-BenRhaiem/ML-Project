"""
Tests du schéma du dataset brut.
Catégorie 🔴 IA interdite — rédigés à la main selon la charte.
"""

import pandas as pd
import pytest


REQUIRED_COLUMNS = {
    'Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite',
    'Source', 'Categorie', 'Rapport_Collecte', 'Prix_Revente',
}

EXPECTED_CATEGORIES = {'Plastique', 'Verre', 'Papier', 'Métal'}
EXPECTED_SOURCES = {'Centre_Tri', 'Collecte_Citoyenne', 'Usine_A', 'Usine_B'}


def test_dataset_exists(dataset_path):
    assert dataset_path.exists(), f"Dataset introuvable : {dataset_path}"


def test_required_columns_present(dataset_path):
    df = pd.read_csv(dataset_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Colonnes manquantes : {missing}"


def test_dataset_size(dataset_path):
    df = pd.read_csv(dataset_path)
    assert len(df) > 5000, f"Dataset trop petit : {len(df)} lignes"


def test_target_clf_classes(dataset_path):
    df = pd.read_csv(dataset_path)
    classes = set(df['Categorie'].dropna().unique())
    assert classes == EXPECTED_CATEGORIES, f"Classes inattendues : {classes}"


def test_source_values(dataset_path):
    df = pd.read_csv(dataset_path)
    sources = set(df['Source'].dropna().unique())
    assert sources == EXPECTED_SOURCES, f"Sources inattendues : {sources}"


def test_numeric_columns_are_numeric(dataset_path):
    df = pd.read_csv(dataset_path)
    for col in ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Prix_Revente']:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} n'est pas numérique"


def test_text_column_no_nan(dataset_path):
    df = pd.read_csv(dataset_path)
    # Le cahier des charges précise 0 NaN sur Rapport_Collecte
    assert df['Rapport_Collecte'].isna().sum() == 0


@pytest.mark.parametrize('col', ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite'])
def test_missing_rate_under_threshold(dataset_path, col):
    df = pd.read_csv(dataset_path)
    rate = df[col].isna().mean()
    # Cahier des charges : ~10 % maximum
    assert rate < 0.15, f"{col} a {rate:.1%} de NaN, > 15 %"
