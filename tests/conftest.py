"""
conftest.py
-----------
Fixtures partagées pour les tests pytest.
"""

import sys
from pathlib import Path

# Permet aux tests d'importer main, nlp_utils, etc. depuis la racine du projet
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def artifacts_dir(project_root):
    return project_root / "artifacts"


@pytest.fixture(scope="session")
def dataset_path(project_root):
    return project_root / "dataset_ProjetML_2026.csv"
