"""
Tests de l'endpoint API FastAPI.
Catégorie 🔴 IA interdite — rédigés à la main selon la charte.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(artifacts_dir):
    if not artifacts_dir.exists():
        pytest.skip("artifacts/ absent — lance d'abord python nettoyage.py")
    import main
    return TestClient(main.app)


@pytest.fixture
def valid_payload():
    return {
        "Source": "Centre_Tri",
        "Poids": 5.0,
        "Volume": 10.0,
        "Conductivite": 0.05,
        "Opacite": 0.3,
        "Rigidite": 5.0,
        "Rapport_Collecte": "Lot de plastique récupéré, état correct",
    }


# ─── /metadata ───────────────────────────────────────────────────
def test_metadata_status_200(client):
    r = client.get("/metadata")
    assert r.status_code == 200


def test_metadata_returns_sources(client):
    r = client.get("/metadata")
    data = r.json()
    sources = data["categories"]["Source"]
    assert isinstance(sources, list)
    assert len(sources) == 4


def test_metadata_returns_target_classes(client):
    r = client.get("/metadata")
    data = r.json()
    assert set(data["target_classes"]) == {"Plastique", "Verre", "Papier", "Métal"}


def test_metadata_documents_knn_k(client):
    r = client.get("/metadata")
    assert r.json()["knn_neighbors"] == 4


# ─── /predict ────────────────────────────────────────────────────
def test_predict_valid_returns_200(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200, r.text


def test_predict_response_schema(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    data = r.json()
    for key in ("predicted_category", "class_probabilities", "price",
                "contrib_num", "contrib_cat", "contrib_nlp", "top_words"):
        assert key in data


def test_predict_category_is_valid(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    cat = r.json()["predicted_category"]
    assert cat in {"Plastique", "Verre", "Papier", "Métal"}


def test_predict_probabilities_sum_to_one(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    probs = r.json()["class_probabilities"]
    total = sum(p["proba"] for p in probs)
    assert abs(total - 1.0) < 1e-6


def test_predict_price_is_finite(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    import math
    price = r.json()["price"]
    assert math.isfinite(price)


def test_predict_unknown_source_returns_400(client, valid_payload):
    bad = dict(valid_payload, Source="UsineX")
    r = client.post("/predict", json=bad)
    assert r.status_code == 400


def test_predict_negative_weight_returns_422(client, valid_payload):
    """Pydantic v2 doit refuser Field(ge=0) avec valeur négative."""
    bad = dict(valid_payload, Poids=-1.0)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field_returns_422(client, valid_payload):
    bad = dict(valid_payload)
    del bad["Source"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


# ─── /classify ───────────────────────────────────────────────────
def test_classify_returns_200(client, valid_payload):
    r = client.post("/classify", json=valid_payload)
    assert r.status_code == 200


def test_classify_returns_probabilities(client, valid_payload):
    r = client.post("/classify", json=valid_payload)
    data = r.json()
    assert "predicted_category" in data
    assert "class_probabilities" in data
    assert len(data["class_probabilities"]) == 4
