"""
main.py
-------
Backend FastAPI :
  - GET  /metadata   → catégories Source + classes Categorie + features
  - POST /predict    → prédit Categorie (classif) + Prix_Revente (régression)
  - POST /classify   → uniquement Categorie + probabilités
  - GET  /           → index.html

Lance : uvicorn main:app --reload
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path("artifacts")
if not ARTIFACTS_DIR.exists():
    raise RuntimeError("Dossier 'artifacts/' introuvable. Lance : python nettoyage.py")

scaler    = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
encoder   = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
tfidf     = joblib.load(ARTIFACTS_DIR / "tfidf.joblib")
reg_model = joblib.load(ARTIFACTS_DIR / "model.joblib")
clf_model = joblib.load(ARTIFACTS_DIR / "clf_model.joblib")
meta      = joblib.load(ARTIFACTS_DIR / "meta.joblib")

FEATURE_NUM_BASE = meta["feature_num_base"]
FEATURE_NUM      = meta["feature_num"]
CAT_COLS         = meta["cat_cols"]                    # ['Source']
REG_NAME         = meta["model_name"]
IS_LINEAR_REG    = meta["is_linear"]
CLF_NAME         = meta["clf_model_name"]
CLF_CLASSES      = meta["clf_classes"]

CAT_FEAT_NAMES   = list(encoder.get_feature_names_out(CAT_COLS))
TFIDF_FEAT_NAMES = list(tfidf.get_feature_names_out())
ALL_FEAT_NAMES   = list(FEATURE_NUM) + CAT_FEAT_NAMES + TFIDF_FEAT_NAMES

app = FastAPI(title="Eco-Smart Classifier", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class PredictRequest(BaseModel):
    Source: str
    Poids: float = Field(ge=0)
    Volume: float = Field(ge=0)
    Conductivite: float
    Opacite: float
    Rigidite: float
    Rapport_Collecte: str = ""


class TopWord(BaseModel):
    word: str
    impact: float


class ClassProba(BaseModel):
    label: str
    proba: float


class PredictResponse(BaseModel):
    # Classification
    predicted_category: str
    class_probabilities: list[ClassProba]
    clf_model_name: str
    # Régression
    price: float
    contrib_num: float
    contrib_cat: float
    contrib_nlp: float
    top_words: list[TopWord]
    reg_model_name: str


def engineer_features(row: dict) -> dict:
    out = dict(row)
    out['Densite']           = row['Poids'] / (row['Volume'] + 1e-6)
    out['Cond_x_Rigidite']   = row['Conductivite'] * row['Rigidite']
    out['Opacite_x_Cond']    = row['Opacite'] * row['Conductivite']
    out['Poids_x_Rigidite']  = row['Poids'] * row['Rigidite']
    return out


def _df(X_row: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([X_row], columns=ALL_FEAT_NAMES)


def build_X(req: PredictRequest) -> np.ndarray:
    valid_source = list(encoder.categories_[CAT_COLS.index("Source")])
    if req.Source not in valid_source:
        raise HTTPException(400, f"Source inconnue. Attendu: {valid_source}")

    raw = {
        "Poids": req.Poids, "Volume": req.Volume,
        "Conductivite": req.Conductivite,
        "Opacite": req.Opacite, "Rigidite": req.Rigidite,
    }
    eng = engineer_features(raw)
    X_num = pd.DataFrame([[eng[c] for c in FEATURE_NUM]], columns=FEATURE_NUM)
    X_cat = pd.DataFrame([[req.Source]], columns=CAT_COLS)

    num_scaled  = scaler.transform(X_num)
    cat_encoded = encoder.transform(X_cat)
    text_vec    = tfidf.transform([req.Rapport_Collecte]).toarray()
    return np.concatenate([num_scaled, cat_encoded, text_vec], axis=1)[0]


def block_contributions(X_row: np.ndarray, n_num: int, n_cat: int):
    if IS_LINEAR_REG:
        coefs = reg_model.coef_
        intercept = float(reg_model.intercept_)
        c_num = intercept + float(np.dot(coefs[:n_num], X_row[:n_num]))
        c_cat = float(np.dot(coefs[n_num:n_num+n_cat], X_row[n_num:n_num+n_cat]))
        c_nlp = float(np.dot(coefs[n_num+n_cat:], X_row[n_num+n_cat:]))
        return c_num, c_cat, c_nlp

    zeros = np.zeros_like(X_row)
    baseline = float(reg_model.predict(_df(zeros))[0])
    x_only_num = zeros.copy(); x_only_num[:n_num] = X_row[:n_num]
    x_only_cat = zeros.copy(); x_only_cat[n_num:n_num+n_cat] = X_row[n_num:n_num+n_cat]
    x_only_nlp = zeros.copy(); x_only_nlp[n_num+n_cat:] = X_row[n_num+n_cat:]
    p_num = float(reg_model.predict(_df(x_only_num))[0])
    p_cat = float(reg_model.predict(_df(x_only_cat))[0])
    p_nlp = float(reg_model.predict(_df(x_only_nlp))[0])
    return p_num, p_cat - baseline, p_nlp - baseline


def top_words_for_row(X_row: np.ndarray, tfidf_offset: int, k: int = 6):
    words = TFIDF_FEAT_NAMES
    n_words = len(words)
    if IS_LINEAR_REG:
        coefs = reg_model.coef_
        impacts = [
            (str(words[i]), abs(float(coefs[tfidf_offset + i] * X_row[tfidf_offset + i])))
            for i in range(n_words)
        ]
        impacts = [t for t in impacts if t[1] > 0]
    else:
        non_zero = [i for i in range(n_words) if X_row[tfidf_offset + i] > 0]
        if not non_zero:
            return []
        full_pred = float(reg_model.predict(_df(X_row))[0])
        impacts = []
        for i in non_zero:
            x_ab = X_row.copy()
            x_ab[tfidf_offset + i] = 0.0
            p = float(reg_model.predict(_df(x_ab))[0])
            imp = abs(full_pred - p)
            if imp > 0:
                impacts.append((str(words[i]), imp))
    impacts.sort(key=lambda t: t[1], reverse=True)
    return [TopWord(word=w, impact=imp) for w, imp in impacts[:k]]


def classify(X_row: np.ndarray):
    X_df = _df(X_row)
    pred = clf_model.predict(X_df)[0]
    if hasattr(clf_model, "predict_proba"):
        probs = clf_model.predict_proba(X_df)[0]
        classes = list(clf_model.classes_)
        probas = sorted(
            [ClassProba(label=str(c), proba=float(p)) for c, p in zip(classes, probs)],
            key=lambda x: x.proba, reverse=True,
        )
    else:
        probas = [ClassProba(label=str(pred), proba=1.0)]
    return str(pred), probas


@app.get("/metadata")
def metadata():
    return {
        "num_features": FEATURE_NUM_BASE,
        "cat_features": CAT_COLS,
        "categories": {
            col: [str(x) for x in encoder.categories_[i]]
            for i, col in enumerate(CAT_COLS)
        },
        "target_classes": CLF_CLASSES,
        "n_tfidf_words": len(TFIDF_FEAT_NAMES),
        "reg_model": REG_NAME,
        "clf_model": CLF_NAME,
        "split_ratios": meta.get("split_ratios"),
        "knn_neighbors": meta.get("knn_neighbors"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X_row = build_X(req)
    n_num = len(FEATURE_NUM)
    n_cat = len(CAT_FEAT_NAMES)
    tfidf_offset = n_num + n_cat

    pred_cat, probas = classify(X_row)
    price = float(reg_model.predict(_df(X_row))[0])
    c_num, c_cat, c_nlp = block_contributions(X_row, n_num, n_cat)
    top_words = top_words_for_row(X_row, tfidf_offset)

    return PredictResponse(
        predicted_category=pred_cat,
        class_probabilities=probas,
        clf_model_name=CLF_NAME,
        price=price,
        contrib_num=c_num,
        contrib_cat=c_cat,
        contrib_nlp=c_nlp,
        top_words=top_words,
        reg_model_name=REG_NAME,
    )


@app.post("/classify")
def classify_only(req: PredictRequest):
    X_row = build_X(req)
    pred_cat, probas = classify(X_row)
    return {
        "predicted_category": pred_cat,
        "class_probabilities": [p.model_dump() for p in probas],
        "model": CLF_NAME,
    }


@app.get("/stats")
def stats():
    """Dashboard data — distribution des classes/sources + stats du dataset."""
    csv_path = Path("dataset_ProjetML_2026.csv")
    if not csv_path.exists():
        raise HTTPException(404, "dataset_ProjetML_2026.csv introuvable")
    df = pd.read_csv(csv_path)

    def _vc(col):
        s = df[col].dropna().value_counts()
        return [{"label": str(k), "count": int(v)} for k, v in s.items()]

    # Stats numériques
    num_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Prix_Revente']
    num_stats = {}
    for c in num_cols:
        s = df[c].dropna()
        num_stats[c] = {
            "count": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "median": float(s.median()),
            "max": float(s.max()),
            "nan_pct": float(df[c].isna().mean() * 100),
        }

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "categorie_distribution": _vc("Categorie"),
        "source_distribution": _vc("Source"),
        "numeric_stats": num_stats,
        "clustering_image": "/artifacts/clustering_pca.png",
    }


@app.get("/")
def root():
    return FileResponse("index.html")


# Sert les images de clustering générées par clustering.py
if Path("artifacts").exists():
    app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")
app.mount("/static", StaticFiles(directory="."), name="static")
