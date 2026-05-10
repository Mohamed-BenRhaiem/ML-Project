"""
benchmark_vectorizers.py
------------------------
Module 4 du cahier des charges : comparaison de 4 approches de vectorisation
pour la classification de Categorie depuis le rapport textuel uniquement.

  - Bag of Words (CountVectorizer baseline)
  - TF-IDF (uni+bi, stopwords FR + stemmer Snowball — référence)
  - Word2Vec (avg embedding, dim=100)
  - FastText (avg embedding, robuste aux fautes / OOV)

Toutes les approches sont évaluées avec un même classifieur (LogisticRegression)
pour comparer le pouvoir prédictif des représentations textuelles seules.

Sortie : artifacts/vectorizer_benchmark.txt + ligne de log MLflow par approche.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import mlflow

from gensim.models import Word2Vec, FastText

from nlp_utils import french_tokenizer

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ─── MLflow ──────────────────────────────────────────────────────
USE_MLFLOW = os.environ.get("DISABLE_MLFLOW", "0") != "1"
if USE_MLFLOW:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("eco-smart-classifier")


# ─── Chargement et split (texte UNIQUEMENT, classification) ──────
df = pd.read_csv("dataset_ProjetML_2026.csv")
df = df.dropna(subset=["Categorie"]).reset_index(drop=True)
df["Rapport_Collecte"] = df["Rapport_Collecte"].fillna("")
print(f"Échantillons : {len(df)}")

df_train, df_test = train_test_split(
    df, test_size=0.2, stratify=df["Categorie"], random_state=42
)
y_train = df_train["Categorie"].values
y_test  = df_test["Categorie"].values

# Pré-tokenisation pour Word2Vec / FastText
train_tokens = [french_tokenizer(t) for t in df_train["Rapport_Collecte"]]
test_tokens  = [french_tokenizer(t) for t in df_test["Rapport_Collecte"]]


def evaluate(name, X_train, X_test, dim_used):
    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="macro")
    elapsed = time.time() - t0

    print(f"  {name:12s} | acc = {acc:.4f} | F1-macro = {f1:.4f} | "
          f"dim = {dim_used} | clf time = {elapsed:.1f}s")

    if USE_MLFLOW:
        with mlflow.start_run(run_name=f"vectorizer_{name}"):
            mlflow.set_tag("task", "vectorizer_benchmark")
            mlflow.set_tag("vectorizer", name)
            mlflow.log_param("dim_used", dim_used)
            mlflow.log_param("classifier", "LogisticRegression")
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_f1_macro", f1)
            mlflow.log_metric("clf_train_time_seconds", elapsed)

    return {"accuracy": acc, "f1_macro": f1, "dim": dim_used, "clf_time_s": elapsed}


print("\n─── 1) Bag of Words ───")
bow = CountVectorizer(
    tokenizer=french_tokenizer, token_pattern=None,
    max_features=300, lowercase=False,
)
X_train_bow = bow.fit_transform(df_train["Rapport_Collecte"])
X_test_bow  = bow.transform(df_test["Rapport_Collecte"])
res_bow = evaluate("BoW", X_train_bow, X_test_bow, X_train_bow.shape[1])


print("\n─── 2) TF-IDF (uni+bi, stopwords FR, stemmer Snowball) ───")
tfidf = TfidfVectorizer(
    tokenizer=french_tokenizer, token_pattern=None,
    max_features=300, ngram_range=(1, 2), min_df=2,
    sublinear_tf=True, lowercase=False,
)
X_train_tf = tfidf.fit_transform(df_train["Rapport_Collecte"])
X_test_tf  = tfidf.transform(df_test["Rapport_Collecte"])
res_tfidf = evaluate("TF-IDF", X_train_tf, X_test_tf, X_train_tf.shape[1])


def avg_embedding(tokens, model, dim):
    if not tokens:
        return np.zeros(dim)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


print("\n─── 3) Word2Vec (avg embedding, dim=100) ───")
W2V_DIM = 100
t0 = time.time()
w2v = Word2Vec(
    sentences=train_tokens,
    vector_size=W2V_DIM, window=5, min_count=2,
    workers=4, epochs=15, seed=42,
)
print(f"  Word2Vec entraîné en {time.time()-t0:.1f}s | vocab = {len(w2v.wv)}")
X_train_w2v = np.array([avg_embedding(t, w2v, W2V_DIM) for t in train_tokens])
X_test_w2v  = np.array([avg_embedding(t, w2v, W2V_DIM) for t in test_tokens])
res_w2v = evaluate("Word2Vec", X_train_w2v, X_test_w2v, W2V_DIM)


print("\n─── 4) FastText (avg embedding, robuste OOV) ───")
FT_DIM = 100
t0 = time.time()
ft = FastText(
    sentences=train_tokens,
    vector_size=FT_DIM, window=5, min_count=2,
    workers=4, epochs=15, seed=42,
)
print(f"  FastText entraîné en {time.time()-t0:.1f}s | vocab = {len(ft.wv)}")
X_train_ft = np.array([avg_embedding(t, ft, FT_DIM) for t in train_tokens])
X_test_ft  = np.array([avg_embedding(t, ft, FT_DIM) for t in test_tokens])
res_ft = evaluate("FastText", X_train_ft, X_test_ft, FT_DIM)


# ─── Résumé ──────────────────────────────────────────────────────
print("\n═" * 30)
print("RÉSUMÉ — classification Categorie depuis texte seul")
print("═" * 30)
results = {"BoW": res_bow, "TF-IDF": res_tfidf, "Word2Vec": res_w2v, "FastText": res_ft}

ranking = sorted(results.items(), key=lambda kv: -kv[1]["f1_macro"])
print(f"{'Vectoriseur':12s} | {'Acc':>7s} | {'F1-macro':>9s} | {'Dim':>5s} | {'Time':>6s}")
print("-" * 55)
for name, r in ranking:
    print(f"{name:12s} | {r['accuracy']:7.4f} | {r['f1_macro']:9.4f} | "
          f"{r['dim']:5d} | {r['clf_time_s']:5.1f}s")
print(f"\n→ Meilleur vectoriseur : {ranking[0][0]} (F1 = {ranking[0][1]['f1_macro']:.4f})")


# ─── Export ──────────────────────────────────────────────────────
out_path = ARTIFACTS_DIR / "vectorizer_benchmark.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("Module 4 — Comparaison des vectoriseurs NLP\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Tâche : classification de Categorie\n")
    f.write(f"Classifieur : LogisticRegression (mêmes hyperparams partout)\n")
    f.write(f"Train : {len(y_train)} | Test : {len(y_test)}\n\n")
    f.write(f"{'Vectoriseur':12s} | {'Acc':>7s} | {'F1-macro':>9s} | {'Dim':>5s} | {'Time':>6s}\n")
    f.write("-" * 55 + "\n")
    for name, r in ranking:
        f.write(f"{name:12s} | {r['accuracy']:7.4f} | {r['f1_macro']:9.4f} | "
                f"{r['dim']:5d} | {r['clf_time_s']:5.1f}s\n")
print(f"\n✓ Sauvegardé : {out_path}")
