import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from features import VaderFeatures
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from preprocess import clean_text, pick_text_series, normalize_sentiment, find_sentiment_column

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(HERE, "data", "raw_reviews.csv")
MODEL_DIR = os.path.join(HERE, "models")
DATA_DIR = os.path.join(HERE, "data")
LABELS = ["negative", "neutral", "positive"]
SEED = 42


def load_data(csv_path=RAW_CSV):
    """Load a dataset with flexible column names.

    Needs a review-text column (Review_clean / Review_en / Review / text / ...)
    and a sentiment column (sentiment / label / ...). Extra columns are ignored.
    """
    df = pd.read_csv(csv_path)
    text = pick_text_series(df)
    sent_col = find_sentiment_column(df)
    if text is None:
        raise ValueError("No review-text column found "
                         "(expected one of: Review, Review_en, text, comment...).")
    if sent_col is None:
        raise ValueError("No sentiment column found "
                         "(expected 'sentiment' or 'label' with positive/neutral/negative).")
    df = df.copy()
    df["text_clean"] = text.apply(clean_text)
    df["sentiment"] = df[sent_col].apply(normalize_sentiment)
    df = df[(df["text_clean"].str.len() > 0) & df["sentiment"].isin(LABELS)].copy()
    if len(df) < 15:
        raise ValueError(f"Only {len(df)} usable labelled rows after cleaning — "
                         "need at least 15 (and ideally some of each class).")
    return df


def build_models():
    """Pipelines sharing a word+char TF-IDF front-end.

    Logistic Regression and Linear SVM additionally use VADER lexicon scores
    (hybrid lexicon + ML) which supply sentiment signal for words absent from
    the small training vocabulary. Complement Naive Bayes is kept as a pure
    bag-of-words baseline (it requires non-negative features, so the signed
    VADER scores are not added to it).
    """
    def tfidf_union():
        return [
            ("word", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.9,
                sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                sublinear_tf=True)),
        ]

    def hybrid_feats():
        return FeatureUnion(tfidf_union() + [
            ("vader", Pipeline([("v", VaderFeatures()),
                                ("scale", StandardScaler())])),
        ])

    return {
        "logreg": Pipeline([
            ("feats", hybrid_feats()),
            ("clf", LogisticRegression(
                max_iter=3000, class_weight="balanced", C=2.0, random_state=SEED)),
        ]),
        "linearsvc": Pipeline([
            ("feats", hybrid_feats()),
            ("clf", LinearSVC(class_weight="balanced", C=1.0, random_state=SEED)),
        ]),
        "complementnb": Pipeline([
            ("feats", FeatureUnion(tfidf_union())),
            ("clf", ComplementNB()),
        ]),
    }


def run(csv_path=RAW_CSV, model_dir=MODEL_DIR, data_dir=DATA_DIR, verbose=True):
    """Train, compare models, save artifacts. Returns the metrics dict."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    df = load_data(csv_path)
    if verbose:
        print(f"Loaded {len(df)} usable reviews")
        print("Label distribution:\n", df["sentiment"].value_counts().to_string())

    X, y = df["text_clean"].values, df["sentiment"].values
    min_class = df["sentiment"].value_counts().min()

    # small datasets: shrink CV folds / handle tiny classes gracefully
    n_splits = max(2, min(5, int(min_class)))
    stratify = y if min_class >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=SEED)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results, fitted = {}, {}

    for name, pipe in build_models().items():
        try:
            cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                    scoring="f1_macro").mean()
        except Exception:
            cv_f1 = float("nan")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "cv_macro_f1": round(float(cv_f1), 4) if cv_f1 == cv_f1 else None,
            "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "test_macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
            "test_weighted_f1": round(float(f1_score(y_te, pred, average="weighted")), 4),
            "per_class": classification_report(
                y_te, pred, labels=LABELS, output_dict=True, zero_division=0),
        }
        fitted[name] = pipe
        if verbose:
            print(f"  {name:<13} acc={results[name]['test_accuracy']:.3f}  "
                  f"macroF1={results[name]['test_macro_f1']:.3f}")

    def score(k):
        v = results[k]["cv_macro_f1"]
        return v if v is not None else results[k]["test_macro_f1"]
    best = max(results, key=score)

    # refit best on ALL data so the deployed model uses every labelled example
    final = build_models()[best].fit(X, y)
    joblib.dump(final, os.path.join(model_dir, "sentiment_model.joblib"))

    # confusion matrix figure (held-out test set)
    cm = confusion_matrix(y_te, fitted[best].predict(X_te), labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    meta = {
        "best_model": best,
        "labels": LABELS,
        "n_train": len(X_tr), "n_test": len(X_te), "n_total": len(X),
        "train_label_counts": pd.Series(y_tr).value_counts().to_dict(),
        "results": results,
    }
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # knowledge representation: cleaned dataset (only columns that exist)
    keep = [c for c in ["Name", "Rating", "Review_en", "text_clean",
                        "food_rating", "service_rating", "atmosphere_rating",
                        "sentiment"] if c in df.columns]
    df[keep].to_csv(os.path.join(data_dir, "reviews_clean.csv"), index=False)

    if verbose:
        print(f"Best model (by CV macro-F1): {best}. Saved model + metrics.")
    return meta


def main():
    run()


if __name__ == "__main__":
    main()
