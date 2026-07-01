"""
app.py  --  MakanSense Flask web service
TNL6323 Natural Language Processing  |  Restaurant & Food Reviews (Malaysia)

Two-model sentiment system served as a single-page web app:

  • Classical model  : TF-IDF (word+char) + Logistic Regression, trained on the
                       project dataset. Always available, no downloads.
  • Transformer model: cardiffnlp/twitter-roberta-base-sentiment-latest
                       (3-class). Loaded lazily; activates on Hugging Face
                       Spaces where the model hub is reachable. Falls back to
                       the classical model if transformers is unavailable.

Routes
  GET  /              dashboard (answers every brief question from analytics.json)
  POST /api/predict   classify a single review (model selectable)
  GET  /api/health    model availability
"""

import json
import os
import re
import warnings

# A model trained on a slightly different scikit-learn version still loads fine
# for this simple pipeline; silence the cosmetic version-mismatch warning so the
# console stays clean (e.g. model trained in Colab, app run locally).
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.simplefilter("ignore", InconsistentVersionWarning)
except Exception:
    pass

import joblib
from flask import Flask, jsonify, render_template, request

import features  # noqa: F401  (registers VaderFeatures for joblib unpickling)
from preprocess import clean_text, extract_emojis

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:          # running inside a notebook cell (no __file__)
    HERE = os.getcwd()
MODEL_PATH = os.path.join(HERE, "models", "sentiment_model.joblib")
METRICS_PATH = os.path.join(HERE, "models", "metrics.json")
ANALYTICS_PATH = os.path.join(HERE, "data", "analytics.json")

LABELS = ["negative", "neutral", "positive"]

# aspect lexicons (kept in sync with build_analytics.py)
ASPECTS = {
    "food": {"food", "dish", "dishes", "taste", "tasty", "delicious", "yummy",
             "flavour", "flavor", "meal", "menu", "portion", "fresh", "cook",
             "cooked", "spicy", "sweet", "rice", "noodle", "chicken", "soup",
             "coffee", "drink", "dessert", "sauce", "fried", "quality", "bland",
             "oily", "salty"},
    "service": {"service", "staff", "staffs", "waiter", "waitress", "server",
                "friendly", "attentive", "helpful", "rude", "slow", "wait",
                "waiting", "attitude", "manager", "polite", "smile", "ignore",
                "unfriendly", "efficient", "prompt", "professional"},
    "ambiance": {"ambience", "ambiance", "atmosphere", "environment", "vibe",
                 "cozy", "cosy", "comfortable", "decor", "music", "clean",
                 "dirty", "seating", "crowded", "noisy", "quiet", "spacious",
                 "aircon", "lighting", "interior", "relaxing", "modern"},
    "price": {"price", "pricey", "expensive", "cheap", "affordable", "value",
              "worth", "reasonable", "overpriced", "cost", "budget"},
}

app = Flask(__name__)

# ------------------------------------------------------------------ classical
_clf = joblib.load(MODEL_PATH)
with open(METRICS_PATH, encoding="utf-8") as f:
    METRICS = json.load(f)
with open(ANALYTICS_PATH, encoding="utf-8") as f:
    ANALYTICS = json.load(f)


def classical_predict(text):
    clean = clean_text(text)
    pred = _clf.predict([clean])[0]
    probs = None
    if hasattr(_clf, "predict_proba"):
        p = _clf.predict_proba([clean])[0]
        classes = list(_clf.named_steps["clf"].classes_)
        probs = {c: round(float(v), 4) for c, v in zip(classes, p)}
        conf = round(float(max(p)), 4)
    else:  # LinearSVC fallback via decision margin -> softmax
        import numpy as np
        d = _clf.decision_function([clean])[0]
        classes = list(_clf.named_steps["clf"].classes_)
        e = np.exp(d - d.max())
        sm = e / e.sum()
        probs = {c: round(float(v), 4) for c, v in zip(classes, sm)}
        conf = round(float(sm.max()), 4)
    return pred, conf, probs


# ----------------------------------------------------------------- transformer
_transformer = None
_transformer_state = "not_loaded"   # not_loaded | ready | unavailable
_TF_MAP = {"negative": "negative", "neutral": "neutral", "positive": "positive",
           "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


def get_transformer():
    """Lazy-load the HF sentiment pipeline. Safe to call repeatedly."""
    global _transformer, _transformer_state
    if _transformer_state == "ready":
        return _transformer
    if _transformer_state == "unavailable":
        return None
    try:
        from transformers import pipeline
        _transformer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )
        _transformer_state = "ready"
        return _transformer
    except Exception as exc:        # no internet / package / RAM
        app.logger.warning("Transformer unavailable: %s", exc)
        _transformer_state = "unavailable"
        return None


def transformer_predict(text):
    pipe = get_transformer()
    if pipe is None:
        return None
    out = pipe(text[:512])[0]       # list of {label, score}
    probs = {}
    for d in out:
        probs[_TF_MAP.get(d["label"], d["label"]).lower()] = round(float(d["score"]), 4)
    for lab in LABELS:
        probs.setdefault(lab, 0.0)
    pred = max(probs, key=probs.get)
    return pred, round(float(probs[pred]), 4), probs


# --------------------------------------------------------------------- aspects
# split a review into clauses so each aspect can be judged on its own words:
# "the dog is ugly and food is delicious" -> ["the dog is ugly", "food is delicious"]
_CLAUSE_SPLIT = re.compile(
    r"\b(?:but|however|although|though|whereas|while|yet|and)\b|[.,;!?]",
    re.IGNORECASE,
)


def analyze_aspects(text):
    """Aspect-based sentiment + the aspect-relevant portion of the review.

    Returns (aspects, relevant_text):
      aspects       {aspect: {sentiment, confidence}} for each aspect mentioned
      relevant_text the clauses that actually mention an aspect, joined — used
                    to judge the overall sentiment on restaurant-related content
                    only (ignoring off-topic text like "the dog is ugly").
    """
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(text) if c.strip()]
    if not clauses:
        clauses = [text]

    aspects, relevant, seen = {}, [], set()
    for aspect, lex in ASPECTS.items():
        rel = [c for c in clauses
               if lex & set(re.findall(r"[a-z]+", c.lower()))]
        if not rel:
            continue
        sent, conf, _ = classical_predict(" ".join(rel))
        aspects[aspect] = {"sentiment": sent, "confidence": conf}
        for c in rel:
            if c not in seen:
                seen.add(c); relevant.append(c)
    return aspects, " ".join(relevant)


# ----------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template(
        "index.html",
        analytics_json=json.dumps(ANALYTICS, ensure_ascii=False),
        metrics_json=json.dumps(METRICS),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    model = (data.get("model") or "classical").lower()
    if not text:
        return jsonify({"error": "Please enter a review."}), 400

    aspects, relevant_text = analyze_aspects(text)

    # No food / service / ambience mentioned -> nothing restaurant-related to
    # judge, so the sentiment is undefined rather than a misleading verdict.
    if not aspects:
        return jsonify({
            "model": model,
            "sentiment": "undefined",
            "confidence": None,
            "probabilities": None,
            "emojis": extract_emojis(text),
            "aspects": {},
        })

    # Judge the overall sentiment on the aspect-relevant clauses only, so
    # off-topic negativity (e.g. "the dog is ugly") doesn't sink a review that
    # is actually praising the food.
    used = model
    if model == "transformer":
        res = transformer_predict(relevant_text)
        if res is None:            # graceful fallback
            res = classical_predict(relevant_text)
            used = "classical (transformer unavailable)"
    else:
        res = classical_predict(relevant_text)

    pred, conf, probs = res
    return jsonify({
        "model": used,
        "sentiment": pred,
        "confidence": conf,
        "probabilities": probs,
        "emojis": extract_emojis(text),
        "aspects": aspects,
    })


@app.route("/api/health")
def health():
    return jsonify({
        "classical": "ready",
        "transformer": _transformer_state,
        "best_model": METRICS.get("best_model"),
        "n_reviews": ANALYTICS.get("n_reviews"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
