---
title: MakanSense Malaysia Restaurant Sentiment
emoji: 🍜
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MakanSense — Malaysia Restaurant & Food Review Sentiment

A sentiment-analysis web app for **TNL6323 Natural Language Processing**
(domain *d. Restaurant & Food Reviews, Malaysia context*). It classifies
Google Maps restaurant reviews as **positive / neutral / negative** and answers
business questions about the reviews — what customers like, what they complain
about, how they describe the ambience, what drives 5-star vs 1-star ratings, and
more.

## What it does

**Live classifier** — type any review and get the sentiment, confidence,
per-class probabilities, detected aspects (food / service / ambience / price)
and any emojis, using either:
- **Classical model** — TF-IDF (word + character n-grams) **+ VADER lexicon**
  features → Logistic Regression, trained on the project dataset.
- **Transformer model** — `cardiffnlp/twitter-roberta-base-sentiment-latest`
  (loaded on demand; an advanced feature).

**Insights dashboard** — pre-computed answers to every project question, drawn
from the whole dataset: overall sentiment split, star distribution, keyword
themes (via log-odds), aspect-based sentiment, and a model-comparison table.

## Advanced features (rubric)

1. **Transformer model** — RoBERTa sentiment pipeline, toggleable in the UI.
2. **Aspect-based sentiment** — every review is split into food / service /
   ambience / price using lexicons, each with its own sentiment breakdown.
3. **Emoji integration** — emojis are converted to words before classification
   (so 😍 contributes positive signal) and surfaced in the result.

## Run locally

```bash
pip install -r requirements.txt
python train.py            # trains the model -> models/sentiment_model.joblib
python build_analytics.py  # builds the dashboard data -> data/analytics.json
python app.py              # serves on http://localhost:7860
```

## Deploy to Hugging Face Spaces

1. Create a new Space → **SDK: Docker**.
2. Upload all files in this folder (keep the structure).
3. The Space builds from the `Dockerfile` and starts automatically.
   The classical model works immediately; the transformer downloads on first
   use of the **Transformer** toggle.

## Project structure

```
app.py              Flask web service (routes + prediction API)
train.py            trains & compares 3 classical models, saves the best
build_analytics.py  pre-computes the dashboard answers (analytics.json)
preprocess.py       text cleaning (metadata stripping, emoji handling)
features.py         VADER lexicon feature transformer
templates/index.html, static/style.css, static/app.js   front-end
models/             trained model, metrics.json, confusion_matrix.png
data/               raw + cleaned reviews, analytics.json
```

Built for TNL6323, March/April 2026.
