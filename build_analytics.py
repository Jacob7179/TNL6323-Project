import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocess import (clean_text, extract_emojis, pick_text_series,
                        normalize_sentiment, find_sentiment_column)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_reviews.csv")

# --- aspect lexicons (Malaysia restaurant context) ------------------------
ASPECTS = {
    "food": [
        "food", "dish", "dishes", "taste", "tasty", "delicious", "yummy",
        "flavour", "flavor", "flavourful", "meal", "menu", "portion", "fresh",
        "cook", "cooked", "spicy", "sweet", "sour", "rice", "noodle", "noodles",
        "chicken", "fish", "soup", "coffee", "drink", "drinks", "dessert",
        "bread", "cheese", "sauce", "fried", "grill", "grilled", "serving",
        "ingredient", "quality", "hot", "cold", "bland", "oily", "salty",
    ],
    "service": [
        "service", "staff", "staffs", "waiter", "waitress", "server", "crew",
        "friendly", "attentive", "helpful", "rude", "slow", "wait", "waiting",
        "attitude", "manager", "order", "ordered", "polite", "smile", "smiling",
        "ignore", "ignored", "unfriendly", "efficient", "prompt", "professional",
        "customer", "welcoming", "responsive",
    ],
    "ambiance": [
        "ambience", "ambiance", "atmosphere", "environment", "vibe", "vibes",
        "cozy", "cosy", "comfortable", "comfy", "decor", "decoration", "music",
        "clean", "cleanliness", "dirty", "seating", "seat", "crowded", "noisy",
        "quiet", "spacious", "aircon", "air-cond", "lighting", "interior",
        "relaxing", "warm", "cool", "modern", "aesthetic", "instagrammable",
    ],
}

# price is a useful 4th aspect in Malaysia reviews
ASPECTS["price"] = [
    "price", "prices", "pricey", "expensive", "cheap", "affordable", "value",
    "worth", "reasonable", "overpriced", "cost", "rm", "ringgit", "budget",
]

STOP = set("""
a an the and or but if then else for to of in on at by with from into about as
is are was were be been being do does did doing have has had having i you he she
it we they me him her us them my your his its our their this that these those
not no so very just too also only out up down off over again more most some any
all can will would could should may might must here there what which who whom
when where why how than then once their our we you i me my place here go went
will been im its u ur also got get really quite will visit came come back again
""".split())


def load(csv_path=RAW_CSV):
    df = pd.read_csv(csv_path)
    base = pick_text_series(df)
    if base is None:
        raise ValueError("No review-text column found.")
    sent_col = find_sentiment_column(df)
    if sent_col is None:
        raise ValueError("No sentiment column found.")
    df = df.copy()
    df["text_clean"] = base.apply(clean_text)
    df["text_raw"] = base.fillna("")
    df["review_display"] = base.fillna("")
    df["sentiment"] = df[sent_col].apply(normalize_sentiment)
    df = df[(df["text_clean"].str.len() > 0) & df["sentiment"].notna()].copy()
    # star rating: prefer an overall "Rating" column ("Rated 5.0 out of 5");
    # otherwise derive it from the per-aspect rating columns (mean of whichever
    # of food/service/atmosphere ratings the review has).
    if "Rating" in df.columns:
        df["stars"] = df["Rating"].astype(str).str.extract(
            r"([0-5](?:\.\d)?)").astype(float)
    else:
        aspect_cols = [c for c in ["food_rating", "service_rating",
                                   "atmosphere_rating"] if c in df.columns]
        if aspect_cols:
            df["stars"] = df[aspect_cols].mean(axis=1, skipna=True)
        else:
            df["stars"] = np.nan
    return df


def tokens(text):
    return [t for t in re.findall(r"[a-z]+", str(text).lower())
            if len(t) > 2 and t not in STOP]


def log_odds_keywords(texts_a, texts_b, top_n=12):
    """Distinctive words for group A vs group B via log-odds w/ Dirichlet prior
    (Monroe et al. 2008). Returns [(word, zscore), ...]."""
    ca, cb = Counter(), Counter()
    for t in texts_a:
        ca.update(tokens(t))
    for t in texts_b:
        cb.update(tokens(t))
    vocab = set(ca) | set(cb)
    a_tot, b_tot = sum(ca.values()), sum(cb.values())
    a0 = a_tot + len(vocab)
    b0 = b_tot + len(vocab)
    scores = {}
    for w in vocab:
        ya, yb = ca[w], cb[w]
        if ya + yb < 2:          # ignore ultra-rare words
            continue
        la = np.log((ya + 1) / (a0 - ya - 1))
        lb = np.log((yb + 1) / (b0 - yb - 1))
        delta = la - lb
        var = 1.0 / (ya + 1) + 1.0 / (yb + 1)
        scores[w] = delta / np.sqrt(var)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(w, round(float(s), 2)) for w, s in ranked[:top_n]]


def freq_keywords(texts, top_n=12):
    c = Counter()
    for t in texts:
        c.update(set(tokens(t)))   # set -> count reviews mentioning the word
    return [(w, n) for w, n in c.most_common(top_n)]


def aspect_breakdown(df):
    """Per-aspect: mention count, sentiment split, avg star, example snippets."""
    out = {}
    for aspect, lex in ASPECTS.items():
        lexset = set(lex)
        mask = df["text_clean"].apply(lambda t: bool(lexset & set(tokens(t))))
        sub = df[mask]
        if len(sub) == 0:
            continue
        sent = sub["sentiment"].value_counts()
        pos = int(sent.get("positive", 0))
        neu = int(sent.get("neutral", 0))
        neg = int(sent.get("negative", 0))
        tot = pos + neu + neg
        # example snippets: one positive, one negative
        ex_pos = sub[sub["sentiment"] == "positive"]["review_display"].head(1).tolist()
        ex_neg = sub[sub["sentiment"] == "negative"]["review_display"].head(1).tolist()
        out[aspect] = {
            "mentions": int(len(sub)),
            "positive": pos, "neutral": neu, "negative": neg,
            "pos_pct": round(100 * pos / tot, 1) if tot else 0,
            "neg_pct": round(100 * neg / tot, 1) if tot else 0,
            "avg_stars": round(float(sub["stars"].mean()), 2) if sub["stars"].notna().any() else None,
            "example_positive": _short(ex_pos[0]) if ex_pos else None,
            "example_negative": _short(ex_neg[0]) if ex_neg else None,
            "top_words": [w for w, _ in freq_keywords(sub["text_clean"], 8)],
        }
    return out


def _short(text, n=180):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:n] + ("…" if len(text) > n else "")


ASPECT_ADVICE = {
    "food": "Check the dishes named in complaints for consistency, freshness, seasoning and portion size.",
    "service": "Speed up service and coach staff on friendliness and attentiveness, especially at peak hours.",
    "ambiance": "Improve cleanliness, seating comfort, noise or lighting to make the space more inviting.",
    "price": "Revisit pricing or raise perceived value — clearer portions, set menus, or promotions.",
}


def build_owner_summary(overall, aspects, n):
    """Plain-language, boss-facing 'where to improve' summary."""
    pos, neg = overall["pos_pct"], overall["neg_pct"]
    # only judge aspects with enough evidence
    thr = max(5, round(0.04 * n))
    rated = {a: d for a, d in aspects.items() if d["mentions"] >= thr}

    doing_well, improve = [], []
    if rated:
        for a, d in sorted(rated.items(), key=lambda kv: kv[1]["pos_pct"], reverse=True):
            if d["pos_pct"] >= 65 and len(doing_well) < 2:
                doing_well.append({"aspect": a, "pos_pct": d["pos_pct"]})
        for a, d in sorted(rated.items(), key=lambda kv: kv[1]["neg_pct"], reverse=True):
            if d["neg_pct"] >= 12 and len(improve) < 3:
                improve.append({
                    "aspect": a, "neg_pct": d["neg_pct"],
                    "advice": ASPECT_ADVICE.get(a, "Review this area with your team."),
                    "keywords": d.get("top_words", [])[:5],
                })

    if neg <= 10:
        tone = "Your shop is in great shape — most customers leave happy."
    elif neg <= 25:
        tone = "Your shop is doing well overall, with a few clear areas to tighten up."
    else:
        tone = "There's real room to improve — a notable share of customers left unhappy."

    priority = None
    if improve:
        top = improve[0]
        priority = (f"Start with {top['aspect']}: it draws the highest share of "
                    f"negative mentions ({top['neg_pct']}%).")

    headline = (f"{round(pos)}% of your customers are happy"
                + (f", but {round(neg)}% left unhappy." if neg else "."))

    return {
        "headline": headline,
        "tone": tone,
        "doing_well": doing_well,
        "improve": improve,
        "priority": priority,
    }


def run(csv_path=RAW_CSV, data_dir=DATA_DIR, verbose=True):
    df = load(csv_path)
    n = len(df)
    if n == 0:
        raise ValueError("No usable labelled rows for analytics.")
    sent = df["sentiment"].value_counts()
    pos, neu, neg = (int(sent.get(k, 0)) for k in ["positive", "neutral", "negative"])

    pos_txt = df[df["sentiment"] == "positive"]["text_clean"]
    neg_txt = df[df["sentiment"] == "negative"]["text_clean"]
    other_txt = df[df["sentiment"] != "positive"]["text_clean"]
    nonneg_txt = df[df["sentiment"] != "negative"]["text_clean"]

    aspects = aspect_breakdown(df)
    has_stars = df["stars"].notna().any()
    avg_stars = round(float(df["stars"].mean()), 2) if has_stars else None

    # star-rating groups (empty if no ratings supplied)
    five = df[df["stars"] == 5.0]["text_clean"]
    one = df[df["stars"] <= 2.0]["text_clean"]
    rest_hi = df[df["stars"] < 5.0]["text_clean"]
    rest_lo = df[df["stars"] > 2.0]["text_clean"]

    aspect_ratings = {}
    for col, key in [("food_rating", "food"), ("service_rating", "service"),
                     ("atmosphere_rating", "atmosphere")]:
        vals = df[col].dropna() if col in df.columns else pd.Series([], dtype=float)
        aspect_ratings[key] = {
            "avg": round(float(vals.mean()), 2) if len(vals) else None,
            "n": int(len(vals)),
        }

    emoji_counter = Counter()
    for t in df["text_raw"]:
        emoji_counter.update(extract_emojis(t))
    top_emojis = [{"emoji": e, "count": int(c)} for e, c in emoji_counter.most_common(10)]

    summary_line = (f"{round(100*pos/n)}% positive, {round(100*neu/n)}% neutral, "
                    f"{round(100*neg/n)}% negative across {n} reviews"
                    + (f" (avg {avg_stars}\u2605)." if avg_stars is not None else "."))

    overall = {
        "positive": pos, "neutral": neu, "negative": neg,
        "pos_pct": round(100 * pos / n, 1),
        "neu_pct": round(100 * neu / n, 1),
        "neg_pct": round(100 * neg / n, 1),
        "avg_stars": avg_stars,
    }

    rounded_stars = df["stars"].dropna().round().astype(int)
    analytics = {
        "n_reviews": n,
        "overall": overall,
        "star_distribution": {
            str(int(s)): int((rounded_stars == s).sum())
            for s in sorted(rounded_stars.unique())
        },
        "aspects": aspects,
        "aspect_ratings": aspect_ratings,
        "top_emojis": top_emojis,
        "owner_summary": build_owner_summary(overall, aspects, n),
        "questions": {
            "like_most": {
                "title": "What do customers like most?",
                "keywords": log_odds_keywords(pos_txt, other_txt, 12),
                "summary": _summary_like(aspects),
            },
            "complaints": {
                "title": "What are the most common complaints?",
                "keywords": log_odds_keywords(neg_txt, nonneg_txt, 12),
                "summary": _summary_complaints(aspects),
            },
            "negative_themes": {
                "title": "What themes appear in negative reviews?",
                "keywords": freq_keywords(neg_txt, 12),
                "examples": [_short(x) for x in
                             df[df["sentiment"] == "negative"]["review_display"].head(3)],
            },
            "ambience": {
                "title": "How do customers describe the ambience?",
                "aspect": aspects.get("ambiance", {}),
            },
            "overall_summary": {
                "title": "Overall customer sentiment",
                "line": summary_line,
            },
        },
    }

    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "analytics.json"), "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Wrote analytics.json  ({n} reviews; {pos} pos / {neu} neu / {neg} neg)")
    return analytics


def main():
    run()


def _summary_like(aspects):
    ranked = sorted(aspects.items(), key=lambda kv: kv[1]["pos_pct"], reverse=True)
    top = [a for a, d in ranked if d["mentions"] >= 10][:2]
    return ("Most appreciated aspects: " + ", ".join(top) + ".") if top else ""


def _summary_complaints(aspects):
    ranked = sorted(aspects.items(), key=lambda kv: kv[1]["neg_pct"], reverse=True)
    top = [a for a, d in ranked if d["mentions"] >= 10][:2]
    return ("Most complained-about aspects: " + ", ".join(top) + ".") if top else ""


if __name__ == "__main__":
    main()
