import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderFeatures(BaseEstimator, TransformerMixin):
    """Turn a list of texts into [neg, neu, pos, compound] VADER scores."""

    def __init__(self):
        self._analyzer = None

    @property
    def analyzer(self):
        # lazy so the object pickles cleanly
        if self._analyzer is None:
            self._analyzer = SentimentIntensityAnalyzer()
        return self._analyzer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            s = self.analyzer.polarity_scores(t if isinstance(t, str) else "")
            rows.append([s["neg"], s["neu"], s["pos"], s["compound"]])
        return np.asarray(rows, dtype=float)

    def __getstate__(self):
        # don't pickle the analyzer instance
        return {}

    def __setstate__(self, state):
        self._analyzer = None
