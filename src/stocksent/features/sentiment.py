"""Headline sentiment, without transformers.

- VADER: a general-purpose sentiment lexicon, kept as the baseline.
- Finance sentiment model: TF-IDF + logistic regression trained on ~12k financial news posts labeled
  bearish / bullish / neutral (zeroshot/twitter-financial-news-sentiment, MIT licence). It learns finance
  wording that VADER misses, such as "cut to Neutral", "PT raised" or "misses estimates".
"""
from __future__ import annotations

import io
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, make_pipeline

log = logging.getLogger(__name__)

SENTIMENT_DATA = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/{}"
LABELS = {0: "bearish", 1: "bullish", 2: "neutral"}
SENT_COLUMNS = ["fs_pos", "fs_neg", "fs_neu"]


def vader_scores(titles: Sequence[str]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return np.array([analyzer.polarity_scores(t)["compound"] for t in titles])


def normalise(text: str) -> str:
    """Make tweets look like headlines: drop links, cashtags and handles."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[$@]\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -:")


def _load_split(name: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / name
    if not path.exists():
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(SENTIMENT_DATA.format(name), timeout=60).content)
    return pd.read_csv(io.BytesIO(path.read_bytes()))


class FinanceSentimentModel:
    """Three-class finance sentiment classifier with word and character n-grams."""

    def __init__(self, pipeline=None, report: dict | None = None):
        self.pipeline = pipeline
        self.report = report or {}

    @classmethod
    def train(cls, cache_dir: Path, seed: int = 42) -> FinanceSentimentModel:
        train = _load_split("sent_train.csv", cache_dir)
        valid = _load_split("sent_valid.csv", cache_dir)
        x_tr, x_va = train["text"].map(normalise), valid["text"].map(normalise)
        features = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])
        pipe = make_pipeline(features, LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))
        search = GridSearchCV(pipe, {"logisticregression__C": [1, 3, 10, 30]}, scoring="f1_macro", cv=5, n_jobs=1)
        search.fit(x_tr, train["label"])
        pred = search.predict(x_va)
        report = {
            "valid_accuracy": accuracy_score(valid["label"], pred),
            "valid_macro_f1": f1_score(valid["label"], pred, average="macro"),
            "best_C": search.best_params_["logisticregression__C"],
            "n_train": len(train),
            "n_valid": len(valid),
        }
        log.info("finance sentiment model: valid accuracy %.3f, macro F1 %.3f", report["valid_accuracy"],
                 report["valid_macro_f1"])
        return cls(search.best_estimator_, report)

    def score(self, titles: Sequence[str]) -> pd.DataFrame:
        """P(bullish), P(bearish), P(neutral) for each headline."""
        proba = self.pipeline.predict_proba([normalise(t) for t in titles])
        order = [list(self.pipeline.classes_).index(c) for c in (1, 0, 2)]
        return pd.DataFrame(proba[:, order], columns=SENT_COLUMNS)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "report": self.report}, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> FinanceSentimentModel:
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["report"])
