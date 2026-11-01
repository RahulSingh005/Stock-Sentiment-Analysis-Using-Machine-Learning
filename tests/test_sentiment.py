from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from stocksent.features.sentiment import SENT_COLUMNS, FinanceSentimentModel, normalise


def test_normalise_strips_links_cashtags_and_handles():
    assert normalise("$AAPL - Apple beats estimates https://t.co/abc @trader") == "Apple beats estimates"


def test_scores_come_back_in_bullish_bearish_neutral_order():
    texts = ["beats estimates strong growth", "misses estimates weak outlook", "company holds annual meeting"] * 5
    labels = [1, 0, 2] * 5  # dataset labels: 0 bearish, 1 bullish, 2 neutral
    pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(C=10)).fit(texts, labels)
    scores = FinanceSentimentModel(pipe).score(["beats estimates", "misses estimates"])
    assert list(scores.columns) == SENT_COLUMNS
    assert scores.loc[0, "fs_pos"] > scores.loc[0, "fs_neg"]
    assert scores.loc[1, "fs_neg"] > scores.loc[1, "fs_pos"]
