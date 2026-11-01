import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from stocksent.api import app as api
from stocksent.features.dataset import FEATURE_SETS
from stocksent.inference import Predictor

from .conftest import StubSentiment, make_prices


@pytest.fixture
def client(monkeypatch):
    features = FEATURE_SETS["price+news"]
    rng = np.random.default_rng(0)
    estimator = LogisticRegression().fit(pd.DataFrame(rng.normal(size=(200, len(features))), columns=features),
                                         rng.integers(0, 2, 200))
    bundle = {"estimator": estimator, "model": "Test model", "features": features, "trained_through": "2022-12-31",
              "benchmark": "SPY", "volatility_index": "^VIX"}
    predictor = Predictor(bundle, StubSentiment())
    monkeypatch.setattr(api, "get_predictor", lambda: predictor)
    return TestClient(api.app)


def _payload(n_days=60):
    prices = make_prices(n=n_days)
    bars = {sym: [{"date": str(r.date.date()), "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                   "volume": r.volume} for r in g.itertuples()] for sym, g in prices.groupby("ticker")}
    last = prices["date"].max()
    yesterday = prices["date"].drop_duplicates().nlargest(2).iloc[-1]
    headlines = [
        {"title": "AAA beats estimates and raises guidance", "published_at": str(yesterday.date())},
        {"title": "AAA downgraded at a major broker", "published_at": f"{last.date()}T12:00:00Z"},  # before the open
        {"title": "AAA announces a new product", "published_at": f"{last.date()}T16:00:00Z"},  # after the open
    ]
    return {"ticker": "aaa", "headlines": headlines, "prices": bars}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_uses_only_news_public_before_the_open(client):
    resp = client.post("/predict", json=_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ticker"] == "AAA"
    assert 0 <= body["probability_beats_market"] <= 1
    assert body["signal"] in {"outperform", "underperform"}
    assert body["headlines_used"] == 2  # yesterday's date-only item and the pre-open item, not the later one


def test_predict_rejects_too_little_history(client):
    assert client.post("/predict", json=_payload(n_days=10)).status_code == 422


def test_sentiment_endpoint(client):
    body = client.post("/sentiment", json={"headlines": ["AAA beats estimates", "AAA misses estimates"]}).json()
    assert body[0]["bullish"] > body[0]["bearish"] and body[1]["bearish"] > body[1]["bullish"]
