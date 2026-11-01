import numpy as np
import pandas as pd
import pytest

from stocksent.features.sentiment import SENT_COLUMNS


def make_prices(symbols=("AAA", "SPY", "^VIX"), n=60, seed=0, start="2023-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    frames = []
    for sym in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        open_ = close * np.exp(rng.normal(0, 0.005, n))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": sym, "open": open_, "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99, "close": close,
            "volume": rng.integers(1_000_000, 2_000_000, n) * 1.0,
        }))
    return pd.concat(frames, ignore_index=True)


class StubSentiment:
    """Keyword sentiment so tests do not need the trained model."""

    report = {}

    def score(self, titles):
        rows = []
        for t in titles:
            t = t.lower()
            if "beat" in t or "upgrade" in t:
                rows.append([0.8, 0.1, 0.1])
            elif "miss" in t or "downgrade" in t:
                rows.append([0.1, 0.8, 0.1])
            else:
                rows.append([0.1, 0.1, 0.8])
        return pd.DataFrame(rows, columns=SENT_COLUMNS)


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def stub_sentiment():
    return StubSentiment()
