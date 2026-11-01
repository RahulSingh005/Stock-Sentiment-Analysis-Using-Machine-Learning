"""Score one ticker for a session, reusing the exact feature code used in training."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .data.sessions import assign_sessions
from .features.dataset import add_interactions
from .features.market import build_market_features
from .features.news_features import build_news_features, score_headlines
from .features.sentiment import FinanceSentimentModel


@dataclass
class Predictor:
    bundle: dict
    sentiment: FinanceSentimentModel

    @classmethod
    def load(cls, models_dir) -> Predictor:
        bundle = joblib.load(models_dir / "production_model.joblib")
        return cls(bundle, FinanceSentimentModel.load(models_dir / "finance_sentiment.joblib"))

    def predict(self, ticker: str, headlines: pd.DataFrame, prices: pd.DataFrame) -> dict:
        """headlines: published (tz-aware), title, date_only. prices: date, ticker, open, high, low, close, volume
        for the ticker, the benchmark and the volatility index; the last date is the session being predicted."""
        benchmark, vix = self.bundle["benchmark"], self.bundle["volatility_index"]
        missing = {ticker, benchmark, vix} - set(prices["ticker"])
        if missing:
            raise ValueError(f"prices missing for {sorted(missing)}")
        sessions = pd.DatetimeIndex(sorted(prices.loc[prices["ticker"] == benchmark, "date"].unique()))
        session = sessions[-1]

        news = headlines.assign(ticker=ticker)
        news["session"] = assign_sessions(news["published"], sessions, news["date_only"])
        scored = score_headlines(news, self.sentiment)
        news_features = build_news_features(scored, sessions, [ticker])
        market = build_market_features(prices, [ticker], benchmark, vix)
        row = add_interactions(market.merge(news_features, on=["ticker", "date"]))
        row = row[row["date"] == session]
        features = self.bundle["features"]
        if row.empty or row[features].isna().any(axis=None):
            raise ValueError("not enough price history to build features (need about 25 sessions)")
        proba = float(self.bundle["estimator"].predict_proba(row[features])[:, 1][0])
        today = scored[scored["session"] == session]
        return {
            "ticker": ticker,
            "session": session.date().isoformat(),
            "probability_beats_market": proba,
            "signal": "outperform" if proba > 0.5 else "underperform",
            "headlines_used": int(len(today)),
            "net_sentiment": float(today["fs_net"].mean()) if len(today) else 0.0,
            "opening_gap": float(row["gap"].iloc[0]),
            "model": self.bundle["model"],
            "trained_through": self.bundle["trained_through"],
            "headline_scores": [
                {"title": r.title, "bullish": round(r.fs_pos, 4), "bearish": round(r.fs_neg, 4),
                 "neutral": round(r.fs_neu, 4)} for r in today.itertuples()
            ],
        }


def headlines_frame(items: list[dict]) -> pd.DataFrame:
    """Build the headline frame from API input; bare dates are treated like FNSPID date-only stamps."""
    if not items:
        return pd.DataFrame({"published": pd.Series(dtype="datetime64[ns, UTC]"), "title": pd.Series(dtype=str),
                             "date_only": pd.Series(dtype=bool)})
    raw = pd.Series([str(i["published_at"]) for i in items])
    date_only = raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    published = pd.to_datetime(raw, utc=True, format="mixed")
    return pd.DataFrame({"published": published, "title": [i["title"] for i in items], "date_only": date_only.to_numpy(
        dtype=bool)})


def prices_frame(rows: dict[str, list[dict]]) -> pd.DataFrame:
    frames = [pd.DataFrame(bars).assign(ticker=sym) for sym, bars in rows.items()]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": np.float64})
