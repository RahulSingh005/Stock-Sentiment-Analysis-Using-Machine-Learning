"""Daily news features per ticker and session.

Headline-level scores (finance sentiment model, VADER, event flags) are aggregated to the session they belong
to, then turned into rolling features that only look backwards. The same functions run in training and in the API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sentiment import SENT_COLUMNS, vader_scores

# events that tend to move a stock beyond its general tone
EVENT_PATTERNS = {
    "ev_upgrade": r"\bupgrade[sd]?\b|\b(?:raises?|boosts?|lifts?) (?:its |the )?(?:price )?target|\boutperform\b",
    "ev_downgrade": r"\bdowngrade[sd]?\b|\b(?:cuts?|lowers?|slashes) (?:its |the )?(?:price )?target|\bunderperform\b",
    "ev_beat": r"\b(?:beats?|tops?|surpass(?:es)?|exceeds?)\b.*\b(?:estimates?|expectations?|consensus|views?)\b",
    "ev_miss": r"\bmiss(?:es|ed)?\b.*\b(?:estimates?|expectations?|consensus|views?)\b|\bfalls? short\b",
    "ev_earnings": r"\bearnings\b|\bq[1-4]\b|\bquarterly\b|\bguidance\b|\boutlook\b",
    "ev_legal": r"\blawsuit\b|\bantitrust\b|\bprobe\b|\binvestigation\b|\bfined?\b|\bregulators?\b",
}
EVENTS = list(EVENT_PATTERNS)

NEWS_FEATURES = [
    "log_news", "news_volume_z", "fs_net_mean", "fs_net_std", "fs_net_min", "fs_net_max", "fs_neg_mean",
    "fs_pos_share", "fs_neg_share", "fs_net_ewm5", "fs_net_surprise", "rating_net_5d", *EVENTS,
]
VADER_FEATURES = ["log_news", "news_volume_z", "vader_mean", "vader_neg_share"]


def score_headlines(news: pd.DataFrame, sentiment_model) -> pd.DataFrame:
    """Add finance-sentiment probabilities, VADER compound and event flags to each headline."""
    if news.empty:
        return news.assign(**{c: pd.Series(dtype=float) for c in [*SENT_COLUMNS, "fs_net", "vader", *EVENTS]})
    titles = news["title"].tolist()
    scored = news.reset_index(drop=True).copy()
    scored[SENT_COLUMNS] = sentiment_model.score(titles).to_numpy()
    scored["fs_net"] = scored["fs_pos"] - scored["fs_neg"]
    scored["vader"] = vader_scores(titles)
    for name, pattern in EVENT_PATTERNS.items():
        scored[name] = scored["title"].str.contains(pattern, case=False, regex=True).astype(int)
    return scored


def aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, session) that had news."""
    label = scored[SENT_COLUMNS].to_numpy().argmax(axis=1)  # 0 bullish, 1 bearish, 2 neutral
    g = scored.assign(is_pos=label == 0, is_neg=label == 1, vader_neg=scored["vader"] <= -0.05).groupby(
        ["ticker", "session"])
    daily = g.agg(
        n_news=("title", "size"),
        fs_net_mean=("fs_net", "mean"),
        fs_net_std=("fs_net", "std"),
        fs_net_min=("fs_net", "min"),
        fs_net_max=("fs_net", "max"),
        fs_neg_mean=("fs_neg", "mean"),
        fs_pos_share=("is_pos", "mean"),
        fs_neg_share=("is_neg", "mean"),
        vader_mean=("vader", "mean"),
        vader_neg_share=("vader_neg", "mean"),
        **{e: (e, "sum") for e in EVENTS},
    )
    return daily.reset_index().rename(columns={"session": "date"})


def build_news_features(scored: pd.DataFrame, sessions: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Full (ticker, date) grid with rolling features; days without news get neutral values."""
    daily = aggregate_daily(scored.dropna(subset=["session"]))
    grid = pd.MultiIndex.from_product([tickers, sessions], names=["ticker", "date"]).to_frame(index=False)
    df = grid.merge(daily, on=["ticker", "date"], how="left")
    df["n_news"] = df["n_news"].fillna(0)
    df = df.fillna(0.0).sort_values(["ticker", "date"]).reset_index(drop=True)

    df["log_news"] = np.log1p(df["n_news"])
    g = df.groupby("ticker", group_keys=False)
    # baselines use the previous 20 sessions only, so today's value is compared with the past
    prev_mean = g["log_news"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    prev_std = g["log_news"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    df["news_volume_z"] = ((df["log_news"] - prev_mean) / prev_std.replace(0, np.nan)).fillna(0).clip(-5, 5)
    df["fs_net_ewm5"] = g["fs_net_mean"].transform(lambda s: s.ewm(span=5, adjust=False).mean())
    prev_net = g["fs_net_mean"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    df["fs_net_surprise"] = (df["fs_net_mean"] - prev_net).fillna(0)
    df["rating_net_5d"] = g["ev_upgrade"].transform(lambda s: s.rolling(5, min_periods=1).sum()) - g[
        "ev_downgrade"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    return df


def daily_text(scored: pd.DataFrame) -> pd.DataFrame:
    """All headlines of a (ticker, session) joined into one document, for the bag-of-words model."""
    text = scored.dropna(subset=["session"]).groupby(["ticker", "session"])["title"].apply(" . ".join)
    return text.reset_index().rename(columns={"session": "date", "title": "text"})
