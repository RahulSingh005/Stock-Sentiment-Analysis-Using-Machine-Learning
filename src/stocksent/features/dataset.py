"""Join market and news features, add labels, and split by time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .market import PRICE_FEATURES
from .news_features import NEWS_FEATURES, VADER_FEATURES

# does the opening gap agree with the overnight news? (news-driven gaps tend to behave differently)
INTERACTIONS = ["sent_gap_agree", "news_gap"]

FEATURE_SETS = {
    "price": PRICE_FEATURES,
    "news": NEWS_FEATURES,
    "price+news": PRICE_FEATURES + NEWS_FEATURES + INTERACTIONS,
    "vader": VADER_FEATURES,
}


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df["sent_gap_agree"] = np.sign(df["fs_net_mean"]) * df["gap_z"].clip(-5, 5)
    df["news_gap"] = df["log_news"] * df["gap_z"].clip(-5, 5)
    return df


def assemble(market: pd.DataFrame, news: pd.DataFrame, text: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = market.merge(news, on=["ticker", "date"], how="inner").merge(text, on=["ticker", "date"], how="left")
    df["text"] = df["text"].fillna("")
    df = add_interactions(df)
    # target: does the stock beat the market (SPY) from today's open to today's close?
    # company news can only explain the company-specific part of a move, not the market's direction
    df["fwd_excess"] = df["fwd_ret"] - df["mkt_fwd_ret"]
    df["y"] = (df["fwd_excess"] > 0).astype(int)
    df["y_raw"] = (df["fwd_ret"] > 0).astype(int)

    start = df["ticker"].map(lambda t: pd.Timestamp(cfg.coverage_start.get(t, cfg.start)))
    df = df[(df["date"] >= start) & (df["date"] >= pd.Timestamp(cfg.start)) & (df["date"] <= pd.Timestamp(cfg.end))]
    df = df.dropna(subset=[*PRICE_FEATURES, "fwd_ret", "fwd_excess"])
    df = df[df["fwd_excess"] != 0]

    df["split"] = "test"
    df.loc[df["date"] <= pd.Timestamp(cfg.valid_end), "split"] = "valid"
    df.loc[df["date"] <= pd.Timestamp(cfg.train_end), "split"] = "train"
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)
