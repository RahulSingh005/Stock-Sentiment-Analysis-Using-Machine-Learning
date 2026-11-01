"""Company news: download from the FNSPID Nasdaq-100 dataset, then clean it.

FNSPID (Dong et al., KDD 2024) maps each article to a ticker. Only the columns this project uses are read,
so the download is a small fraction of the full parquet files.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

COLUMNS = ["Date", "Article_title", "Stock_symbol", "Publisher"]

# auto-generated or non-informative headlines that would only add noise
SPAM_PATTERNS = re.compile(
    r"shares (?:bought|sold|acquired) by|position (?:in|lifted|trimmed|raised|lowered)|stake in|holdings? (?:in|of)"
    r"|\b13f\b|options? (?:volume|activity) |unusual options|premarket movers|pre-market movers|stocks? to watch"
    r"|validea|validea's|guru fundamental report|noteworthy etf|etf (?:inflows|outflows)|implied volatility"
    r"|first week of .* options trading|interesting .* put and call options",
    re.IGNORECASE,
)


def download_news(cfg: Config) -> pd.DataFrame:
    """Headline columns for every symbol in the dataset (cached once), filtered to the configured universe."""
    out = cfg.raw_dir / "fnspid_nasdaq100_titles.parquet"
    if not out.exists():
        frames = []
        for name in cfg.news_files:
            path = f"hf://datasets/{cfg.news_dataset}/{name}"
            log.info("reading %s", path)
            frames.append(pd.read_parquet(path, columns=COLUMNS))
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    news = pd.read_parquet(out)
    wanted = {cfg.news_symbol_aliases.get(t, t) for t in cfg.tickers} | set(cfg.tickers)
    return news[news["Stock_symbol"].isin(wanted)].reset_index(drop=True)


def clean_news(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Normalise tickers and timestamps, drop spam and duplicates. Returns ticker, published, title, publisher."""
    alias_to_ticker = {v: k for k, v in cfg.news_symbol_aliases.items()}
    df = pd.DataFrame({
        "ticker": raw["Stock_symbol"].map(lambda s: alias_to_ticker.get(s, s)),
        "published": pd.to_datetime(raw["Date"], utc=True, errors="coerce"),
        "title": raw["Article_title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "publisher": raw["Publisher"].fillna("unknown"),
    })
    df = df.dropna(subset=["published"])
    df = df[df["ticker"].isin(cfg.tickers) & (df["title"].str.len() >= 15)]
    start, end = pd.Timestamp(cfg.start, tz="UTC"), pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["published"] >= start) & (df["published"] <= end)]
    df = df[~df["title"].str.contains(SPAM_PATTERNS)]
    # keep headlines that are actually about the company: they name it or its ticker
    named = pd.Series(False, index=df.index)
    for ticker, names in cfg.company_names.items():
        rows = df["ticker"] == ticker
        pattern = re.compile(rf"\b(?:{names})\b", re.IGNORECASE)
        symbol = re.compile(rf"\b{re.escape(ticker)}\b")
        named[rows] = df.loc[rows, "title"].map(lambda s, p=pattern, q=symbol: bool(p.search(s) or q.search(s)))
    df = df[named]
    # timestamps that sit exactly on midnight UTC carry a date but no reliable time of day
    ts = df["published"].dt
    df["date_only"] = (ts.hour == 0) & (ts.minute == 0) & (ts.second == 0)
    df["title_key"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    df = df.sort_values("published").drop_duplicates(["ticker", "title_key"]).drop(columns="title_key")
    return df.reset_index(drop=True)
