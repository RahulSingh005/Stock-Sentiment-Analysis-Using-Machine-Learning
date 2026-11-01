"""Daily prices from Yahoo Finance (split/dividend adjusted), cached as parquet."""
from __future__ import annotations

import logging
import time

import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)


def download_prices(cfg: Config) -> pd.DataFrame:
    """Long table: date, ticker, open, high, low, close, volume for the universe, benchmark and VIX."""
    out = cfg.raw_dir / "prices.parquet"
    if out.exists():
        return pd.read_parquet(out)
    import yfinance as yf

    symbols = [*cfg.tickers, cfg.benchmark, cfg.volatility_index]
    # a little history before the start so rolling features are defined from day one
    start = (pd.Timestamp(cfg.start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(cfg.end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    log.info("downloading prices for %d symbols", len(symbols))
    frames = []
    for sym in symbols:
        for attempt in range(3):  # Yahoo occasionally drops a symbol from a request; retry it on its own
            part = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False)
            if not part.empty:
                break
            time.sleep(2 * (attempt + 1))
        if part.empty:
            raise RuntimeError(f"no price data returned for {sym}")
        part = part.dropna(how="all").rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        frames.append(part.assign(ticker=sym).rename_axis("date").reset_index())
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out, index=False)
    return prices


def recent_prices(symbols: list[str], days: int = 120) -> pd.DataFrame:
    """Last few months of daily bars for live scoring (same layout as download_prices)."""
    import yfinance as yf

    frames = []
    for sym in symbols:
        part = yf.download(sym, period=f"{days}d", auto_adjust=True, progress=False, multi_level_index=False)
        if part.empty:
            raise ValueError(f"no recent prices for {sym}")
        part = part.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        frames.append(part.assign(ticker=sym).rename_axis("date").reset_index())
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    return prices


def trading_sessions(prices: pd.DataFrame, cfg: Config) -> pd.DatetimeIndex:
    """Trading dates of the benchmark, used as the session calendar."""
    dates = prices.loc[prices["ticker"] == cfg.benchmark, "date"]
    return pd.DatetimeIndex(sorted(dates.unique()))
