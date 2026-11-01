"""Price and market features known at the open of session t, and the open-to-close target.

Everything except the opening gap comes from the previous session's close or earlier. The gap is the first
price of session t, so it is known at the moment the prediction is made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = [
    "gap", "gap_z", "rel_gap", "mkt_gap", "ret_1", "ret_5", "ret_20", "vol_20", "prev_intraday", "prev_range",
    "volume_z", "rsi_14", "ma_gap_20", "mkt_ret_1", "mkt_ret_5", "vix_log", "vix_chg_1",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _per_ticker(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values("date").copy()
    prev_close = p["close"].shift(1)
    ret = np.log(p["close"] / prev_close)
    intraday = np.log(p["close"] / p["open"])
    log_vol = np.log1p(p["volume"])
    vol_20 = ret.rolling(20).std()

    out = p[["ticker", "date"]].copy()
    out["gap"] = np.log(p["open"] / prev_close)  # known at the open
    out["ret_1"] = ret.shift(1)
    out["ret_5"] = ret.rolling(5).sum().shift(1)
    out["ret_20"] = ret.rolling(20).sum().shift(1)
    out["vol_20"] = vol_20.shift(1)
    out["gap_z"] = out["gap"] / out["vol_20"]
    out["prev_intraday"] = intraday.shift(1)
    out["prev_range"] = np.log(p["high"] / p["low"]).shift(1)
    out["volume_z"] = ((log_vol - log_vol.shift(1).rolling(20).mean()) / log_vol.shift(1).rolling(20).std()).shift(1)
    out["rsi_14"] = (_rsi(p["close"]) / 100).shift(1)
    out["ma_gap_20"] = (p["close"] / p["close"].rolling(20).mean() - 1).shift(1)
    out["fwd_ret"] = intraday  # target: open(t) -> close(t)
    return out


def build_market_features(prices: pd.DataFrame, tickers: list[str], benchmark: str, vix: str) -> pd.DataFrame:
    stocks = pd.concat([_per_ticker(prices[prices["ticker"] == t]) for t in tickers], ignore_index=True)
    mkt = _per_ticker(prices[prices["ticker"] == benchmark])[["date", "gap", "ret_1", "ret_5", "fwd_ret"]]
    mkt = mkt.rename(columns={"gap": "mkt_gap", "ret_1": "mkt_ret_1", "ret_5": "mkt_ret_5", "fwd_ret": "mkt_fwd_ret"})
    v = prices[prices["ticker"] == vix].sort_values("date")[["date", "close"]]
    log_v = np.log(v["close"])
    v = pd.DataFrame({"date": v["date"], "vix_log": log_v.shift(1), "vix_chg_1": log_v.diff().shift(1)})
    df = stocks.merge(mkt, on="date", how="left").merge(v, on="date", how="left")
    df[["vix_log", "vix_chg_1"]] = df[["vix_log", "vix_chg_1"]].ffill()
    df["rel_gap"] = df["gap"] - df["mkt_gap"]
    return df[["ticker", "date", *PRICE_FEATURES, "fwd_ret", "mkt_fwd_ret"]]
