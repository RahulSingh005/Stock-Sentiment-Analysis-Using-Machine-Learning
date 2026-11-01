"""Daily backtests of model signals, net of transaction costs.

Signals are formed before the open of session t. Positions are bought at the open and sold at the close of the
same session, so each position pays the cost twice (in and out) and nothing is held overnight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="ticker", values=col).sort_index()


def _returns(weights: pd.DataFrame, simple_ret: pd.DataFrame, cost_bps: float) -> pd.Series:
    gross = (weights * simple_ret).sum(axis=1)
    return gross - 2 * weights.abs().sum(axis=1) * cost_bps / 1e4


def long_short(df: pd.DataFrame, frac: float = 1 / 3, cost_bps: float = 2.0) -> pd.Series:
    """Each day: long the top `frac` of stocks by predicted probability, short the bottom `frac`, equal weights."""
    proba, ret = _pivot(df, "proba"), np.expm1(_pivot(df, "fwd_ret")).fillna(0)
    k = np.maximum(1, np.floor(proba.notna().sum(axis=1) * frac))
    longs = proba.rank(axis=1, ascending=False, method="first").le(k, axis=0).astype(float)
    shorts = proba.rank(axis=1, ascending=True, method="first").le(k, axis=0).astype(float)
    weights = longs.div(longs.sum(axis=1), axis=0) - shorts.div(shorts.sum(axis=1), axis=0)
    return _returns(weights.fillna(0) / 2, ret, cost_bps)  # half the capital long, half short


def long_top(df: pd.DataFrame, frac: float = 1 / 3, cost_bps: float = 2.0) -> pd.Series:
    """Long-only version: hold the top `frac` of stocks by predicted probability, equal weights."""
    proba, ret = _pivot(df, "proba"), np.expm1(_pivot(df, "fwd_ret")).fillna(0)
    k = np.maximum(1, np.floor(proba.notna().sum(axis=1) * frac))
    longs = proba.rank(axis=1, ascending=False, method="first").le(k, axis=0).astype(float)
    return _returns(longs.div(longs.sum(axis=1), axis=0).fillna(0), ret, cost_bps)


def always_long(df: pd.DataFrame, cost_bps: float = 2.0) -> pd.Series:
    """Benchmark: buy every stock at the open and sell at the close, every day."""
    ret = np.expm1(_pivot(df, "fwd_ret"))
    weights = ret.notna().astype(float).div(ret.notna().sum(axis=1), axis=0)
    return _returns(weights, ret.fillna(0), cost_bps)


def performance(r: pd.Series) -> dict:
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    active = r[r != 0]
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": float(equity.iloc[-1] ** (1 / years) - 1),
        "annual_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "days": int(len(r)),
    }
