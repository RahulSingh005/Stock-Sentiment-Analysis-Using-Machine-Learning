import numpy as np
import pandas as pd
import pytest

from stocksent.evaluation.backtest import long_short, long_top, performance


def _frame(n_days=20, n_stocks=6, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = [(d, f"S{i}", rng.normal(0, 0.01)) for d in dates for i in range(n_stocks)]
    return pd.DataFrame(rows, columns=["date", "ticker", "fwd_ret"])


def test_perfect_foresight_long_short_makes_money_every_day():
    df = _frame()
    r = long_short(df.assign(proba=df["fwd_ret"]), cost_bps=0)
    assert (r > 0).all()


def test_costs_are_charged_on_both_sides_of_every_position():
    df = _frame()
    df["proba"] = np.random.default_rng(2).random(len(df))
    gross, net = long_short(df, cost_bps=0), long_short(df, cost_bps=10)
    # half the capital long and half short: gross exposure 1, paid on the way in and out
    np.testing.assert_allclose(gross - net, 2 * 1.0 * 10 / 1e4)


def test_long_top_holds_a_third_of_the_stocks():
    df = _frame()
    df["proba"] = df["fwd_ret"]
    r = long_top(df, cost_bps=0)
    best_two = df.groupby("date")["fwd_ret"].apply(lambda s: np.expm1(s.nlargest(2)).mean())
    np.testing.assert_allclose(r.to_numpy(), best_two.to_numpy())


def test_performance_metrics():
    stats = performance(pd.Series([0.1, -0.5, 0.2]))
    assert stats["max_drawdown"] == pytest.approx(-0.5)
    assert stats["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert stats["win_rate"] == pytest.approx(2 / 3)
