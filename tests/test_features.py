import numpy as np
import pandas as pd

from stocksent.data.sessions import assign_sessions
from stocksent.features.market import PRICE_FEATURES, build_market_features
from stocksent.features.news_features import build_news_features, score_headlines


def test_market_features_do_not_see_anything_after_the_open(prices):
    """Changing the close, high, low and volume of the last day must not change that day's features."""
    last = prices["date"].max()
    base = build_market_features(prices, ["AAA"], "SPY", "^VIX")
    shocked = prices.copy()
    after_open = shocked["date"] == last
    shocked.loc[after_open, ["close", "high", "low", "volume"]] *= 1.5
    moved = build_market_features(shocked, ["AAA"], "SPY", "^VIX")
    a = base.loc[base["date"] == last, PRICE_FEATURES].reset_index(drop=True)
    b = moved.loc[moved["date"] == last, PRICE_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_target_is_open_to_close_return(prices):
    feats = build_market_features(prices, ["AAA"], "SPY", "^VIX").set_index("date")
    p = prices[prices["ticker"] == "AAA"].set_index("date")
    np.testing.assert_allclose(feats["fwd_ret"], np.log(p["close"] / p["open"]))


def _news(rows):
    df = pd.DataFrame(rows, columns=["published", "title"])
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["date_only"] = False
    df["ticker"] = "AAA"
    return df


def test_event_flags(stub_sentiment):
    news = _news([("2023-01-03 12:00", "Intel downgraded to Sell at Goldman"),
                  ("2023-01-03 12:00", "Nvidia beats Q4 estimates"),
                  ("2023-01-03 12:00", "Apple faces antitrust probe in EU")])
    scored = score_headlines(news, stub_sentiment)
    assert scored[["ev_downgrade", "ev_beat", "ev_legal"]].to_numpy().diagonal().tolist() == [1, 1, 1]


def test_news_features_only_look_backwards(prices, stub_sentiment):
    sessions = pd.DatetimeIndex(sorted(prices["date"].unique()))
    early = _news([(f"{d.date()} 12:00", "Company beats estimates") for d in sessions[5:40:3]])
    late = pd.concat([early, _news([(f"{sessions[50].date()} 12:00", "Company misses estimates")])])

    def build(news):
        news = news.assign(session=assign_sessions(news["published"], sessions, news["date_only"]))
        return build_news_features(score_headlines(news, stub_sentiment), sessions, ["AAA"]).set_index("date")

    a, b = build(early), build(late)
    pd.testing.assert_frame_equal(a.loc[: sessions[49]], b.loc[: sessions[49]])
    assert b.loc[sessions[50], "fs_neg_share"] == 1.0
    assert a.loc[sessions[1], "n_news"] == 0 and a.loc[sessions[1], "fs_net_mean"] == 0
