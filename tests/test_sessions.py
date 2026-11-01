import pandas as pd

from stocksent.data.sessions import assign_sessions

# Friday, Monday, Tuesday (January, so New York is UTC-5)
SESSIONS = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])


def _assign(stamps, date_only=None):
    published = pd.Series(pd.to_datetime(stamps, utc=True))
    mask = None if date_only is None else pd.Series(date_only)
    return assign_sessions(published, SESSIONS, mask).tolist()


def test_news_before_the_open_is_usable_the_same_day():
    assert _assign(["2024-01-08 13:00"]) == [pd.Timestamp("2024-01-08")]  # 08:00 New York


def test_news_after_the_open_waits_for_the_next_session():
    assert _assign(["2024-01-08 15:00"]) == [pd.Timestamp("2024-01-09")]  # 10:00 New York


def test_weekend_news_rolls_to_monday():
    assert _assign(["2024-01-06 15:00"]) == [pd.Timestamp("2024-01-08")]


def test_date_only_news_is_used_on_the_next_session_after_its_date():
    assert _assign(["2024-01-08 00:00"], [True]) == [pd.Timestamp("2024-01-09")]


def test_news_after_the_last_session_has_no_session():
    assert pd.isna(_assign(["2024-01-09 18:00"])[0])
