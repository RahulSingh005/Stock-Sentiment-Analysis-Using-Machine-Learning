"""Map news timestamps to the trading session they can first be traded on.

Predictions are made before the open (09:30 New York time). An article counts toward session t only if it was
public before session t opened. Most FNSPID timestamps carry a date but no time of day; such an article could
have appeared at any point of that date, so it is first used on the next session after its date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_TZ = "America/New_York"
OPEN_MINUTES = 9 * 60 + 30


def assign_sessions(published: pd.Series, sessions: pd.DatetimeIndex, date_only: pd.Series | None = None) -> pd.Series:
    """Return the session date each article is first usable for (NaT if after the last session).

    published: tz-aware timestamps. sessions: sorted, tz-naive trading dates.
    date_only: mask of rows whose timestamp is a bare calendar date (stored as midnight UTC).
    """
    local = published.dt.tz_convert(MARKET_TZ)
    minutes = local.dt.hour * 60 + local.dt.minute
    day = local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta((minutes >= OPEN_MINUTES).astype(int), unit="D")
    if date_only is not None:
        calendar_day = published.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        day = day.where(~date_only.to_numpy(), calendar_day + pd.Timedelta(days=1))
    idx = np.searchsorted(sessions.values, day.values, side="left")
    out = pd.Series(pd.NaT, index=published.index, dtype="datetime64[ns]")
    valid = idx < len(sessions)
    out[valid] = sessions.values[idx[valid]]
    return out
