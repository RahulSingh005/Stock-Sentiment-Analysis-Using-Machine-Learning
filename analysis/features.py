"""Feature engineering. Every feature for row t uses data available at the close of day t."""
import numpy as np
import pandas as pd

# first session after each earnings release (both companies report after the close);
# the dates are published weeks in advance, so using them is not look-ahead
EARNINGS_REACTION = {
    "GOOGL": ["2023-02-03", "2023-04-26", "2023-07-26", "2023-10-25"],
    "MSFT": ["2023-01-25", "2023-04-26", "2023-07-26", "2023-10-25"],
}

SENTIMENT = ["sent_mean_z", "sent_neg_z", "mkt_sent_z"]
HAR = ["tr", "tr_ma5", "tr_ma20"]
LOG_HAR = ["log_tr", "log_tr_ma5", "log_tr_ma20"]


def zscore(s, window=20):
    rolling = s.rolling(window)
    return (s - rolling.mean()) / rolling.std()


def build_features(prices, sentiment, ticker):
    df = prices.join(sentiment)
    prev_close = df["Close"].shift(1)

    df["ret"] = np.log(df["Close"] / prev_close)
    # true range (in %) includes the overnight gap, which is where earnings moves show up
    df["tr"] = 100 * np.log(np.maximum(df["High"], prev_close) / np.minimum(df["Low"], prev_close))
    df["tr_ma5"] = df["tr"].rolling(5).mean()
    df["tr_ma20"] = df["tr"].rolling(20).mean()
    for col in HAR:
        df[f"log_{col}"] = np.log(df[col])
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    for k in (1, 2, 3):
        df[f"ret_l{k}"] = df["ret"].shift(k - 1)

    for col in ("sent_mean", "sent_neg", "mkt_sent"):
        df[f"{col}_z"] = zscore(df[col])

    reaction_days = pd.to_datetime(EARNINGS_REACTION[ticker])
    df["earn_next"] = df["Date"].shift(-1).isin(reaction_days).astype(int)

    # targets: tomorrow's direction and tomorrow's true range
    df["target_date"] = df["Date"].shift(-1)
    df["y_up"] = (df["ret"].shift(-1) > 0).astype(int)
    df["y_tr"] = df["tr"].shift(-1)
    df["y_log_tr"] = np.log(df["y_tr"])
    df["ret_next"] = df["ret"].shift(-1)
    return df.dropna().reset_index(drop=True)
