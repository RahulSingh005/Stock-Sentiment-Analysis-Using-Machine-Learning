"""Load prices and news, score articles with VADER, and map each article to a trading session."""
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR = Path(__file__).resolve().parent.parent / "Data-Set"
NEWS_FILE = {"GOOGL": "Google_News_2023.csv", "MSFT": "Microsoft_News_2023.csv"}

# headlines that plausibly move markets (macro, rates, tech, earnings)
MARKET_TERMS = (r"stock|market|wall street|nasdaq|s&p|dow|economy|economic|inflation|interest rate|fed\b"
                r"|federal reserve|recession|bank|earnings|tech|artificial intelligence|\bai\b|chip|jobs report|layoff")


def load_prices(ticker, data_dir=DATA_DIR):
    prices = pd.read_csv(data_dir / f"{ticker}_Financial_Data.csv")
    prices["Date"] = pd.to_datetime(prices["Date"], format="%d-%m-%Y")
    return prices.sort_values("Date").reset_index(drop=True)


def news_overlap(data_dir=DATA_DIR):
    """The two news files turned out to be the same feed; return (identical?, share mentioning Google)."""
    google = pd.read_csv(data_dir / NEWS_FILE["GOOGL"])
    microsoft = pd.read_csv(data_dir / NEWS_FILE["MSFT"])
    identical = google.drop(columns="Query").equals(microsoft.drop(columns="Query"))
    mentions = google["Headlines"].str.contains("Google|Alphabet", case=False, na=False).mean()
    return identical, mentions


def load_news(data_dir=DATA_DIR):
    """Score every article on its own (headline + summary) rather than one long string per day."""
    news = pd.read_csv(data_dir / NEWS_FILE["GOOGL"])
    text = news["Headlines"].fillna("") + ". " + news["Summary"].fillna("")
    analyzer = SentimentIntensityAnalyzer()
    news["compound"] = [analyzer.polarity_scores(t)["compound"] for t in text]
    news["market"] = text.str.contains(MARKET_TERMS, case=False, regex=True)
    news["published"] = pd.to_datetime(news["Publication_Date"], utc=True).dt.tz_convert("America/New_York")
    return news


def daily_sentiment(news, sessions):
    """Aggregate articles into the first session that closes after they were published.

    Anything published after the 4 pm ET close counts toward the next session, and weekend news rolls
    into Monday, so a feature for day t never contains information from after day t's close.
    """
    after_close = news["published"].dt.hour >= 16
    day = news["published"].dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    session = np.searchsorted(sessions.values, day.values, side="left")
    news = news.assign(session=session)[session < len(sessions)]

    def negative_share(s):
        return (s <= -0.05).mean()

    by_day = news.groupby("session")["compound"]
    by_day_market = news[news["market"]].groupby("session")["compound"]
    daily = pd.DataFrame({
        "sent_mean": by_day.mean(),
        "sent_pos": by_day.apply(lambda s: (s >= 0.05).mean()),
        "sent_neg": by_day.apply(negative_share),
        "sent_std": by_day.std(),
        "n_news": by_day.size(),
        "mkt_sent": by_day_market.mean(),
        "mkt_neg": by_day_market.apply(negative_share),
        "n_mkt": by_day_market.size(),
    })
    return daily.reindex(range(len(sessions)))
