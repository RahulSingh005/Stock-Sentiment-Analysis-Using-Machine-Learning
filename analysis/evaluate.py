"""Experiments: audit of the original setup, next-day direction, and next-day volatility."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .features import HAR, LOG_HAR, SENTIMENT

FIRST_TEST_ROW = 120  # walk-forward starts here; roughly the last five months of 2023 are out-of-sample
SEED = 42


def walk_forward(df, cols, make_model, target, proba=False):
    """Refit on all rows before t, predict row t, step forward one day."""
    preds = []
    for t in range(FIRST_TEST_ROW, len(df)):
        model = make_model().fit(df[cols].iloc[:t], df[target].iloc[:t])
        x = df[cols].iloc[[t]]
        preds.append(model.predict_proba(x)[0, 1] if proba else model.predict(x)[0])
    return np.array(preds)


def audit_original(prices, sentiment):
    """Reproduce the two headline numbers from the original notebooks and show where they come from."""
    df = prices.join(sentiment).fillna(0)
    split = 186  # same split as the notebooks
    label = (df["sent_pos"] > df["sent_neg"]).astype(int)  # 'Movement' was defined from the sentiment inputs
    cols = ["Open", "High", "Low", "Volume", "sent_pos", "sent_neg"]
    clf = RandomForestClassifier(random_state=SEED).fit(df[cols][:split], label[:split])

    same_day = DecisionTreeRegressor(random_state=SEED).fit(df[["Open", "High", "Low", "Volume", "sent_mean"]][:split],
                                                            df["Close"][:split])
    test = df[split:]
    return {
        "circular_accuracy": accuracy_score(label[split:], clf.predict(df[cols][split:])),
        "same_day_r2_model": r2_score(test["Close"], same_day.predict(test[["Open", "High", "Low", "Volume", "sent_mean"]])),
        "same_day_r2_midpoint": r2_score(test["Close"], (test["High"] + test["Low"]) / 2),
    }


DIRECTION_FEATURES = {
    "All-news sentiment": ["sent_mean", "sent_neg", "sent_std", "sent_mean_z"],
    "Market-news sentiment": ["mkt_sent", "mkt_neg", "mkt_sent_z", "n_mkt"],
    "Sentiment + price": SENTIMENT + ["ret_l1", "ret_l2", "tr", "vol_ratio"],
}
DIRECTION_MODELS = {
    "Logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                                                random_state=SEED),
}


def direction_results(df):
    y = df["y_up"].iloc[FIRST_TEST_ROW:].to_numpy()
    majority = np.array([df["y_up"].iloc[:t].mode()[0] for t in range(FIRST_TEST_ROW, len(df))])
    rows = [("Majority class", "-", accuracy_score(y, majority), np.nan)]
    for fname, cols in DIRECTION_FEATURES.items():
        for mname, make in DIRECTION_MODELS.items():
            p = walk_forward(df, cols, make, "y_up", proba=True)
            rows.append((fname, mname, accuracy_score(y, (p > 0.5).astype(int)), roc_auc_score(y, p)))
    rho, pval = spearmanr(df["sent_mean_z"], df["ret_next"])
    return rows, (rho, pval)


def _log_model(cols):
    """Fit on log true range; predictions come back in % (volatility is right-skewed)."""
    return cols, "y_log_tr"


VOL_MODELS = {
    "20-day average (baseline)": None,
    "Yesterday's range (baseline)": None,
    "HAR": (HAR, "y_tr"),
    "HAR + earnings": (HAR + ["earn_next"], "y_tr"),
    "Log-HAR + earnings": _log_model(LOG_HAR + ["earn_next"]),
    "Log-HAR + earnings + sentiment": _log_model(LOG_HAR + ["earn_next"] + SENTIMENT),
    "Gradient boosting (all features)": (HAR + ["earn_next", "vol_ratio"] + SENTIMENT, "y_tr"),
}
FINAL_MODEL = "Log-HAR + earnings"


def volatility_results(df):
    test = df.iloc[FIRST_TEST_ROW:]
    y = test["y_tr"].to_numpy()
    baseline = test["tr_ma20"].to_numpy()
    no_earnings = test["earn_next"].to_numpy() == 0
    # a day counts as "high volatility" if it beats the 75th percentile of everything seen so far
    high_vol = np.array([y[i] > np.percentile(df["y_tr"].iloc[:t], 75) for i, t in enumerate(range(FIRST_TEST_ROW, len(df)))])

    preds = {}
    for name, spec in VOL_MODELS.items():
        if name.startswith("20-day"):
            preds[name] = baseline
        elif name.startswith("Yesterday"):
            preds[name] = test["tr"].to_numpy()
        else:
            cols, target = spec
            make = (lambda: HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=200,
                                                          random_state=SEED)) if "boosting" in name else LinearRegression
            p = walk_forward(df, cols, make, target)
            preds[name] = np.exp(p) if target == "y_log_tr" else p

    def skill(pred, mask):
        return 1 - np.sum((y[mask] - pred[mask]) ** 2) / np.sum((y[mask] - baseline[mask]) ** 2)

    every_day = np.ones(len(y), bool)
    rows = []
    for name, pred in preds.items():
        mae = mean_absolute_error(y, pred)
        rows.append({
            "model": name,
            "mae": mae,
            "mae_gain": 1 - mae / mean_absolute_error(y, baseline),
            "skill": skill(pred, every_day),
            "skill_no_earnings": skill(pred, no_earnings),
            "corr": np.corrcoef(y, pred)[0, 1],
            "auc_high_vol": roc_auc_score(high_vol, pred),
        })
    return rows, preds, test
