"""End-to-end pipeline: ingest -> features -> train/evaluate -> report."""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Config
from .data.news import clean_news, download_news
from .data.prices import download_prices, trading_sessions
from .data.sessions import assign_sessions
from .evaluation.backtest import always_long, long_short, long_top, performance
from .evaluation.metrics import auc_ci, classification_metrics, confidence_curve, information_coefficient
from .features.dataset import FEATURE_SETS, assemble
from .features.market import build_market_features
from .features.news_features import build_news_features, daily_text, score_headlines
from .features.sentiment import FinanceSentimentModel
from .models.classical import tune_and_fit

log = logging.getLogger(__name__)

CLASSICAL = ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM"]


def sentiment_model(cfg: Config) -> FinanceSentimentModel:
    path = cfg.models_dir / "finance_sentiment.joblib"
    if path.exists():
        return FinanceSentimentModel.load(path)
    model = FinanceSentimentModel.train(cfg.raw_dir / "sentiment", seed=cfg.seed)
    model.save(path)
    return model


def ingest(cfg: Config) -> None:
    download_news(cfg)
    download_prices(cfg)


def build_features(cfg: Config) -> pd.DataFrame:
    prices = download_prices(cfg)
    sessions = trading_sessions(prices, cfg)
    news = clean_news(download_news(cfg), cfg)
    news["session"] = assign_sessions(news["published"], sessions, news["date_only"])
    scored = score_headlines(news, sentiment_model(cfg))
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(cfg.processed_dir / "headlines.parquet", index=False)
    news_features = build_news_features(scored, sessions, cfg.tickers)
    market = build_market_features(prices, cfg.tickers, cfg.benchmark, cfg.volatility_index)
    data = assemble(market, news_features, daily_text(scored), cfg)
    data.to_parquet(cfg.processed_dir / "dataset.parquet", index=False)
    log.info("dataset: %d rows, %d headlines", len(data), len(scored))
    return data


def _evaluate(name: str, feature_set: str, test: pd.DataFrame, test_proba, cfg: Config) -> dict:
    y, p = test["y"].to_numpy(), np.asarray(test_proba)
    frame = test[["date", "ticker", "fwd_ret"]].assign(proba=p)
    return {
        "model": name,
        "feature_set": feature_set,
        "test": classification_metrics(y, p),
        "auc_ci": auc_ci(y, p, test["date"].to_numpy(), seed=cfg.seed),
        "ic": information_coefficient(frame),
        "confidence": confidence_curve(y, p).to_dict("records"),
        "long_short": performance(long_short(frame, cost_bps=cfg.cost_bps)),
        "long_short_gross": performance(long_short(frame, cost_bps=0)),
        "long_top": performance(long_top(frame, cost_bps=cfg.cost_bps)),
    }


def train(cfg: Config) -> dict:
    data = pd.read_parquet(cfg.processed_dir / "dataset.parquet")
    split = {s: data[data["split"] == s] for s in ("train", "valid", "test")}
    tr, va, te = split["train"], split["valid"], split["test"]

    fits = []
    for feature_set in ("price", "news", "price+news"):
        for name in CLASSICAL:
            fits.append(tune_and_fit(name, feature_set, FEATURE_SETS[feature_set], tr, va, te, cfg.n_trials, cfg.seed))
    fits.append(tune_and_fit("Logistic Regression", "vader", FEATURE_SETS["vader"], tr, va, te, cfg.n_trials, cfg.seed))
    fits.append(tune_and_fit("TF-IDF + LogReg (headline text)", "text", "text", tr, va, te, cfg.n_trials, cfg.seed))

    from .models.lstm import tune_and_fit_lstm

    fits.append(tune_and_fit_lstm(data, FEATURE_SETS["price+news"], "price+news", cfg.lstm_trials, cfg.seed))

    # ensemble of the three best price+news models, chosen on validation AUC
    pool = sorted([f for f in fits if f.feature_set == "price+news"], key=lambda f: f.valid_auc, reverse=True)[:3]
    ens_valid = np.mean([f.valid_proba for f in pool], axis=0)
    ens_test = np.mean([f.test_proba for f in pool], axis=0)

    preds = te[["date", "ticker", "y", "y_raw", "fwd_ret", "fwd_excess", "gap", "n_news"]].copy()
    results = []
    for f in fits:
        preds[f"{f.model} | {f.feature_set}"] = f.test_proba
        row = _evaluate(f.model, f.feature_set, te, f.test_proba, cfg)
        row.update(valid_auc=f.valid_auc, best_params=f.best_params)
        results.append(row)
    ensemble_name = "Ensemble (top 3 by validation AUC)"
    preds[f"{ensemble_name} | price+news"] = ens_test
    row = _evaluate(ensemble_name, "price+news", te, ens_test, cfg)
    row.update(valid_auc=roc_auc_score(va["y"], ens_valid), best_params={"members": [f.model for f in pool]})
    results.append(row)

    # simple rules every model has to beat
    majority = float(tr["y"].mean() > 0.5)
    rules = {"Gap reversal rule (minus opening gap)": -te["gap"],
             "Sentiment rule (mean net sentiment)": te["fs_net_mean"],
             "Momentum rule (yesterday's return)": te["ret_1"]}
    baselines = {"Majority class from training": classification_metrics(te["y"].to_numpy(), np.full(len(te), majority))}
    for name, score in rules.items():
        frame = te[["date", "ticker", "fwd_ret"]].assign(proba=score.to_numpy())
        baselines[name] = {"auc": roc_auc_score(te["y"], score), "ic": information_coefficient(frame),
                           "long_short": performance(long_short(frame, cost_bps=cfg.cost_bps))}
    bench = performance(always_long(te[["date", "ticker", "fwd_ret"]], cost_bps=cfg.cost_bps))

    # production model for the API: best classical price+news model on validation
    best = max((f for f in fits if f.feature_set == "price+news" and f.model in CLASSICAL), key=lambda f: f.valid_auc)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": best.estimator, "model": best.model, "features": FEATURE_SETS["price+news"],
                 "trained_through": cfg.valid_end, "tickers": cfg.tickers, "benchmark": cfg.benchmark,
                 "volatility_index": cfg.volatility_index}, cfg.models_dir / "production_model.joblib")

    metrics = {
        "data": {
            "rows": {s: int(len(d)) for s, d in split.items()},
            "periods": {s: [str(d["date"].min().date()), str(d["date"].max().date())] for s, d in split.items()},
            "tickers": cfg.tickers,
            "test_up_rate": float(te["y"].mean()),
        },
        "sentiment_model": sentiment_model(cfg).report,
        "models": results,
        "baselines": baselines,
        "buy_and_hold": bench,
        "production_model": best.model,
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    preds.to_parquet(cfg.processed_dir / "test_predictions.parquet", index=False)
    return metrics
