"""Classification and ranking metrics, with uncertainty estimates that respect the time structure."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p > threshold).astype(int)
    acc = accuracy_score(y, pred)
    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "auc": roc_auc_score(y, p),
        "mcc": matthews_corrcoef(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "p_value_vs_50": binomtest(int(round(acc * len(y))), len(y), 0.5, alternative="greater").pvalue,
    }


def auc_ci(y: np.ndarray, p: np.ndarray, dates: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval for AUC from a bootstrap that resamples whole days (stocks on a day move together)."""
    rng = np.random.default_rng(seed)
    days = np.unique(dates)
    by_day = {d: np.flatnonzero(dates == d) for d in days}
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([by_day[d] for d in rng.choice(days, size=len(days), replace=True)])
        if len(np.unique(y[idx])) == 2:
            stats.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def information_coefficient(df: pd.DataFrame, pred_col: str = "proba", ret_col: str = "fwd_ret") -> dict:
    """Daily cross-sectional rank correlation between predictions and next-day returns."""
    daily = df.groupby("date").apply(
        lambda g: spearmanr(g[pred_col], g[ret_col]).statistic if g[pred_col].nunique() > 1 else np.nan,
        include_groups=False,
    ).dropna()
    mean, std = daily.mean(), daily.std()
    return {"ic_mean": mean, "ic_std": std, "ic_t_stat": mean / std * np.sqrt(len(daily)),
            "ic_positive_days": (daily > 0).mean(), "ic_days": len(daily)}


def confidence_curve(y: np.ndarray, p: np.ndarray, coverages=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Accuracy when only the most confident predictions are acted on."""
    order = np.argsort(-np.abs(p - 0.5))
    rows = []
    for c in coverages:
        k = max(1, int(round(c * len(p))))
        idx = order[:k]
        rows.append({"coverage": c, "n": k, "accuracy": accuracy_score(y[idx], (p[idx] > 0.5).astype(int))})
    return pd.DataFrame(rows)
