"""Figures used in the README."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from .evaluate import FINAL_MODEL  # noqa: E402

BLUE, TEAL, GRAY, ORANGE, INK = "#2563eb", "#0d9488", "#9ca3af", "#ea580c", "#1f2937"
TICKER_COLOR = {"GOOGL": BLUE, "MSFT": TEAL}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d1d5db", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
})


def leakage_vs_reality(results, path):
    labels = ["Original setup, reproduced\n(label leaks)", "Same model,\nhonest test", "Majority-class\nbaseline"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.36
    for i, (ticker, r) in enumerate(results.items()):
        rows = {(f, m): acc for f, m, acc, _ in r["direction"][0]}
        values = [r["audit"]["circular_accuracy"], rows[("All-news sentiment", "Logistic")], rows[("Majority class", "-")]]
        x = np.arange(3) + (i - 0.5) * width
        bars = ax.bar(x, [v * 100 for v in values], width, color=TICKER_COLOR[ticker], label=ticker)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=9)
    ax.axhline(50, color=GRAY, ls="--", lw=1)
    ax.text(2.62, 51, "coin flip", color="#6b7280", fontsize=8, ha="right")
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Next-day accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("The high accuracy came from leakage; an honest test lands at chance")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def sentiment_vs_returns(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (ticker, r) in zip(axes, results.items()):
        df = r["df"]
        rho, p = spearmanr(df["sent_mean_z"], df["ret_next"])
        ax.scatter(df["sent_mean_z"], df["ret_next"] * 100, s=12, alpha=0.6, color=TICKER_COLOR[ticker], lw=0)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_title(f"{ticker}: Spearman rho = {rho:+.2f} (p = {p:.2f})")
        ax.set_xlabel("News sentiment, z-score vs prior 20 days")
    axes[0].set_ylabel("Next-day return (%)")
    fig.suptitle("Daily news sentiment says little about tomorrow's direction", fontweight="bold", y=1.03)
    fig.savefig(path)
    plt.close(fig)


def volatility_forecast(results, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for ax, (ticker, r) in zip(axes, results.items()):
        _, preds, test = r["volatility"]
        dates = test["target_date"]
        ax.plot(dates, test["y_tr"], color=GRAY, lw=1, marker="o", ms=2.5, label="Actual")
        ax.plot(dates, preds["20-day average (baseline)"], color=INK, lw=1, ls="--", label="20-day average")
        ax.plot(dates, preds[FINAL_MODEL], color=TICKER_COLOR[ticker], lw=1.8, label=FINAL_MODEL)
        for d in dates[test["earn_next"] == 1]:
            ax.axvline(d, color=ORANGE, lw=1, ls=":")
            ax.text(d, ax.get_ylim()[1] * 0.92, " earnings", color=ORANGE, fontsize=8)
        ax.set_title(ticker, loc="left")
        ax.set_ylabel("True range (%)")
    axes[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.0), fontsize=9, borderaxespad=0.2)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.15)
    fig.suptitle("Next-day volatility: walk-forward forecasts vs actual", fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def volatility_models(results, path):
    names = [row["model"] for row in next(iter(results.values()))["volatility"][0]][1:]  # skip the reference
    fig, ax = plt.subplots(figsize=(8, 3.8))
    height = 0.38
    y = np.arange(len(names))
    for i, (ticker, r) in enumerate(results.items()):
        gains = {row["model"]: row["mae_gain"] * 100 for row in r["volatility"][0]}
        bars = ax.barh(y + (i - 0.5) * height, [gains[n] for n in names], height, color=TICKER_COLOR[ticker],
                       label=ticker)
        ax.bar_label(bars, fmt="%+.1f%%", padding=3, fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("MAE improvement vs 20-day average (%)")
    ax.set_title("Recent volatility + earnings calendar beat the baseline; sentiment adds nothing")
    ax.legend(loc="lower right")
    ax.margins(x=0.15)
    fig.savefig(path)
    plt.close(fig)


def make_all(results, out_dir):
    out_dir.mkdir(exist_ok=True)
    leakage_vs_reality(results, out_dir / "leakage_vs_reality.png")
    sentiment_vs_returns(results, out_dir / "sentiment_vs_returns.png")
    volatility_forecast(results, out_dir / "volatility_forecast.png")
    volatility_models(results, out_dir / "volatility_models.png")
