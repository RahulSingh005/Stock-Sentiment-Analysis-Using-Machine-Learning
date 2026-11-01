"""Figures and a markdown summary built from reports/metrics.json and the saved test predictions.

The headline model is the one with the best validation AUC among the price+news models, so the test year is
never used to pick a winner.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from .config import Config  # noqa: E402
from .evaluation.backtest import long_short, performance  # noqa: E402

BLUE, TEAL, ORANGE, GRAY, INK = "#2563eb", "#0d9488", "#ea580c", "#9ca3af", "#1f2937"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d1d5db", "legend.frameon": False,
})


def _key(row) -> str:
    return f"{row['model']} | {row['feature_set']}"


def selected_model(metrics: dict) -> dict:
    pool = [m for m in metrics["models"] if m["feature_set"] == "price+news"]
    return max(pool, key=lambda m: m["valid_auc"])


def _model_comparison(metrics, path):
    rows = [m for m in metrics["models"] if m["feature_set"] in ("price+news", "text", "vader")]
    labels = [f"{m['model']}" + ("" if m["feature_set"] == "price+news" else f" [{m['feature_set']}]") for m in rows]
    auc = np.array([m["test"]["auc"] for m in rows])
    lo = auc - np.array([m["auc_ci"][0] for m in rows])
    hi = np.array([m["auc_ci"][1] for m in rows]) - auc
    for name, b in metrics["baselines"].items():
        if "Majority" not in name:
            labels.append(name)
            auc, lo, hi = np.append(auc, b["auc"]), np.append(lo, 0), np.append(hi, 0)
    order = np.argsort(auc)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1))
    colors = [GRAY if "rule" in labels[i] else BLUE for i in order]
    ax.barh(range(len(order)), auc[order] - 0.5, left=0.5, xerr=[lo[order], hi[order]], color=colors,
            error_kw={"ecolor": INK, "lw": 0.8, "capsize": 2})
    ax.set_yticks(range(len(order)), [labels[i] for i in order])
    ax.axvline(0.5, color=INK, lw=0.8)
    ax.set_xlabel("Test ROC AUC, 2023 (95% CI from a day-level bootstrap)")
    ax.set_title("Will the stock beat the market today? Out-of-sample AUC by model")
    fig.savefig(path)
    plt.close(fig)


def _ablation(metrics, path):
    models = ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM"]
    sets = ["price", "news", "price+news"]
    val = {(m["model"], m["feature_set"]): m["test"]["auc"] for m in metrics["models"]}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    width = 0.26
    for i, (fs, color) in enumerate(zip(sets, [GRAY, ORANGE, BLUE], strict=True)):
        x = np.arange(len(models)) + (i - 1) * width
        bars = ax.bar(x, [val[(m, fs)] - 0.5 for m in models], width, bottom=0.5, color=color, label=fs)
        ax.bar_label(bars, labels=[f"{val[(m, fs)]:.3f}" for m in models], padding=2, fontsize=7)
    ax.axhline(0.5, color=INK, lw=0.8)
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel("Test ROC AUC")
    ax.set_title("What carries the signal: price features, news features, or both")
    ax.legend(title="Features", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(path)
    plt.close(fig)


def _backtest(preds, sel_key, cfg, path):
    frame = preds[["date", "ticker", "fwd_ret"]].assign(proba=preds[sel_key])
    gap = preds[["date", "ticker", "fwd_ret"]].assign(proba=-preds["gap"])
    curves = {
        f"{sel_key.split(' | ')[0]}, net of costs": long_short(frame, cost_bps=cfg.cost_bps),
        f"{sel_key.split(' | ')[0]}, before costs": long_short(frame, cost_bps=0),
        "Gap reversal rule, net of costs": long_short(gap, cost_bps=cfg.cost_bps),
    }
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for (name, r), color, ls in zip(curves.items(), [BLUE, BLUE, GRAY], ["-", "--", "-"], strict=True):
        ax.plot((1 + r).cumprod().index, ((1 + r).cumprod() - 1) * 100, color=color, ls=ls, lw=1.6, label=name)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Market-neutral backtest, 2023: long top third, short bottom third, open to close")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def _confidence(sel, path):
    conf = pd.DataFrame(sel["confidence"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(conf["coverage"] * 100, conf["accuracy"] * 100, marker="o", color=BLUE)
    ax.axhline(50, color=GRAY, ls="--", lw=1)
    for _, r in conf.iterrows():
        ax.annotate(f"{r['accuracy']:.1%}", (r["coverage"] * 100, r["accuracy"] * 100), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Share of predictions acted on, most confident first (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by confidence: only the most confident calls stand out")
    fig.savefig(path)
    plt.close(fig)


def _monthly_ic(preds, sel_key, path):
    daily = preds.groupby("date").apply(lambda g: spearmanr(g[sel_key], g["fwd_ret"]).statistic, include_groups=False)
    monthly = daily.groupby(daily.index.to_period("M")).mean()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(monthly.index.astype(str), monthly.values, color=[BLUE if v > 0 else ORANGE for v in monthly.values])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Mean daily IC")
    ax.set_title("Information coefficient by month, 2023: positive overall, but unstable")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(path)
    plt.close(fig)


def reaction_table(data: pd.DataFrame) -> pd.DataFrame:
    """Market-adjusted returns on news days by sentiment: before the open (already priced) vs after it.

    'Before the open' runs from the close two sessions back to today's open, so it contains the whole news day;
    'after the open' is the open-to-close move the models try to predict.
    """
    news = data[data["n_news"] > 0].copy()
    news["before_open"] = (news["ret_1"] + news["gap"]) - (news["mkt_ret_1"] + news["mkt_gap"])
    news["bucket"] = pd.cut(news["fs_net_mean"], [-1.01, -0.2, 0.2, 1.01], labels=["Bearish", "Neutral", "Bullish"])
    out = news.groupby("bucket", observed=True).agg(before_open=("before_open", "mean"),
                                                    after_open=("fwd_excess", "mean"), days=("y", "size"))
    out[["before_open", "after_open"]] *= 1e4  # basis points
    return out


def _reaction(table: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(table))
    b1 = ax.bar(x - 0.2, table["before_open"], 0.4, color=BLUE, label="Before the open (already priced in)")
    b2 = ax.bar(x + 0.2, table["after_open"], 0.4, color=ORANGE, label="After the open (what a trader can capture)")
    ax.bar_label(b1, fmt="%+.0f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%+.0f", padding=2, fontsize=8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [f"{b} news\n({n:,} stock-days)" for b, n in zip(table.index, table["days"], strict=True)])
    ax.set_ylabel("Return vs market (bps)")
    ax.set_title("The sentiment model reads the news correctly, but the market prices it before the open")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_report(cfg: Config) -> None:
    metrics = json.loads((cfg.reports_dir / "metrics.json").read_text(encoding="utf-8"))
    preds = pd.read_parquet(cfg.processed_dir / "test_predictions.parquet")
    sel = selected_model(metrics)
    sel_key = _key(sel)
    fig = cfg.figures_dir
    fig.mkdir(parents=True, exist_ok=True)
    _model_comparison(metrics, fig / "model_comparison.png")
    _ablation(metrics, fig / "ablation.png")
    _backtest(preds, sel_key, cfg, fig / "backtest.png")
    _confidence(sel, fig / "confidence.png")
    _monthly_ic(preds, sel_key, fig / "monthly_ic.png")
    data = pd.read_parquet(cfg.processed_dir / "dataset.parquet")
    reaction = reaction_table(data)
    _reaction(reaction, fig / "reaction.png")
    news = data[data["n_news"] > 0]
    before = (news["ret_1"] + news["gap"]) - (news["mkt_ret_1"] + news["mkt_gap"])
    bull, bear = before[news["fs_net_mean"] > 0.2], before[news["fs_net_mean"] < -0.2]
    t_stat = (bull.mean() - bear.mean()) / np.sqrt(bull.var() / len(bull) + bear.var() / len(bear))

    frame = preds[["date", "ticker", "fwd_ret"]].assign(proba=preds[sel_key])
    costs = [(f"{c} bps", f"{performance(long_short(frame, cost_bps=c))['sharpe']:.2f}",
              f"{performance(long_short(frame, cost_bps=c))['annual_return']:.1%}") for c in (0, 1, 2, 5)]
    model_rows = []
    for m in sorted(metrics["models"], key=lambda m: -m["test"]["auc"]):
        t, ls = m["test"], m["long_short"]
        model_rows.append((m["model"], m["feature_set"], f"{m['valid_auc']:.3f}",
                           f"{t['auc']:.3f} [{m['auc_ci'][0]:.3f}, {m['auc_ci'][1]:.3f}]", f"{t['accuracy']:.1%}",
                           f"{t['mcc']:+.3f}", f"{m['ic']['ic_mean']:+.4f} ({m['ic']['ic_t_stat']:+.1f})",
                           f"{ls['sharpe']:+.2f}", f"{ls['max_drawdown']:.1%}"))
    base_rows = []
    for name, b in metrics["baselines"].items():
        if "ic" in b:
            base_rows.append((name, f"{b['auc']:.3f}", f"{b['ic']['ic_mean']:+.4f} ({b['ic']['ic_t_stat']:+.1f})",
                              f"{b['long_short']['sharpe']:+.2f}"))
        else:
            base_rows.append((name, f"{b['auc']:.3f} (accuracy {b['accuracy']:.1%})", "-", "-"))
    sm = metrics["sentiment_model"]
    d = metrics["data"]
    periods = [(s, d["rows"][s], f"{d['periods'][s][0]} to {d['periods'][s][1]}") for s in ("train", "valid", "test")]
    ls, ls_gross = sel["long_short"], sel["long_short_gross"]
    text = [
        "# Results", "",
        "Generated by `stocksent report`. Test period: 2023, never used for training, tuning or model selection.", "",
        "## Data", "",
        _table(periods, ["Split", "Stock-days", "Period"]), "",
        f"Tickers: {', '.join(d['tickers'])}. "
        f"Share of test stock-days that beat the market: {d['test_up_rate']:.1%}.", "",
        "## Finance sentiment model", "",
        f"TF-IDF + logistic regression on {sm['n_train']:,} labeled financial headlines; held-out accuracy "
        f"**{sm['valid_accuracy']:.1%}**, macro F1 **{sm['valid_macro_f1']:.3f}** "
        f"on {sm['n_valid']:,} examples (3 classes).", "",
        "## News is priced in before the open", "",
        "Market-adjusted return on news days (2017-2023), by the sentiment of that day's company headlines:", "",
        _table([(b, f"{r.before_open:+.1f} bps", f"{r.after_open:+.1f} bps", f"{int(r.days):,}")
                for b, r in reaction.iterrows()],
               ["Sentiment", "Before the open (already priced)", "After the open (tradeable)", "Stock-days"]), "",
        f"Bullish minus bearish, before the open: {bull.mean() * 1e4 - bear.mean() * 1e4:+.0f} bps (t = {t_stat:.1f}).",
        "",
        f"## Selected model: {sel['model']} ({sel['feature_set']})", "",
        "Chosen on validation AUC before looking at the test year.", "",
        _table([("Test AUC", f"{sel['test']['auc']:.3f} [{sel['auc_ci'][0]:.3f}, {sel['auc_ci'][1]:.3f}]"),
                ("Accuracy", f"{sel['test']['accuracy']:.1%} (p = {sel['test']['p_value_vs_50']:.3f} vs 50%)"),
                ("Accuracy on the 10% most confident calls", f"{sel['confidence'][0]['accuracy']:.1%}"),
                ("Daily IC (t-stat)", f"{sel['ic']['ic_mean']:+.4f} ({sel['ic']['ic_t_stat']:+.2f})"),
                ("Long-short Sharpe, net / gross", f"{ls['sharpe']:+.2f} / {ls_gross['sharpe']:+.2f}"),
                ("Long-short annual return, net", f"{ls['annual_return']:.1%}"),
                ("Long-short max drawdown, net", f"{ls['max_drawdown']:.1%}")], ["Metric", "Value"]),
        "", "Cost sensitivity (long-short, per-side cost):", "",
        _table(costs, ["Cost", "Sharpe", "Annual return"]), "",
        "## All models", "",
        _table(model_rows, ["Model", "Features", "Valid AUC", "Test AUC [95% CI]", "Accuracy", "MCC", "IC (t)",
                            "L/S Sharpe", "L/S max DD"]), "",
        "## Baselines", "",
        _table(base_rows, ["Rule", "Test AUC", "IC (t)", "L/S Sharpe"]), "",
        f"Always long every stock, open to close, after costs: Sharpe {metrics['buy_and_hold']['sharpe']:+.2f}, "
        f"annual return {metrics['buy_and_hold']['annual_return']:.1%}.",
    ]
    (cfg.reports_dir / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
