"""LSTM over each ticker's last L sessions of features, tuned with Optuna.

Sequences may reach back into earlier periods for history (that data was known at the time), but the row being
predicted always belongs to the split being evaluated. Scaling statistics come from the training period only.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from .classical import FitResult

log = logging.getLogger(__name__)


def _torch():
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return torch


def build_sequences(df: pd.DataFrame, features: list[str], seq_len: int, mean: np.ndarray, std: np.ndarray):
    """Return X (n, seq_len, f), y, split labels and the original row index of each predicted row."""
    xs, ys, splits, rows = [], [], [], []
    for _, g in df.sort_values("date").groupby("ticker", sort=False):
        arr = np.clip((g[features].to_numpy(np.float32) - mean) / std, -5, 5)
        arr = np.nan_to_num(arr)
        for i in range(seq_len - 1, len(g)):
            xs.append(arr[i - seq_len + 1:i + 1])
        ys.append(g["y"].to_numpy()[seq_len - 1:])
        splits.append(g["split"].to_numpy()[seq_len - 1:])
        rows.append(g.index.to_numpy()[seq_len - 1:])
    return np.stack(xs), np.concatenate(ys), np.concatenate(splits), np.concatenate(rows)


def _model(n_features: int, hidden: int, layers: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    return LSTMClassifier()


def _train(params, x_tr, y_tr, x_ev=None, y_ev=None, epochs=30, patience=5, seed=42):
    """Train; with an evaluation set, early-stop on its AUC and return (model, best_auc, best_epoch)."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _model(x_tr.shape[2], params["hidden"], params["layers"], params["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype(np.float32))
    best_auc, best_epoch, best_state, stale = -1.0, epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if x_ev is None:
            continue
        auc = roc_auc_score(y_ev, _predict(model, x_ev))
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def _predict(model, x):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x))).numpy()


def tune_and_fit_lstm(df: pd.DataFrame, features: list[str], feature_set: str, n_trials: int, seed: int) -> FitResult:
    train_rows = df["split"] == "train"
    mean = df.loc[train_rows, features].mean().to_numpy(np.float32)
    std = df.loc[train_rows, features].std().replace(0, 1).to_numpy(np.float32)
    cache = {}

    def data(seq_len):
        if seq_len not in cache:
            cache[seq_len] = build_sequences(df, features, seq_len, mean, std)
        return cache[seq_len]

    def objective(trial):
        params = {
            "seq_len": trial.suggest_categorical("seq_len", [5, 10, 20]),
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "layers": trial.suggest_int("layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
        x, y, split, _ = data(params["seq_len"])
        _, auc, epoch = _train(params, x[split == "train"], y[split == "train"], x[split == "valid"],
                               y[split == "valid"], seed=seed)
        trial.set_user_attr("best_epoch", epoch)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    params, epochs = study.best_params, study.best_trial.user_attrs["best_epoch"]
    log.info("LSTM best valid AUC %.4f with %s (epochs %d)", study.best_value, params, epochs)

    x, y, split, rows = data(params["seq_len"])
    tr, va, te = split == "train", split == "valid", split == "test"
    on_train, _, _ = _train(params, x[tr], y[tr], x[va], y[va], seed=seed)
    valid_proba = _predict(on_train, x[va])
    final, _, _ = _train(params, x[tr | va], y[tr | va], epochs=epochs, seed=seed)
    test_proba = _predict(final, x[te])

    # align to the dataframe's valid/test rows (rows without enough history keep the training base rate)
    base = float(df.loc[train_rows, "y"].mean())
    valid_full = pd.Series(base, index=df.index[df["split"] == "valid"])
    test_full = pd.Series(base, index=df.index[df["split"] == "test"])
    valid_full.loc[rows[va]] = valid_proba
    test_full.loc[rows[te]] = test_proba
    return FitResult("LSTM (deep learning)", feature_set, {**params, "epochs": epochs}, study.best_value,
                     valid_full.to_numpy(), test_full.to_numpy(), final)
