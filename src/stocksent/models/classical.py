"""Classical models and their Optuna search spaces.

Every model is tuned on the validation period only (ROC AUC), refit on train + validation with the best
parameters, and scored once on the test period.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_JOBS = 4  # threads for tree ensembles; kept modest so the pipeline runs on a small laptop


def _logreg(trial, seed):
    c = trial.suggest_float("C", 1e-4, 10, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, penalty=penalty, solver="liblinear", max_iter=2000))


def _tree(trial, seed):
    return DecisionTreeClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 600, log=True),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        random_state=seed,
    )


def _forest(trial, seed):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 14),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
        max_features=trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
        n_jobs=N_JOBS,
        random_state=seed,
    )


def _lightgbm(trial, seed):
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
    )


def _tfidf(trial, seed):
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, trial.suggest_int("max_ngram", 1, 2)),
            min_df=trial.suggest_int("min_df", 3, 30, log=True),
            max_features=30000,
            sublinear_tf=True,
            stop_words="english",
        ),
        LogisticRegression(C=trial.suggest_float("C", 1e-3, 10, log=True), max_iter=2000),
    )


MODELS = {
    "Logistic Regression": _logreg,
    "Decision Tree": _tree,
    "Random Forest": _forest,
    "LightGBM": _lightgbm,
    "TF-IDF + LogReg (headline text)": _tfidf,
}


@dataclass
class FitResult:
    model: str
    feature_set: str
    best_params: dict
    valid_auc: float
    valid_proba: np.ndarray
    test_proba: np.ndarray
    estimator: object = field(repr=False)


def tune_and_fit(name: str, feature_set: str, features, train: pd.DataFrame, valid: pd.DataFrame,
                 test: pd.DataFrame, n_trials: int, seed: int) -> FitResult:
    builder = MODELS[name]
    cols = features if isinstance(features, str) else list(features)

    def objective(trial):
        est = builder(trial, seed).fit(train[cols], train["y"])
        return roc_auc_score(valid["y"], est.predict_proba(valid[cols])[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = optuna.trial.FixedTrial(study.best_params)

    on_train = builder(best, seed).fit(train[cols], train["y"])
    valid_proba = on_train.predict_proba(valid[cols])[:, 1]
    both = pd.concat([train, valid])
    final = builder(optuna.trial.FixedTrial(study.best_params), seed).fit(both[cols], both["y"])
    log.info("%s [%s] valid AUC %.4f", name, feature_set, study.best_value)
    return FitResult(name, feature_set, study.best_params, study.best_value, valid_proba,
                     final.predict_proba(test[cols])[:, 1], final)
