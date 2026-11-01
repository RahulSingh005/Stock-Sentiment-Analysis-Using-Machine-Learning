"""Project configuration, loaded from configs/*.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Config:
    tickers: list[str]
    coverage_start: dict[str, str]
    company_names: dict[str, str]
    benchmark: str
    volatility_index: str
    start: str
    end: str
    train_end: str
    valid_end: str
    news_dataset: str
    news_files: list[str]
    news_symbol_aliases: dict[str, str]
    cost_bps: float
    n_trials: int
    lstm_trials: int
    seed: int
    data_dir: Path
    reports_dir: Path
    models_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


def load_config(path: str | Path | None = None) -> Config:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    paths = {k: ROOT / raw["paths"][k] for k in ("data_dir", "reports_dir", "models_dir")}
    return Config(
        tickers=list(raw["universe"]["tickers"]),
        coverage_start={k: str(v) for k, v in (raw["universe"].get("coverage_start") or {}).items()},
        company_names=dict(raw["universe"]["company_names"]),
        benchmark=raw["universe"]["benchmark"],
        volatility_index=raw["universe"]["volatility_index"],
        start=str(raw["period"]["start"]),
        end=str(raw["period"]["end"]),
        train_end=str(raw["period"]["train_end"]),
        valid_end=str(raw["period"]["valid_end"]),
        news_dataset=raw["news"]["dataset"],
        news_files=list(raw["news"]["files"]),
        news_symbol_aliases=dict(raw["news"].get("symbol_aliases") or {}),
        cost_bps=float(raw["backtest"]["cost_bps"]),
        n_trials=int(raw["tuning"]["n_trials"]),
        lstm_trials=int(raw["tuning"]["lstm_trials"]),
        seed=int(raw["tuning"]["seed"]),
        **paths,
    )
