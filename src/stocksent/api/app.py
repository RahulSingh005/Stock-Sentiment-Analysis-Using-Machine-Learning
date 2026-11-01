"""FastAPI service: headline sentiment and a pre-open "beats the market today" probability for one stock.

Run with `stocksent serve` (or uvicorn stocksent.api.app:app). Prices can be sent in the request; if they are
omitted the service downloads the last few months from Yahoo Finance.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .. import __version__
from ..config import ROOT
from ..inference import Predictor, headlines_frame, prices_frame


class Headline(BaseModel):
    title: str = Field(min_length=3)
    published_at: str = Field(description="ISO timestamp (UTC if no offset is given), or a bare YYYY-MM-DD date "
                                          "when the time of day is unknown")

    @field_validator("published_at")
    @classmethod
    def _parseable(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("published_at must be an ISO date or timestamp") from exc
        return value


class PriceBar(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    headlines: list[Headline] = Field(default_factory=list, max_length=2000)
    prices: dict[str, list[PriceBar]] | None = Field(
        default=None, description="bars for the ticker, the benchmark (SPY) and ^VIX; the last date is predicted")


class SentimentRequest(BaseModel):
    headlines: list[str] = Field(min_length=1, max_length=500)


app = FastAPI(title="Stock news signal", version=__version__,
              description="Predicts whether a stock beats the market from open to close, using company news.")


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor.load(Path(os.environ.get("STOCKSENT_MODELS_DIR", ROOT / "models")))


@app.get("/health")
def health() -> dict:
    p = get_predictor()
    return {"status": "ok", "version": __version__, "model": p.bundle["model"],
            "trained_through": p.bundle["trained_through"]}


@app.post("/sentiment")
def sentiment(req: SentimentRequest) -> list[dict]:
    scores = get_predictor().sentiment.score(req.headlines)
    return [{"title": t, "bullish": round(float(s.fs_pos), 4), "bearish": round(float(s.fs_neg), 4),
             "neutral": round(float(s.fs_neu), 4)} for t, s in zip(req.headlines, scores.itertuples(), strict=True)]


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    predictor = get_predictor()
    ticker = req.ticker.upper()
    if req.prices is not None:
        prices = prices_frame({k: [b.model_dump() for b in v] for k, v in req.prices.items()})
    else:
        from ..data.prices import recent_prices

        prices = recent_prices([ticker, predictor.bundle["benchmark"], predictor.bundle["volatility_index"]])
    try:
        return predictor.predict(ticker, headlines_frame([h.model_dump() for h in req.headlines]), prices)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
