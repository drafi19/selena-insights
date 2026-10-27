from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, Field


class Txn(BaseModel):
    date: Date
    amount: float = Field(gt=0, le=1_000_000_000)
    category: str | None = None


class ScoreRequest(BaseModel):
    transactions: list[Txn] = Field(min_length=1, max_length=500)
    detector: Literal["zscore", "autoencoder"] | None = None


class ScoredTxn(BaseModel):
    date: Date
    amount: float
    category: str | None = None
    score: float
    is_anomaly: bool


class ScoreResponse(BaseModel):
    model_version: str
    detector: str
    threshold: float
    flagged: int
    results: list[ScoredTxn]
