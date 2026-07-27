"""Pydantic request schemas for trading operations."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


PortfolioName = Literal[
    "Baseline_Robot",
    "ML_Challenger",
]


class InitialCapitalRequest(BaseModel):
    """Set the starting cash of an empty portfolio."""

    initial_capital: float = Field(gt=0)


class ResetPortfolioRequest(BaseModel):
    """Delete all portfolio records and return to setup mode."""

    confirmation: Literal["RESET"]


class BuyRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    portfolio_name: PortfolioName
    ticker: str
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    trade_date: date
    fees: float = Field(default=0, ge=0)
    stop_loss: float = Field(gt=0)
    signal_score: int | None = Field(default=None, ge=0)
    note: str | None = Field(default=None, max_length=500)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not ticker:
            raise ValueError("Hisse kodu boş olamaz.")
        return ticker

    @model_validator(mode="after")
    def validate_stop(self) -> "BuyRequest":
        if self.stop_loss >= self.price:
            raise ValueError(
                "Uzun pozisyonda stop seviyesi alış fiyatından "
                "düşük olmalıdır."
            )
        return self


class SellRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    portfolio_name: PortfolioName
    ticker: str
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    trade_date: date
    fees: float = Field(default=0, ge=0)
    note: str | None = Field(default=None, max_length=500)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not ticker:
            raise ValueError("Hisse kodu boş olamaz.")
        return ticker
