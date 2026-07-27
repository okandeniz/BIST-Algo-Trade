"""Small serialization and normalization utilities."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd


def normalize_ticker(value: str) -> str:
    ticker = str(value).strip().upper()

    if not ticker:
        raise ValueError("Hisse kodu boş olamaz.")

    return ticker if ticker.endswith(".IS") else f"{ticker}.IS"


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into strict JSON-compatible objects."""
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, pd.DataFrame):
        return dataframe_records(value)

    if isinstance(value, pd.Series):
        return [json_safe(item) for item in value.tolist()]

    return str(value)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    records = frame.to_dict(orient="records")
    return [
        {
            str(key): json_safe(value)
            for key, value in record.items()
        }
        for record in records
    ]
