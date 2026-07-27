"""Read walk-forward ML comparison artifacts for the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import pandas as pd

from backend.app.utils import dataframe_records


class WalkForwardService:
    def __init__(
        self,
        results_directory: str | Path,
    ):
        self.results_directory = Path(
            results_directory
        )

    def _path(self, filename: str) -> Path:
        path = self.results_directory / filename

        if not path.exists():
            raise FileNotFoundError(
                "Walk-forward sonucu bulunamadı: "
                f"{path}. Önce "
                "15_walk_forward_ml_comparison_robot.ipynb "
                "notebook'unu çalıştır."
            )

        return path

    def summary(self) -> dict[str, Any]:
        metadata = json.loads(
            self._path(
                "walk_forward_metadata.json"
            ).read_text(
                encoding="utf-8"
            )
        )
        metrics = pd.read_csv(
            self._path(
                "walk_forward_comparison_metrics.csv"
            )
        )
        active = pd.read_csv(
            self._path(
                "walk_forward_active_metrics.csv"
            )
        )

        return {
            "metadata": metadata,
            "metrics": dataframe_records(metrics),
            "active_metrics": dataframe_records(active),
        }

    def equity(self) -> list[dict[str, Any]]:
        frame = pd.read_parquet(
            self._path(
                "walk_forward_equity.parquet"
            )
        )
        return dataframe_records(frame)

    def yearly(self) -> list[dict[str, Any]]:
        frame = pd.read_csv(
            self._path(
                "walk_forward_yearly.csv"
            )
        )
        return dataframe_records(frame)

    def training_log(self) -> list[dict[str, Any]]:
        frame = pd.read_csv(
            self._path(
                "walk_forward_training_log.csv"
            )
        )
        return dataframe_records(frame)
