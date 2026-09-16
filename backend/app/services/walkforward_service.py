"""Read walk-forward ML comparison artifacts for the API."""

from __future__ import annotations

from typing import Any

import json
import pandas as pd

from backend.app.utils import dataframe_records
from src.release_manifest import ReleaseManifest


class WalkForwardService:
    def __init__(
        self,
        release_manifest: ReleaseManifest,
    ):
        self.release_manifest = release_manifest

    def summary(self) -> dict[str, Any]:
        metadata = json.loads(
            self.release_manifest.resolve(
                "walk_forward.metadata"
            ).read_text(
                encoding="utf-8"
            )
        )
        metrics = pd.read_csv(
            self.release_manifest.resolve("walk_forward.metrics")
        )
        active = pd.read_csv(
            self.release_manifest.resolve(
                "walk_forward.active_metrics"
            )
        )

        return {
            "metadata": metadata,
            "metrics": dataframe_records(metrics),
            "active_metrics": dataframe_records(active),
        }

    def equity(self) -> list[dict[str, Any]]:
        frame = pd.read_parquet(
            self.release_manifest.resolve("walk_forward.equity")
        )
        return dataframe_records(frame)

    def yearly(self) -> list[dict[str, Any]]:
        frame = pd.read_csv(
            self.release_manifest.resolve("walk_forward.yearly")
        )
        return dataframe_records(frame)

    def training_log(self) -> list[dict[str, Any]]:
        frame = pd.read_csv(
            self.release_manifest.resolve("walk_forward.training_log")
        )
        return dataframe_records(frame)
