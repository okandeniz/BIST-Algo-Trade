"""Health and system status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.dependencies import (
    get_market_service,
    get_repository,
)


router = APIRouter(tags=["System"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    market = get_market_service().status()

    return {
        "status": "ok",
        "project_root": str(settings.project_root),
        "database_path": str(settings.database_path),
        "market_data": market,
        "portfolios": get_repository().list_portfolios(),
    }
