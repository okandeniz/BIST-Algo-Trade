"""Health and system status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.dependencies import (
    get_market_service,
    get_release_manifest,
    get_repository,
)
from src.product_config import public_trading_config


router = APIRouter(tags=["System"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()

    try:
        market = get_market_service().status()
    except Exception as error:
        market = {
            "data_file_exists": False,
            "data_file": None,
            "data_source": "error",
            "latest_date": None,
            "ticker_count": 0,
            "error": str(error),
        }

    try:
        portfolios = get_repository().list_portfolios()
        database_error = None
    except Exception as error:
        portfolios = []
        database_error = str(error)

    try:
        release = get_release_manifest().status()
    except Exception as error:
        release = {
            "ready": False,
            "release_id": None,
            "artifacts": {},
            "error": str(error),
        }

    missing_artifacts = [
        key
        for key, artifact in release.get("artifacts", {}).items()
        if not artifact.get("exists", False)
    ]
    data_source = market.get("data_source")
    daily_plan_exists = (
        settings.daily_plans_dir.exists()
        and any(settings.daily_plans_dir.iterdir())
    )

    checks = [
        {
            "key": "database",
            "label": "Portföy veritabanı",
            "status": "ready" if database_error is None else "error",
            "detail": (
                "Hazır."
                if database_error is None
                else f"Veritabanı açılamadı: {database_error}"
            ),
            "path": str(settings.database_path),
        },
        {
            "key": "ticker_file",
            "label": "BIST100 ticker listesi",
            "status": (
                "ready" if settings.ticker_file.exists() else "error"
            ),
            "detail": (
                "Hazır."
                if settings.ticker_file.exists()
                else "Veri yenileme için bu dosya gerekli."
            ),
            "path": str(settings.ticker_file),
        },
        {
            "key": "release",
            "label": "Rapor ve model artifact'leri",
            "status": "ready" if release.get("ready") else "error",
            "detail": (
                f"{release.get('release_id')} hazır."
                if release.get("ready")
                else (
                    f"{len(missing_artifacts)} artifact eksik."
                    if missing_artifacts
                    else release.get("error", "Release hazır değil.")
                )
            ),
            "path": str(settings.release_manifest_path),
        },
        {
            "key": "market_data",
            "label": "Piyasa verisi",
            "status": (
                "ready"
                if data_source == "live"
                else "warning"
                if data_source == "research_fallback"
                else "error"
            ),
            "detail": (
                "Güncel uygulama verisi kullanılıyor."
                if data_source == "live"
                else "Araştırma verisi kullanılıyor; sinyalleri yenileyebilirsiniz."
                if data_source == "research_fallback"
                else market.get("error", "Kullanılabilir piyasa verisi yok.")
            ),
            "path": market.get("data_file"),
        },
        {
            "key": "daily_plan",
            "label": "Günlük işlem planı",
            "status": "ready" if daily_plan_exists else "warning",
            "detail": (
                "Hazır."
                if daily_plan_exists
                else "Henüz oluşturulmadı; ilk sinyal yenilemesinde hazırlanır."
            ),
            "path": str(settings.daily_plans_dir),
        },
        {
            "key": "environment",
            "label": "Kişisel ortam ayarları",
            "status": (
                "ready"
                if (settings.project_root / ".env").exists()
                else "info"
            ),
            "detail": (
                ".env yüklendi."
                if (settings.project_root / ".env").exists()
                else "Varsayılan ayarlar kullanılıyor; .env isteğe bağlıdır."
            ),
            "path": str(settings.project_root / ".env"),
        },
    ]
    blocking_checks = [
        check for check in checks if check["status"] == "error"
    ]

    return {
        "status": "ok",
        "project_root": str(settings.project_root),
        "database_path": str(settings.database_path),
        "market_data": market,
        "portfolios": portfolios,
        "trading_config": public_trading_config(),
        "release": release,
        "setup": {
            "ready": not blocking_checks,
            "checks": checks,
            "blocking_count": len(blocking_checks),
            "missing_artifacts": missing_artifacts,
        },
    }
