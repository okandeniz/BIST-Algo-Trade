"""System readiness must remain informative even during first setup."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.app.dependencies import get_market_service
from backend.app.config import get_settings
from backend.app.routers.health import health


class SetupStatusTests(unittest.TestCase):
    def test_market_service_prefers_live_data_with_research_fallback(self) -> None:
        settings = get_settings()
        service = get_market_service()

        self.assertEqual(service.live_path, settings.live_stock_path)
        self.assertEqual(service.fallback_path, settings.processed_stock_path)

    def test_health_exposes_guided_setup_checks(self) -> None:
        payload = health()
        checks = {
            check["key"]: check
            for check in payload["setup"]["checks"]
        }

        self.assertEqual(payload["status"], "ok")
        self.assertIn("ready", payload["setup"])
        self.assertEqual(
            set(checks),
            {
                "database",
                "ticker_file",
                "release",
                "market_data",
                "daily_plan",
                "environment",
            },
        )

    def test_missing_manifest_does_not_break_health_endpoint(self) -> None:
        with patch(
            "backend.app.routers.health.get_release_manifest",
            side_effect=FileNotFoundError("manifest yok"),
        ):
            payload = health()

        release_check = next(
            check
            for check in payload["setup"]["checks"]
            if check["key"] == "release"
        )
        self.assertEqual(payload["status"], "ok")
        self.assertFalse(payload["setup"]["ready"])
        self.assertEqual(release_check["status"], "error")
        self.assertIn("manifest yok", release_check["detail"])


if __name__ == "__main__":
    unittest.main()
