"""Release manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.release_manifest import ReleaseManifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseManifestTests(unittest.TestCase):
    def test_promoted_release_is_complete(self) -> None:
        release = ReleaseManifest(PROJECT_ROOT)
        status = release.status()

        self.assertEqual(status["schema_version"], 1)
        self.assertEqual(status["primary_strategy"], "Baseline_Robot")
        self.assertTrue(status["ready"])
        self.assertGreaterEqual(len(status["artifacts"]), 18)

    def test_logical_keys_resolve_inside_project(self) -> None:
        release = ReleaseManifest(PROJECT_ROOT)

        for key in (
            "baseline.periods",
            "walk_forward.equity",
            "challenger.model",
        ):
            path = release.resolve(key)
            self.assertTrue(path.is_relative_to(PROJECT_ROOT))
            self.assertTrue(path.exists())

    def test_manifest_rejects_project_escape(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "release.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "release_id": "invalid",
                        "artifacts": {"escape": "../outside.txt"},
                    }
                ),
                encoding="utf-8",
            )
            release = ReleaseManifest(root, manifest_path)

            with self.assertRaises(ValueError):
                release.resolve("escape", must_exist=False)


if __name__ == "__main__":
    unittest.main()
