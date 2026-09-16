"""Versioned lookup for promoted application artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReleaseManifest:
    """Resolve logical artifact keys without coupling runtime to notebooks."""

    def __init__(
        self,
        project_root: str | Path,
        manifest_path: str | Path | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.project_root / "artifacts" / "release_manifest.json"
        )
        self._document = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Release manifest bulunamadı: {self.manifest_path}"
            )

        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError("Desteklenmeyen release manifest şema sürümü.")
        if not document.get("release_id"):
            raise ValueError("Release manifest içinde release_id eksik.")
        if not isinstance(document.get("artifacts"), dict):
            raise ValueError("Release manifest artifacts alanı geçersiz.")
        return document

    @property
    def release_id(self) -> str:
        return str(self._document["release_id"])

    def resolve(self, key: str, *, must_exist: bool = True) -> Path:
        value = self._document["artifacts"].get(key)
        if not value:
            raise KeyError(f"Release manifest artifact anahtarı bulunamadı: {key}")

        relative = Path(str(value))
        if relative.is_absolute():
            raise ValueError(f"Artifact yolu göreli olmalıdır: {key}")

        resolved = (self.project_root / relative).resolve()
        try:
            resolved.relative_to(self.project_root)
        except ValueError as error:
            raise ValueError(
                f"Artifact yolu proje kökü dışına çıkamaz: {key}"
            ) from error

        if must_exist and not resolved.exists():
            raise FileNotFoundError(
                f"Release '{self.release_id}' artifact'i eksik: {key} -> {resolved}"
            )
        return resolved

    def status(self) -> dict[str, Any]:
        artifacts = {
            key: {
                "path": str(self.resolve(key, must_exist=False)),
                "exists": self.resolve(key, must_exist=False).exists(),
            }
            for key in self._document["artifacts"]
        }
        return {
            "schema_version": self._document["schema_version"],
            "release_id": self.release_id,
            "promoted_at": self._document.get("promoted_at"),
            "primary_strategy": self._document.get("primary_strategy"),
            "ready": all(item["exists"] for item in artifacts.values()),
            "artifacts": artifacts,
        }
