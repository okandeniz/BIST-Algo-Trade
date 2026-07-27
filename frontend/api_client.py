"""Small HTTP client used by the Streamlit frontend."""

from __future__ import annotations

from typing import Any

import httpx


class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: float = 30.0,
        json: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"

        try:
            response = httpx.request(
                method,
                url,
                timeout=timeout,
                json=json,
            )
        except httpx.RequestError as error:
            raise RuntimeError(
                f"API bağlantı hatası: {error}"
            ) from error

        if response.is_error:
            try:
                detail = response.json().get(
                    "detail",
                    response.text,
                )
            except ValueError:
                detail = response.text

            raise RuntimeError(
                f"API {response.status_code}: {detail}"
            )

        if not response.content:
            return None

        return response.json()

    def get(
        self,
        path: str,
        *,
        timeout: float = 30.0,
    ) -> Any:
        return self.request(
            "GET",
            path,
            timeout=timeout,
        )

    def post(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> Any:
        return self.request(
            "POST",
            path,
            timeout=timeout,
            json=payload,
        )
