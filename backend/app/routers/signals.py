"""Daily signal endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.dependencies import get_signal_service


router = APIRouter(
    prefix="/api/signals",
    tags=["Signals"],
)


@router.get("/latest")
def latest():
    try:
        return get_signal_service().latest()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.post("/refresh")
def refresh():
    try:
        return get_signal_service().refresh()
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
