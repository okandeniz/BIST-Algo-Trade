"""Backtest API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.dependencies import (
    get_backtest_service,
    get_walkforward_service,
    get_enhanced_backtest_service,
)


router = APIRouter(
    prefix="/api/backtest",
    tags=["Backtest"],
)


def _handle(callable_):
    try:
        return callable_()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.get("/summary")
def summary():
    return _handle(
        get_backtest_service().summary
    )


@router.get("/equity")
def equity():
    return _handle(
        get_backtest_service().equity
    )


@router.get("/yearly")
def yearly():
    return _handle(
        get_backtest_service().yearly
    )


@router.get("/monthly-summary")
def monthly_summary():
    return _handle(
        get_backtest_service().monthly_summary
    )



@router.get("/walk-forward/summary")
def walk_forward_summary():
    return _handle(
        get_walkforward_service().summary
    )


@router.get("/walk-forward/equity")
def walk_forward_equity():
    return _handle(
        get_walkforward_service().equity
    )


@router.get("/walk-forward/yearly")
def walk_forward_yearly():
    return _handle(
        get_walkforward_service().yearly
    )


@router.get("/walk-forward/training-log")
def walk_forward_training_log():
    return _handle(
        get_walkforward_service().training_log
    )


@router.get("/enhanced/summary")
def enhanced_summary():
    return _handle(
        get_enhanced_backtest_service().summary
    )


@router.get("/enhanced/equity")
def enhanced_equity():
    return _handle(
        get_enhanced_backtest_service().equity
    )


@router.get("/enhanced/yearly")
def enhanced_yearly():
    return _handle(
        get_enhanced_backtest_service().yearly
    )


@router.post("/enhanced/refresh")
def enhanced_refresh():
    return _handle(
        get_enhanced_backtest_service().refresh
    )
