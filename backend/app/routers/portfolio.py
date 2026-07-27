"""Portfolio, transaction and P&L endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.dependencies import (
    get_market_service,
    get_repository,
)
from backend.app.schemas import (
    BuyRequest,
    InitialCapitalRequest,
    PortfolioName,
    ResetPortfolioRequest,
    SellRequest,
)


router = APIRouter(
    prefix="/api",
    tags=["Portfolio"],
)


@router.get("/portfolios")
def portfolios():
    return get_repository().list_portfolios()


@router.get("/portfolio/{portfolio_name}/summary")
def portfolio_summary(
    portfolio_name: PortfolioName,
):
    try:
        return get_repository().summary(
            portfolio_name,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.get("/portfolio/{portfolio_name}/positions")
def positions(
    portfolio_name: PortfolioName,
):
    try:
        return get_repository().positions(
            portfolio_name,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.get("/portfolio/{portfolio_name}/transactions")
def transactions(
    portfolio_name: PortfolioName,
    limit: int = Query(default=500, ge=1, le=5000),
):
    try:
        return get_repository().transactions(
            portfolio_name,
            limit=limit,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.post(
    "/portfolio/{portfolio_name}/initial-capital"
)
def set_initial_capital(
    portfolio_name: PortfolioName,
    request: InitialCapitalRequest,
):
    try:
        return get_repository().set_initial_capital(
            portfolio_name,
            request.initial_capital,
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error


@router.post("/portfolio/{portfolio_name}/reset")
def reset_portfolio(
    portfolio_name: PortfolioName,
    request: ResetPortfolioRequest,
):
    try:
        return get_repository().reset_portfolio(
            portfolio_name,
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error


@router.post("/portfolio/buy")
def buy(request: BuyRequest):
    try:
        return get_repository().buy(request)
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error


@router.post("/portfolio/sell")
def sell(request: SellRequest):
    try:
        return get_repository().sell(request)
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error


@router.post(
    "/portfolio/{portfolio_name}/refresh-prices"
)
def refresh_position_prices(
    portfolio_name: PortfolioName,
):
    """Refresh only the last-price field of current open positions."""
    repository = get_repository()

    try:
        current_positions = repository.positions(
            portfolio_name,
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    tickers = [
        str(position["Ticker"])
        for position in current_positions
    ]

    if not tickers:
        return {
            "portfolio_name": portfolio_name,
            "requested_count": 0,
            "updated_count": 0,
            "failed_count": 0,
            "quotes": [],
            "message": "Güncellenecek açık pozisyon yok.",
        }

    quotes = get_market_service().fetch_latest_quotes(
        tickers
    )

    successful_prices = {
        str(record["Ticker"]): float(
            record["Price"]
        )
        for record in quotes
        if (
            record.get("Status") == "OK"
            and record.get("Price") is not None
        )
    }

    if not successful_prices:
        errors = [
            str(record.get("Error"))
            for record in quotes
            if record.get("Error")
        ]

        raise HTTPException(
            status_code=502,
            detail=(
                "Açık pozisyonlar için güncel fiyat "
                "alınamadı. "
                + " | ".join(errors)
            ),
        )

    try:
        updated_count = (
            repository.update_last_prices(
                portfolio_name,
                successful_prices,
            )
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    failed_count = len(quotes) - len(
        successful_prices
    )

    return {
        "portfolio_name": portfolio_name,
        "requested_count": len(tickers),
        "updated_count": updated_count,
        "failed_count": failed_count,
        "quotes": quotes,
        "message": (
            f"{updated_count} pozisyonun son fiyatı "
            "güncellendi."
        ),
    }
