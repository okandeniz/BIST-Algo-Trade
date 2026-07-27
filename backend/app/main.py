"""FastAPI entry point for the BIST100 Robot application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.dependencies import get_database
from backend.app.routers import (
    backtest,
    health,
    portfolio,
    signals,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_database().initialize()
    yield


settings = get_settings()

app = FastAPI(
    title="BIST100 Robot API",
    version="1.0.0",
    description=(
        "Final Robot backtest, daily signals and "
        "paper-trading portfolio API."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(backtest.router)
app.include_router(signals.router)
app.include_router(portfolio.router)


@app.get("/")
def root():
    return {
        "message": "BIST100 Robot API",
        "docs": "/docs",
        "health": "/health",
    }
