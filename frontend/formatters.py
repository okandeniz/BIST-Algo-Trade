"""Turkish display formatters."""

from __future__ import annotations


def format_tl(value: float | int | None) -> str:
    if value is None:
        return "—"

    formatted = f"{float(value):,.2f}"
    formatted = (
        formatted
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    return f"{formatted} TL"


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "—"

    formatted = f"{float(value):,.2f}"
    formatted = (
        formatted
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    return f"%{formatted}"
