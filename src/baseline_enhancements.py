"""Controlled enhancement experiments for the Baseline Robot.

This module does not modify the production signal rules. It adds research
features, filters existing Baseline AL signals and compares the resulting
variants with the unchanged Baseline through the existing portfolio backtest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.backtest import run_portfolio_backtest
from src.config import PortfolioConfig, StrategyConfig
from src.metrics import portfolio_metrics
from src.signals import add_robot_scores


@dataclass(frozen=True)
class EnhancementVariant:
    """A transparent filter applied only to existing Baseline AL signals."""

    name: str
    family: str
    minimum_rs_63: float | None = None
    minimum_rs_126: float | None = None
    maximum_ema50_distance_atr: float | None = None
    minimum_breadth_ema50: float | None = None
    minimum_ema50_slope_10: float | None = None


def baseline_variant() -> EnhancementVariant:
    return EnhancementVariant(
        name="Baseline",
        family="Baseline",
    )


def default_single_factor_variants() -> list[EnhancementVariant]:
    """Return the pre-registered single-factor experiment set."""
    return [
        baseline_variant(),
        EnhancementVariant(
            name="RS63_Positive",
            family="Relative_Strength",
            minimum_rs_63=0.0,
        ),
        EnhancementVariant(
            name="RS126_Positive",
            family="Relative_Strength",
            minimum_rs_126=0.0,
        ),
        EnhancementVariant(
            name="RS63_RS126_Positive",
            family="Relative_Strength",
            minimum_rs_63=0.0,
            minimum_rs_126=0.0,
        ),
        EnhancementVariant(
            name="EMA50_Distance_ATR_2_5",
            family="EMA50_Distance",
            maximum_ema50_distance_atr=2.5,
        ),
        EnhancementVariant(
            name="EMA50_Distance_ATR_3_0",
            family="EMA50_Distance",
            maximum_ema50_distance_atr=3.0,
        ),
        EnhancementVariant(
            name="EMA50_Distance_ATR_3_5",
            family="EMA50_Distance",
            maximum_ema50_distance_atr=3.5,
        ),
        EnhancementVariant(
            name="EMA50_Distance_ATR_4_0",
            family="EMA50_Distance",
            maximum_ema50_distance_atr=4.0,
        ),
        EnhancementVariant(
            name="Breadth_EMA50_45",
            family="Market_Breadth",
            minimum_breadth_ema50=0.45,
        ),
        EnhancementVariant(
            name="Breadth_EMA50_50",
            family="Market_Breadth",
            minimum_breadth_ema50=0.50,
        ),
        EnhancementVariant(
            name="Breadth_EMA50_55",
            family="Market_Breadth",
            minimum_breadth_ema50=0.55,
        ),
        EnhancementVariant(
            name="Breadth_EMA50_60",
            family="Market_Breadth",
            minimum_breadth_ema50=0.60,
        ),
        EnhancementVariant(
            name="EMA50_Slope_10_Positive",
            family="EMA50_Slope",
            minimum_ema50_slope_10=0.0,
        ),
    ]


def _merge_constraint(
    current: float | None,
    candidate: float | None,
    mode: str,
) -> float | None:
    if current is None:
        return candidate
    if candidate is None:
        return current

    if mode == "minimum":
        return max(current, candidate)
    if mode == "maximum":
        return min(current, candidate)

    raise ValueError(f"Bilinmeyen birleştirme modu: {mode}")


def combine_variants(
    variants: Sequence[EnhancementVariant],
    name: str | None = None,
) -> EnhancementVariant:
    """Combine independent family winners into one transparent filter."""
    if len(variants) < 2:
        raise ValueError("Kombinasyon için en az iki varyant gerekir.")

    families = [variant.family for variant in variants]
    if len(set(families)) != len(families):
        raise ValueError(
            "Aynı aileden birden fazla eşik tek kombinasyona eklenemez."
        )

    combined = EnhancementVariant(
        name=name or "Combo__" + "__".join(
            variant.name for variant in variants
        ),
        family="Combination",
    )

    for variant in variants:
        combined = EnhancementVariant(
            name=combined.name,
            family=combined.family,
            minimum_rs_63=_merge_constraint(
                combined.minimum_rs_63,
                variant.minimum_rs_63,
                "minimum",
            ),
            minimum_rs_126=_merge_constraint(
                combined.minimum_rs_126,
                variant.minimum_rs_126,
                "minimum",
            ),
            maximum_ema50_distance_atr=_merge_constraint(
                combined.maximum_ema50_distance_atr,
                variant.maximum_ema50_distance_atr,
                "maximum",
            ),
            minimum_breadth_ema50=_merge_constraint(
                combined.minimum_breadth_ema50,
                variant.minimum_breadth_ema50,
                "minimum",
            ),
            minimum_ema50_slope_10=_merge_constraint(
                combined.minimum_ema50_slope_10,
                variant.minimum_ema50_slope_10,
                "minimum",
            ),
        )

    return combined


def build_combination_variants(
    family_winners: Sequence[EnhancementVariant],
    minimum_size: int = 2,
    maximum_size: int = 3,
) -> list[EnhancementVariant]:
    """Build combinations only from Development-selected family winners."""
    usable = [
        variant
        for variant in family_winners
        if variant.family not in {"Baseline", "Combination"}
    ]

    results: list[EnhancementVariant] = []

    upper = min(maximum_size, len(usable))
    for size in range(minimum_size, upper + 1):
        for group in combinations(usable, size):
            results.append(combine_variants(group))

    return results


def add_enhancement_features(
    stock_features: pd.DataFrame,
    market_features: pd.DataFrame,
    slope_window: int = 10,
) -> pd.DataFrame:
    """Add orthogonal research features without changing Robot scores."""
    stock_required = {
        "Date",
        "Ticker",
        "Close",
        "EMA50",
        "ATR",
        "RET_63",
        "RET_126",
    }
    market_required = {"Date", "RET_63", "RET_126"}

    stock_missing = stock_required.difference(stock_features.columns)
    market_missing = market_required.difference(market_features.columns)

    if stock_missing:
        raise KeyError(
            f"Hisse özelliklerinde eksik sütunlar: {sorted(stock_missing)}"
        )
    if market_missing:
        raise KeyError(
            f"Endeks özelliklerinde eksik sütunlar: {sorted(market_missing)}"
        )
    if slope_window < 1:
        raise ValueError("slope_window en az 1 olmalıdır.")

    result = stock_features.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    result = result.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    market = (
        market_features[["Date", "RET_63", "RET_126"]]
        .copy()
        .assign(Date=lambda frame: pd.to_datetime(frame["Date"]))
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .rename(
            columns={
                "RET_63": "MARKET_RET_63",
                "RET_126": "MARKET_RET_126",
            }
        )
    )

    result = result.merge(
        market,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    result["RS_63"] = (
        result["RET_63"] - result["MARKET_RET_63"]
    )
    result["RS_126"] = (
        result["RET_126"] - result["MARKET_RET_126"]
    )

    result["EMA50_SLOPE_10"] = (
        result.groupby("Ticker", sort=False)["EMA50"]
        .pct_change(slope_window, fill_method=None)
    )

    valid_atr = result["ATR"].where(result["ATR"] > 0)
    result["EMA50_DISTANCE_ATR"] = (
        result["Close"] - result["EMA50"]
    ) / valid_atr

    breadth_source = result.loc[
        result["Close"].notna() & result["EMA50"].notna(),
        ["Date", "Close", "EMA50"],
    ].copy()
    breadth_source["ABOVE_EMA50"] = (
        breadth_source["Close"] > breadth_source["EMA50"]
    )

    breadth = (
        breadth_source.groupby("Date", as_index=False)
        .agg(
            BREADTH_EMA50=("ABOVE_EMA50", "mean"),
            BREADTH_UNIVERSE_COUNT=("ABOVE_EMA50", "size"),
        )
    )

    result = result.merge(
        breadth,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    return result.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def prepare_enhancement_dataset(
    stock_features: pd.DataFrame,
    market_features: pd.DataFrame,
    market_regime: pd.Series,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    """Create one reusable scored dataset for every enhancement variant."""
    enhanced = add_enhancement_features(
        stock_features=stock_features,
        market_features=market_features,
    )

    scored = add_robot_scores(
        stock_features=enhanced,
        market_regime=market_regime,
        config=strategy_config,
        include_reasons=False,
    )

    scored["Baseline_Signal"] = scored["Signal"]
    return scored


def apply_enhancement_variant(
    scored_prices: pd.DataFrame,
    variant: EnhancementVariant,
) -> pd.DataFrame:
    """Filter only Baseline AL rows; exits and ranking remain unchanged."""
    if "Baseline_Signal" not in scored_prices.columns:
        raise KeyError(
            "Önce prepare_enhancement_dataset kullanılmalıdır."
        )

    result = scored_prices.copy()
    passes = pd.Series(True, index=result.index, dtype=bool)

    constraints = [
        (
            "RS_63",
            variant.minimum_rs_63,
            "minimum",
        ),
        (
            "RS_126",
            variant.minimum_rs_126,
            "minimum",
        ),
        (
            "EMA50_DISTANCE_ATR",
            variant.maximum_ema50_distance_atr,
            "maximum",
        ),
        (
            "BREADTH_EMA50",
            variant.minimum_breadth_ema50,
            "minimum",
        ),
        (
            "EMA50_SLOPE_10",
            variant.minimum_ema50_slope_10,
            "minimum",
        ),
    ]

    for column, threshold, mode in constraints:
        if threshold is None:
            continue
        if column not in result.columns:
            raise KeyError(
                f"{variant.name} için gerekli sütun yok: {column}"
            )

        values = pd.to_numeric(result[column], errors="coerce")
        if mode == "minimum":
            condition = values > threshold
        else:
            condition = values <= threshold

        passes &= condition.fillna(False)

    baseline_buy = result["Baseline_Signal"].eq("AL")
    filtered = baseline_buy & ~passes

    result["Signal"] = result["Baseline_Signal"]
    result.loc[filtered, "Signal"] = "İZLE"
    result["Enhancement_Pass"] = passes
    result["Enhancement_Filtered"] = filtered
    result["Enhancement_Variant"] = variant.name

    return result


def _slice_period(
    frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    result = frame.loc[
        frame["Date"].between(
            start_ts,
            end_ts,
            inclusive="both",
        )
    ].copy()

    return result.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def evaluate_enhancement_variant(
    prepared_prices: pd.DataFrame,
    variant: EnhancementVariant,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run one enhancement variant through the existing Robot backtest."""
    filtered_prices = apply_enhancement_variant(
        prepared_prices,
        variant,
    )
    period_prices = _slice_period(
        filtered_prices,
        start=start,
        end=end,
    )

    if period_prices.empty:
        raise ValueError(
            f"Seçilen dönemde veri yok: {start} - {end}"
        )

    equity, trades = run_portfolio_backtest(
        scored_prices=period_prices,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
    )

    metrics = portfolio_metrics(equity, trades)

    baseline_buy_count = int(
        period_prices["Baseline_Signal"].eq("AL").sum()
    )
    passed_buy_count = int(
        (
            period_prices["Baseline_Signal"].eq("AL")
            & period_prices["Enhancement_Pass"]
        ).sum()
    )

    metrics.update(
        {
            "Variant": variant.name,
            "Family": variant.family,
            "Period_Start": pd.Timestamp(start),
            "Period_End": pd.Timestamp(end),
            "Baseline_Buy_Signal_Count": baseline_buy_count,
            "Passed_Buy_Signal_Count": passed_buy_count,
            "Signal_Pass_Rate_%": (
                passed_buy_count / baseline_buy_count * 100
                if baseline_buy_count > 0
                else np.nan
            ),
        }
    )

    for key, value in asdict(variant).items():
        metrics[f"Enhancement_{key}"] = value

    return metrics, equity, trades


def run_enhancement_grid(
    prepared_prices: pd.DataFrame,
    variants: Sequence[EnhancementVariant],
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    periods: Mapping[str, tuple[str, str]],
    show_progress: bool = True,
) -> pd.DataFrame:
    """Evaluate every variant independently on every named period."""
    jobs = [
        (variant, period_name, start, end)
        for variant in variants
        for period_name, (start, end) in periods.items()
    ]

    iterator: Iterable[
        tuple[EnhancementVariant, str, str, str]
    ] = jobs

    if show_progress:
        iterator = tqdm(
            jobs,
            desc="Baseline enhancement experiments",
        )

    records: list[dict[str, object]] = []

    for variant, period_name, start, end in iterator:
        try:
            metrics, _, _ = evaluate_enhancement_variant(
                prepared_prices=prepared_prices,
                variant=variant,
                strategy_config=strategy_config,
                portfolio_config=portfolio_config,
                start=start,
                end=end,
            )
            metrics["Status"] = "OK"
            metrics["Error"] = ""
        except Exception as error:
            metrics = {
                "Variant": variant.name,
                "Family": variant.family,
                "Period_Start": pd.Timestamp(start),
                "Period_End": pd.Timestamp(end),
                "Status": "ERROR",
                "Error": str(error),
            }
            for key, value in asdict(variant).items():
                metrics[f"Enhancement_{key}"] = value

        metrics["Period_Name"] = period_name
        records.append(metrics)

    return pd.DataFrame(records)


def add_baseline_deltas(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Add per-period deltas relative to the unchanged Baseline."""
    required = {
        "Variant",
        "Period_Name",
        "CAGR_%",
        "Max_Drawdown_%",
        "Profit_Factor",
        "Sharpe",
        "Calmar",
        "Trade_Count",
    }
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(
            f"Karşılaştırma için eksik sütunlar: {sorted(missing)}"
        )

    output_frames: list[pd.DataFrame] = []

    for period_name, group in results.groupby("Period_Name"):
        baseline_rows = group.loc[
            group["Variant"].eq("Baseline")
            & group["Status"].eq("OK")
        ]
        if len(baseline_rows) != 1:
            raise ValueError(
                f"{period_name} için tek Baseline satırı bulunmalıdır."
            )

        baseline = baseline_rows.iloc[0]
        work = group.copy()

        work["CAGR_Delta_pp"] = (
            work["CAGR_%"] - baseline["CAGR_%"]
        )
        work["Max_DD_Improvement_pp"] = (
            work["Max_Drawdown_%"]
            - baseline["Max_Drawdown_%"]
        )
        work["Profit_Factor_Delta"] = (
            work["Profit_Factor"]
            - baseline["Profit_Factor"]
        )
        work["Sharpe_Delta"] = (
            work["Sharpe"] - baseline["Sharpe"]
        )
        work["Calmar_Delta"] = (
            work["Calmar"] - baseline["Calmar"]
        )
        work["Calmar_Improvement_%"] = np.where(
            pd.notna(baseline["Calmar"])
            & (abs(float(baseline["Calmar"])) > 1e-12),
            work["Calmar_Delta"]
            / abs(float(baseline["Calmar"]))
            * 100,
            np.nan,
        )
        work["Trade_Fraction"] = np.where(
            float(baseline["Trade_Count"]) > 0,
            work["Trade_Count"]
            / float(baseline["Trade_Count"]),
            np.nan,
        )
        output_frames.append(work)

    return pd.concat(
        output_frames,
        ignore_index=True,
    )


def select_development_family_winners(
    development_results: pd.DataFrame,
    minimum_trade_fraction: float = 0.70,
    maximum_drawdown_deterioration_pp: float = 3.0,
    minimum_profit_factor_ratio: float = 0.95,
) -> pd.DataFrame:
    """Select at most one threshold from each family using Development only."""
    results = add_baseline_deltas(development_results)
    development = results.loc[
        results["Period_Name"].eq("Development")
        & results["Status"].eq("OK")
    ].copy()

    baseline = development.loc[
        development["Variant"].eq("Baseline")
    ].iloc[0]

    candidates = development.loc[
        development["Family"].ne("Baseline")
        & development["Trade_Fraction"].ge(
            minimum_trade_fraction
        )
        & development["Max_DD_Improvement_pp"].ge(
            -maximum_drawdown_deterioration_pp
        )
        & development["Profit_Factor"].ge(
            float(baseline["Profit_Factor"])
            * minimum_profit_factor_ratio
        )
    ].copy()

    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        [
            "Family",
            "Calmar",
            "CAGR_%",
            "Profit_Factor",
            "Sharpe",
            "Trade_Count",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
        ],
    )

    return (
        candidates.groupby("Family", as_index=False)
        .head(1)
        .sort_values(
            ["Calmar", "CAGR_%"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_validation_acceptance_table(
    validation_results: pd.DataFrame,
    minimum_cagr_improvement_pp: float = 1.5,
    maximum_return_case_drawdown_deterioration_pp: float = 0.0,
    minimum_return_case_calmar_improvement_pct: float = 5.0,
    minimum_risk_case_cagr_delta_pp: float = -1.0,
    minimum_drawdown_improvement_pp: float = 2.0,
    minimum_risk_case_calmar_improvement_pct: float = 8.0,
    minimum_trade_fraction: float = 0.70,
    minimum_profit_factor_ratio: float = 0.95,
) -> pd.DataFrame:
    """Apply pre-declared return- and risk-focused acceptance rules."""
    table = add_baseline_deltas(validation_results)
    table = table.loc[
        table["Period_Name"].eq("Validation")
        & table["Status"].eq("OK")
    ].copy()

    baseline = table.loc[
        table["Variant"].eq("Baseline")
    ].iloc[0]

    table["Return_Case_Accepted"] = (
        table["CAGR_Delta_pp"].ge(
            minimum_cagr_improvement_pp
        )
        & table["Max_DD_Improvement_pp"].ge(
            -maximum_return_case_drawdown_deterioration_pp
        )
        & table["Calmar_Improvement_%"].ge(
            minimum_return_case_calmar_improvement_pct
        )
        & table["Profit_Factor_Delta"].ge(0.0)
    )

    table["Risk_Case_Accepted"] = (
        table["CAGR_Delta_pp"].ge(
            minimum_risk_case_cagr_delta_pp
        )
        & table["Max_DD_Improvement_pp"].ge(
            minimum_drawdown_improvement_pp
        )
        & table["Calmar_Improvement_%"].ge(
            minimum_risk_case_calmar_improvement_pct
        )
        & table["Profit_Factor"].ge(
            float(baseline["Profit_Factor"])
            * minimum_profit_factor_ratio
        )
    )

    table["Enough_Trades"] = table[
        "Trade_Fraction"
    ].ge(minimum_trade_fraction)

    table["Enhancement_Accepted"] = (
        table["Variant"].ne("Baseline")
        & table["Enough_Trades"]
        & (
            table["Return_Case_Accepted"]
            | table["Risk_Case_Accepted"]
        )
    )

    return table.sort_values(
        [
            "Enhancement_Accepted",
            "Calmar",
            "CAGR_%",
            "Profit_Factor",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def summarize_block_stability(
    block_results: pd.DataFrame,
    minimum_positive_blocks: int = 3,
) -> pd.DataFrame:
    """Count blocks where a candidate improves risk-adjusted performance."""
    table = add_baseline_deltas(block_results)

    candidates = table.loc[
        table["Variant"].ne("Baseline")
        & table["Status"].eq("OK")
    ].copy()

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "Variant",
                "Family",
                "Positive_Blocks",
                "Total_Blocks",
                "Stable_Across_Blocks",
            ]
        )

    candidates["Positive_Block"] = (
        candidates["Calmar_Delta"].ge(0.0)
        & (
            candidates["CAGR_Delta_pp"].ge(-1.0)
            | candidates["Max_DD_Improvement_pp"].ge(1.0)
        )
        & candidates["Trade_Fraction"].ge(0.60)
    )

    summary = (
        candidates.groupby(
            ["Variant", "Family"],
            as_index=False,
        )
        .agg(
            Positive_Blocks=(
                "Positive_Block",
                "sum",
            ),
            Total_Blocks=(
                "Period_Name",
                "nunique",
            ),
            Worst_CAGR_Delta_pp=(
                "CAGR_Delta_pp",
                "min",
            ),
            Worst_DD_Improvement_pp=(
                "Max_DD_Improvement_pp",
                "min",
            ),
            Median_Calmar_Delta=(
                "Calmar_Delta",
                "median",
            ),
            Minimum_Trade_Fraction=(
                "Trade_Fraction",
                "min",
            ),
        )
    )

    summary["Stable_Across_Blocks"] = (
        summary["Positive_Blocks"].ge(
            minimum_positive_blocks
        )
    )

    return summary.sort_values(
        [
            "Stable_Across_Blocks",
            "Positive_Blocks",
            "Median_Calmar_Delta",
            "Worst_CAGR_Delta_pp",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def variants_by_name(
    variants: Sequence[EnhancementVariant],
) -> dict[str, EnhancementVariant]:
    return {variant.name: variant for variant in variants}


def save_enhancement_artifacts(
    output_directory: str | Path,
    development_results: pd.DataFrame,
    development_winners: pd.DataFrame,
    validation_results: pd.DataFrame,
    acceptance_table: pd.DataFrame,
    block_results: pd.DataFrame | None = None,
    block_summary: pd.DataFrame | None = None,
    audit_results: pd.DataFrame | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Save reproducible experiment tables under the ignored results folder."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    tables = {
        "development_results": development_results,
        "development_winners": development_winners,
        "validation_results": validation_results,
        "acceptance_table": acceptance_table,
        "block_results": block_results,
        "block_summary": block_summary,
        "audit_results": audit_results,
    }

    for name, frame in tables.items():
        if frame is None:
            continue
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifacts[name] = path

    metadata_path = output / "metadata.json"
    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dict(metadata or {}),
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    artifacts["metadata"] = metadata_path

    return artifacts
