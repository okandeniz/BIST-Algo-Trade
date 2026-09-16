"""Soft RS126 experiments for the Baseline Robot.

The production Baseline rules remain unchanged. This module tests whether
126-day relative strength should influence candidate priority or marginal
Robot scores without becoming a hard eligibility filter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.backtest import run_portfolio_backtest
from src.config import PortfolioConfig, StrategyConfig
from src.metrics import portfolio_metrics
from src.signals import add_robot_scores
from src.baseline_enhancements import (
    add_enhancement_features,
    add_baseline_deltas,
    build_validation_acceptance_table,
    select_development_family_winners,
    summarize_block_stability,
)


@dataclass(frozen=True)
class RS126SoftVariant:
    """A soft relative-strength rule.

    The ranking component changes only which existing AL candidate receives
    priority when portfolio slots are limited. Score adjustments may promote
    or demote only marginal Robot scores.
    """

    name: str
    family: str
    rank_weight: float = 0.0
    positive_bonus: int = 0
    negative_penalty: int = 0
    top_percentile: float | None = None
    bottom_percentile: float | None = None
    percentile_bonus: int = 0
    percentile_penalty: int = 0


def baseline_variant() -> RS126SoftVariant:
    return RS126SoftVariant(
        name="Baseline",
        family="Baseline",
    )


def default_soft_variants() -> list[RS126SoftVariant]:
    """Pre-register a deliberately small soft-RS126 experiment grid."""
    return [
        baseline_variant(),
        RS126SoftVariant(
            name="RS126_RankWeight_0_25",
            family="RS126_Ranking",
            rank_weight=0.25,
        ),
        RS126SoftVariant(
            name="RS126_RankWeight_0_50",
            family="RS126_Ranking",
            rank_weight=0.50,
        ),
        RS126SoftVariant(
            name="RS126_RankWeight_0_75",
            family="RS126_Ranking",
            rank_weight=0.75,
        ),
        RS126SoftVariant(
            name="RS126_RankWeight_1_00",
            family="RS126_Ranking",
            rank_weight=1.00,
        ),
        RS126SoftVariant(
            name="RS126_Positive_Bonus_1",
            family="RS126_Score_Adjustment",
            positive_bonus=1,
        ),
        RS126SoftVariant(
            name="RS126_Negative_Penalty_1",
            family="RS126_Score_Adjustment",
            negative_penalty=1,
        ),
        RS126SoftVariant(
            name="RS126_Symmetric_Sign_1",
            family="RS126_Score_Adjustment",
            positive_bonus=1,
            negative_penalty=1,
        ),
        RS126SoftVariant(
            name="RS126_Top30_Bottom30_1",
            family="RS126_Score_Adjustment",
            top_percentile=0.70,
            bottom_percentile=0.30,
            percentile_bonus=1,
            percentile_penalty=1,
        ),
        RS126SoftVariant(
            name="RS126_Top20_Bottom20_1",
            family="RS126_Score_Adjustment",
            top_percentile=0.80,
            bottom_percentile=0.20,
            percentile_bonus=1,
            percentile_penalty=1,
        ),
    ]


def combine_soft_variants(
    first: RS126SoftVariant,
    second: RS126SoftVariant,
    name: str | None = None,
) -> RS126SoftVariant:
    """Combine one Development ranking winner and one score winner."""
    if first.family == second.family:
        raise ValueError(
            "Kombinasyon farklı ailelerden iki varyant içermelidir."
        )

    ranking = (
        first
        if first.family == "RS126_Ranking"
        else second
    )
    score_adjustment = (
        first
        if first.family == "RS126_Score_Adjustment"
        else second
    )

    if ranking.family != "RS126_Ranking":
        raise ValueError("Bir RS126_Ranking varyantı bulunmalıdır.")
    if score_adjustment.family != "RS126_Score_Adjustment":
        raise ValueError(
            "Bir RS126_Score_Adjustment varyantı bulunmalıdır."
        )

    return RS126SoftVariant(
        name=name or (
            "Combo__"
            f"{ranking.name}__{score_adjustment.name}"
        ),
        family="Combination",
        rank_weight=ranking.rank_weight,
        positive_bonus=score_adjustment.positive_bonus,
        negative_penalty=score_adjustment.negative_penalty,
        top_percentile=score_adjustment.top_percentile,
        bottom_percentile=score_adjustment.bottom_percentile,
        percentile_bonus=score_adjustment.percentile_bonus,
        percentile_penalty=score_adjustment.percentile_penalty,
    )


def prepare_soft_rs126_dataset(
    stock_features: pd.DataFrame,
    market_features: pd.DataFrame,
    market_regime: pd.Series,
    strategy_config: StrategyConfig,
    minimum_cross_section_size: int = 20,
) -> pd.DataFrame:
    """Create Baseline scores and cross-sectional RS126 features once."""
    if minimum_cross_section_size < 2:
        raise ValueError(
            "minimum_cross_section_size en az 2 olmalıdır."
        )

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

    scored["Baseline_Score"] = pd.to_numeric(
        scored["Score"],
        errors="coerce",
    )
    scored["Baseline_Signal"] = scored["Signal"]

    valid_counts = (
        scored["RS_126"]
        .notna()
        .groupby(scored["Date"])
        .transform("sum")
    )

    percentile = (
        scored.groupby("Date")["RS_126"]
        .rank(
            pct=True,
            method="average",
        )
    )

    scored["RS126_CrossSection_Count"] = valid_counts
    scored["RS126_Percentile"] = percentile.where(
        valid_counts >= minimum_cross_section_size
    )
    scored["RS126_Centered_Percentile"] = (
        scored["RS126_Percentile"] * 2.0 - 1.0
    ).fillna(0.0)

    return scored.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)


def _signals_from_score(
    score: pd.Series,
    strategy_config: StrategyConfig,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                score >= strategy_config.buy_score,
                score >= strategy_config.watch_score,
            ],
            ["AL", "İZLE"],
            default="ALMA",
        ),
        index=score.index,
    )


def apply_soft_variant(
    prepared_prices: pd.DataFrame,
    variant: RS126SoftVariant,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    """Apply a soft rule while preserving all original diagnostic columns."""
    required = {
        "Baseline_Score",
        "Baseline_Signal",
        "RS_126",
        "RS126_Percentile",
        "RS126_Centered_Percentile",
    }
    missing = required.difference(prepared_prices.columns)
    if missing:
        raise KeyError(
            f"Soft RS126 deneyi için eksik sütunlar: {sorted(missing)}"
        )

    result = prepared_prices.copy()
    baseline_score = pd.to_numeric(
        result["Baseline_Score"],
        errors="coerce",
    )

    score_adjustment = pd.Series(
        0.0,
        index=result.index,
    )

    valid_rs = result["RS_126"].notna()

    if variant.positive_bonus:
        score_adjustment += np.where(
            valid_rs & result["RS_126"].gt(0),
            float(variant.positive_bonus),
            0.0,
        )

    if variant.negative_penalty:
        score_adjustment -= np.where(
            valid_rs & result["RS_126"].le(0),
            float(variant.negative_penalty),
            0.0,
        )

    valid_percentile = result[
        "RS126_Percentile"
    ].notna()

    if (
        variant.top_percentile is not None
        and variant.percentile_bonus
    ):
        score_adjustment += np.where(
            valid_percentile
            & result["RS126_Percentile"].ge(
                variant.top_percentile
            ),
            float(variant.percentile_bonus),
            0.0,
        )

    if (
        variant.bottom_percentile is not None
        and variant.percentile_penalty
    ):
        score_adjustment -= np.where(
            valid_percentile
            & result["RS126_Percentile"].le(
                variant.bottom_percentile
            ),
            float(variant.percentile_penalty),
            0.0,
        )

    eligibility_score = baseline_score + score_adjustment

    has_score_adjustment = bool(
        variant.positive_bonus
        or variant.negative_penalty
        or variant.percentile_bonus
        or variant.percentile_penalty
    )

    if has_score_adjustment:
        signal = _signals_from_score(
            eligibility_score,
            strategy_config,
        )
    else:
        signal = result["Baseline_Signal"].copy()

    ranking_adjustment = (
        float(variant.rank_weight)
        * result["RS126_Centered_Percentile"]
    )
    candidate_priority = (
        eligibility_score + ranking_adjustment
    )

    result["Soft_Score_Adjustment"] = score_adjustment
    result["Eligibility_Score"] = eligibility_score
    result["Ranking_Adjustment"] = ranking_adjustment
    result["Candidate_Priority"] = candidate_priority

    # run_portfolio_backtest ranks by Score. Signal remains explicit, so
    # replacing Score with the candidate priority is safe for this experiment.
    result["Score"] = candidate_priority
    result["Signal"] = signal
    result["RS126_Soft_Variant"] = variant.name

    result["Promoted_To_AL"] = (
        result["Baseline_Signal"].ne("AL")
        & result["Signal"].eq("AL")
    )
    result["Demoted_From_AL"] = (
        result["Baseline_Signal"].eq("AL")
        & result["Signal"].ne("AL")
    )

    return result


def _slice_period(
    frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    return (
        frame.loc[
            frame["Date"].between(
                start_ts,
                end_ts,
                inclusive="both",
            )
        ]
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )


def evaluate_soft_variant(
    prepared_prices: pd.DataFrame,
    variant: RS126SoftVariant,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run one soft variant through the unchanged portfolio backtest."""
    variant_prices = apply_soft_variant(
        prepared_prices=prepared_prices,
        variant=variant,
        strategy_config=strategy_config,
    )
    period_prices = _slice_period(
        variant_prices,
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

    daily_al_counts = (
        period_prices.loc[
            period_prices["Signal"].eq("AL")
        ]
        .groupby("Date")
        .size()
    )

    metrics.update(
        {
            "Variant": variant.name,
            "Family": variant.family,
            "Period_Start": pd.Timestamp(start),
            "Period_End": pd.Timestamp(end),
            "Baseline_AL_Count": int(
                period_prices[
                    "Baseline_Signal"
                ].eq("AL").sum()
            ),
            "Variant_AL_Count": int(
                period_prices["Signal"].eq("AL").sum()
            ),
            "Promoted_To_AL_Count": int(
                period_prices["Promoted_To_AL"].sum()
            ),
            "Demoted_From_AL_Count": int(
                period_prices["Demoted_From_AL"].sum()
            ),
            "Signal_Change_Count": int(
                (
                    period_prices["Signal"]
                    != period_prices["Baseline_Signal"]
                ).sum()
            ),
            "Mean_AL_Candidates_Per_Signal_Day": (
                float(daily_al_counts.mean())
                if not daily_al_counts.empty
                else 0.0
            ),
            "Crowded_Signal_Days": int(
                daily_al_counts.gt(
                    portfolio_config.max_positions
                ).sum()
            ),
            "Median_AL_RS126": float(
                period_prices.loc[
                    period_prices["Signal"].eq("AL"),
                    "RS_126",
                ].median()
            ),
        }
    )

    for key, value in asdict(variant).items():
        metrics[f"RS126_{key}"] = value

    return metrics, equity, trades


def run_soft_grid(
    prepared_prices: pd.DataFrame,
    variants: Sequence[RS126SoftVariant],
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
        tuple[RS126SoftVariant, str, str, str]
    ] = jobs

    if show_progress:
        iterator = tqdm(
            jobs,
            desc="Soft RS126 experiments",
        )

    records: list[dict[str, object]] = []

    for variant, period_name, start, end in iterator:
        try:
            metrics, _, _ = evaluate_soft_variant(
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
                metrics[f"RS126_{key}"] = value

        metrics["Period_Name"] = period_name
        records.append(metrics)

    return pd.DataFrame(records)


def variants_by_name(
    variants: Sequence[RS126SoftVariant],
) -> dict[str, RS126SoftVariant]:
    return {
        variant.name: variant
        for variant in variants
    }


def build_validation_variants(
    development_winners: pd.DataFrame,
    development_variants: Sequence[RS126SoftVariant],
) -> list[RS126SoftVariant]:
    """Create Validation candidates without looking at Validation results."""
    variant_map = variants_by_name(
        development_variants
    )

    winners = [
        variant_map[name]
        for name in development_winners[
            "Variant"
        ].tolist()
    ]

    result = [baseline_variant()] + winners

    ranking_winners = [
        variant
        for variant in winners
        if variant.family == "RS126_Ranking"
    ]
    score_winners = [
        variant
        for variant in winners
        if variant.family == "RS126_Score_Adjustment"
    ]

    if ranking_winners and score_winners:
        result.append(
            combine_soft_variants(
                ranking_winners[0],
                score_winners[0],
            )
        )

    return result


def save_soft_rs126_artifacts(
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
    """Save experiment tables under the ignored local results directory."""
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


__all__ = [
    "RS126SoftVariant",
    "add_baseline_deltas",
    "baseline_variant",
    "build_validation_acceptance_table",
    "build_validation_variants",
    "combine_soft_variants",
    "default_soft_variants",
    "evaluate_soft_variant",
    "prepare_soft_rs126_dataset",
    "run_soft_grid",
    "save_soft_rs126_artifacts",
    "select_development_family_winners",
    "summarize_block_stability",
    "variants_by_name",
]
