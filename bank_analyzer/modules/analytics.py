"""Analytics helpers: KPIs, trends, anomalies, and budget checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .anomaly import AnomalyDetector, ZScoreAnomalyDetector
from .constants import INCOME_CATEGORIES
from .models import KPIMetrics, TransactionType


def _expense_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "transaction_type" in df.columns:
        return df[df["transaction_type"] == TransactionType.EXPENSE.value].copy()
    return df.copy()


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total spend, percentage, and transaction count per category.

    Returns DataFrame with columns: category, total_spend, pct_of_total, transaction_count.
    """
    expenses = _expense_df(df)
    if expenses.empty or "category" not in expenses.columns:
        return pd.DataFrame(
            columns=["category", "total_spend", "pct_of_total", "transaction_count"]
        )

    grouped = (
        expenses.groupby("category", dropna=False)["amount"]
        .agg(total_spend="sum", transaction_count="count")
        .reset_index()
    )
    total = grouped["total_spend"].sum()
    grouped["pct_of_total"] = np.where(
        total > 0,
        (grouped["total_spend"] / total * 100).round(2),
        0.0,
    )
    return grouped.sort_values("total_spend", ascending=False).reset_index(drop=True)


def get_monthly_trends(
    df: pd.DataFrame,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Resample expenses by month (default) or week for time-series charts.

    freq: 'ME' for month-end, 'W' for weekly.
    Returns DataFrame with period index as 'period' column and total_spend.
    """
    expenses = _expense_df(df)
    if expenses.empty or "date" not in expenses.columns:
        return pd.DataFrame(columns=["period", "total_spend"])

    work = expenses.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.set_index("date").sort_index()
    resampled = work["amount"].resample(freq).sum().reset_index()
    resampled.columns = ["period", "total_spend"]
    resampled["period"] = resampled["period"].dt.strftime("%Y-%m-%d")
    return resampled


def get_weekly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly expense resampling wrapper."""
    return get_monthly_trends(df, freq="W")


def detect_anomalies(
    df: pd.DataFrame,
    threshold_std: float = 2.5,
    detector: AnomalyDetector | None = None,
) -> pd.DataFrame:
    """
    Flag unusually large transactions using the configured anomaly strategy.

    Adds columns: category_mean, category_std, z_score, is_anomaly.
    """
    strategy = detector or ZScoreAnomalyDetector(threshold_std=threshold_std)
    return strategy.detect(_expense_df(df))


def check_budget_limits(
    df: pd.DataFrame,
    budget_dict: dict[str, float],
) -> pd.DataFrame:
    """
    Compare actual category spending against user-defined budget caps.

    Returns DataFrame: category, budget, actual, remaining, pct_used, over_budget.
    """
    summary = get_category_summary(df)
    rows: list[dict[str, Any]] = []

    actual_categories = set(summary["category"].unique()) if not summary.empty else set()
    budget_categories = set(budget_dict.keys())

    all_categories = sorted((actual_categories | budget_categories) - INCOME_CATEGORIES)
    for category in all_categories:
        budget = float(budget_dict.get(category, 0.0))

        actual_row = (
            summary[summary["category"] == category] if not summary.empty else pd.DataFrame()
        )
        actual = float(actual_row["total_spend"].iloc[0]) if not actual_row.empty else 0.0

        remaining = budget - actual
        pct_used = round((actual / budget * 100), 2) if budget > 0 else 0.0

        rows.append(
            {
                "category": category,
                "budget": budget,
                "actual": round(actual, 2),
                "remaining": round(remaining, 2),
                "pct_used": pct_used,
                "over_budget": actual > budget if budget > 0 else False,
            }
        )

    return pd.DataFrame(rows).sort_values("pct_used", ascending=False).reset_index(drop=True)


def compute_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute top-level KPI metrics for dashboard display (dict for UI compatibility)."""
    return AnalyticsService().compute_kpis(df).as_dict()


class AnalyticsService:
    """Facade over analytics helpers so UIs depend on one object, not free functions."""

    def category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        return get_category_summary(df)

    def monthly_trends(self, df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
        return get_monthly_trends(df, freq=freq)

    def weekly_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        return get_weekly_trends(df)

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        threshold_std: float = 2.5,
        detector: AnomalyDetector | None = None,
    ) -> pd.DataFrame:
        return detect_anomalies(df, threshold_std=threshold_std, detector=detector)

    def check_budget_limits(
        self, df: pd.DataFrame, budget_dict: dict[str, float]
    ) -> pd.DataFrame:
        return check_budget_limits(df, budget_dict)

    def compute_kpis(self, df: pd.DataFrame) -> KPIMetrics:
        if df.empty:
            return KPIMetrics(0.0, 0.0, 0.0, 0)

        income = df.loc[
            df["transaction_type"] == TransactionType.INCOME.value, "amount"
        ].sum()
        expenses = df.loc[
            df["transaction_type"] == TransactionType.EXPENSE.value, "amount"
        ].sum()
        return KPIMetrics(
            total_income=round(float(income), 2),
            total_expenses=round(float(expenses), 2),
            net_savings=round(float(income - expenses), 2),
            transaction_count=len(df),
        )
