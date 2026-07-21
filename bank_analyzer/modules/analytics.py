"""Analytics helpers: KPIs, trends, anomalies, and budget checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _expense_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "transaction_type" in df.columns:
        return df[df["transaction_type"] == "expense"].copy()
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
) -> pd.DataFrame:
    """
    Flag unusually large transactions based on category mean + threshold * std.

    Adds columns: category_mean, category_std, z_score, is_anomaly.
    """
    expenses = _expense_df(df)
    if expenses.empty or "category" not in expenses.columns:
        return pd.DataFrame()

    work = expenses.copy()
    stats = work.groupby("category")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "category_mean", "std": "category_std"}
    )
    work = work.merge(stats, left_on="category", right_index=True, how="left")
    work["category_std"] = work["category_std"].fillna(0)
    work["z_score"] = np.where(
        work["category_std"] > 0,
        (work["amount"] - work["category_mean"]) / work["category_std"],
        0.0,
    )
    work["is_anomaly"] = work["z_score"] >= threshold_std
    return work[work["is_anomaly"]].sort_values("z_score", ascending=False).reset_index(drop=True)


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

    for category, budget in budget_dict.items():
        actual_row = summary[summary["category"] == category]
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
                "over_budget": actual > budget,
            }
        )

    return pd.DataFrame(rows).sort_values("pct_used", ascending=False).reset_index(drop=True)


def compute_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute top-level KPI metrics for dashboard display."""
    if df.empty:
        return {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_savings": 0.0,
            "transaction_count": 0,
        }

    income = df.loc[df["transaction_type"] == "income", "amount"].sum()
    expenses = df.loc[df["transaction_type"] == "expense", "amount"].sum()
    return {
        "total_income": round(float(income), 2),
        "total_expenses": round(float(expenses), 2),
        "net_savings": round(float(income - expenses), 2),
        "transaction_count": len(df),
    }
