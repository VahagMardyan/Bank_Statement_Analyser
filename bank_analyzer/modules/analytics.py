"""Analytics: KPIs, category summary, trends, anomalies and budget checks.

All logic lives in :class:`AnalyticsService`. The module-level functions are
thin aliases kept for scripts/notebooks that imported them before.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .anomaly import AnomalyDetector, ZScoreAnomalyDetector
from .constants import INCOME_CATEGORIES
from .models import KPIMetrics, TransactionType

SUMMARY_COLUMNS = ["category", "total_spend", "pct_of_total", "transaction_count"]
TREND_COLUMNS = ["period", "total_spend"]


class AnalyticsService:
    """Read-only computations over a classified transaction DataFrame.

    The anomaly strategy is injected (Strategy pattern); a z-score detector is
    used when none is given.
    """

    def __init__(self, anomaly_detector: Optional[AnomalyDetector] = None) -> None:
        self._anomaly_detector = anomaly_detector

    # --------------------------------------------------------------- helpers
    @staticmethod
    def expenses(df: pd.DataFrame) -> pd.DataFrame:
        """Expense rows only.

        Raises if ``transaction_type`` is missing: silently treating every row
        as an expense would count income as spending.
        """
        if df.empty:
            return df.copy()
        if "transaction_type" not in df.columns:
            raise ValueError(
                "DataFrame has no 'transaction_type' column; cannot separate "
                "expenses from income."
            )
        return df[df["transaction_type"] == TransactionType.EXPENSE.value].copy()

    # ---------------------------------------------------------------- queries
    def compute_kpis(self, df: pd.DataFrame) -> KPIMetrics:
        if df.empty:
            return KPIMetrics.empty()
        income = df.loc[df["transaction_type"] == TransactionType.INCOME.value, "amount"].sum()
        spent = df.loc[df["transaction_type"] == TransactionType.EXPENSE.value, "amount"].sum()
        return KPIMetrics(
            total_income=round(float(income), 2),
            total_expenses=round(float(spent), 2),
            net_savings=round(float(income - spent), 2),
            transaction_count=len(df),
        )

    def category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Total spend, % of total and count per category (expenses only)."""
        expenses = self.expenses(df)
        if expenses.empty or "category" not in expenses.columns:
            return pd.DataFrame(columns=SUMMARY_COLUMNS)

        grouped = (
            expenses.groupby("category", dropna=False)["amount"]
            .agg(total_spend="sum", transaction_count="count")
            .reset_index()
        )
        total = grouped["total_spend"].sum()
        grouped["pct_of_total"] = np.where(
            total > 0, (grouped["total_spend"] / total * 100).round(2), 0.0
        )
        return grouped.sort_values("total_spend", ascending=False).reset_index(drop=True)

    def monthly_trends(self, df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
        """Expenses resampled by ``freq`` ('ME' month-end, 'W' weekly)."""
        expenses = self.expenses(df)
        if expenses.empty or "date" not in expenses.columns:
            return pd.DataFrame(columns=TREND_COLUMNS)

        work = expenses.assign(date=pd.to_datetime(expenses["date"]))
        resampled = (
            work.set_index("date").sort_index()["amount"].resample(freq).sum().reset_index()
        )
        resampled.columns = TREND_COLUMNS
        resampled["period"] = resampled["period"].dt.strftime("%Y-%m-%d")
        return resampled

    def weekly_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.monthly_trends(df, freq="W")

    def top_expenses(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        return self.expenses(df).nlargest(n, "amount")

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        threshold_std: float = 2.5,
        detector: Optional[AnomalyDetector] = None,
    ) -> pd.DataFrame:
        """Unusually large expenses according to the anomaly strategy.

        Precedence: ``detector`` argument > injected detector > z-score with
        ``threshold_std``.
        """
        strategy = (
            detector
            or self._anomaly_detector
            or ZScoreAnomalyDetector(threshold_std=threshold_std)
        )
        return strategy.detect(self.expenses(df))

    def check_budget_limits(
        self, df: pd.DataFrame, budget_dict: dict[str, float]
    ) -> pd.DataFrame:
        """Actual spend vs. budget caps.

        Columns: category, budget, actual, remaining, pct_used, over_budget.
        """
        summary = self.category_summary(df)
        actual_by_cat: dict[str, float] = (
            dict(zip(summary["category"], summary["total_spend"].astype(float)))
            if not summary.empty
            else {}
        )
        categories = sorted((set(actual_by_cat) | set(budget_dict)) - INCOME_CATEGORIES)

        rows: list[dict[str, Any]] = []
        for category in categories:
            budget = float(budget_dict.get(category, 0.0))
            actual = actual_by_cat.get(category, 0.0)
            rows.append(
                {
                    "category": category,
                    "budget": budget,
                    "actual": round(actual, 2),
                    "remaining": round(budget - actual, 2),
                    "pct_used": round(actual / budget * 100, 2) if budget > 0 else 0.0,
                    "over_budget": actual > budget if budget > 0 else False,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["category", "budget", "actual", "remaining", "pct_used", "over_budget"]
            )
        return pd.DataFrame(rows).sort_values("pct_used", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------
# Backward-compatible function aliases
# ------------------------------------------------------------------------
_DEFAULT = AnalyticsService()


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _DEFAULT.category_summary(df)


def get_monthly_trends(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    return _DEFAULT.monthly_trends(df, freq)


def get_weekly_trends(df: pd.DataFrame) -> pd.DataFrame:
    return _DEFAULT.weekly_trends(df)


def detect_anomalies(
    df: pd.DataFrame,
    threshold_std: float = 2.5,
    detector: Optional[AnomalyDetector] = None,
) -> pd.DataFrame:
    return _DEFAULT.detect_anomalies(df, threshold_std, detector)


def check_budget_limits(df: pd.DataFrame, budget_dict: dict[str, float]) -> pd.DataFrame:
    return _DEFAULT.check_budget_limits(df, budget_dict)


def compute_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    return _DEFAULT.compute_kpis(df).as_dict()
