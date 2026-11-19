"""Spending advisor: where this month's spending is heading, and what to cut.

Everything is computed from the stored history; nothing is saved. "This month"
is the month of the latest transaction, not the calendar month, because
statements are usually uploaded days or weeks late.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Iterable, Optional

import pandas as pd

from .constants import INCOME_CATEGORIES
from .models import TransactionType

# Money moved, not spent: advice about them would only be noise.
NOT_SPENDING = frozenset({"Transfer", "ATM"})


class AdviceKind(str, Enum):
    OVER_BUDGET = "over_budget"      # already above the monthly limit
    BUDGET_PACE = "budget_pace"      # not over yet, but will be at this pace
    ABOVE_USUAL = "above_usual"      # well above the average of earlier months


@dataclass(frozen=True)
class Advice:
    kind: AdviceKind
    category: str
    spent: float          # so far this month
    projected: float      # expected by the end of the month at the current pace
    reference: float      # the budget, or the usual monthly spend for ABOVE_USUAL
    excess: float         # how much over the reference (projected or already spent)

    @property
    def pct_over(self) -> float:
        return round(self.excess / self.reference * 100) if self.reference else 0.0


@dataclass(frozen=True)
class MonthForecast:
    month: pd.Period
    as_of: date
    days_passed: int
    days_in_month: int
    spent: float               # all spending so far this month
    variable_projection: float  # non-recurring spending expected in the remaining days
    upcoming_recurring: float   # recurring payments still due this month
    upcoming_count: int

    @property
    def total(self) -> float:
        return round(self.spent + self.variable_projection + self.upcoming_recurring, 2)


@dataclass(frozen=True)
class MonthStatus:
    month: pd.Period
    as_of: date
    days_passed: int
    days_in_month: int
    spent_by_category: dict[str, float]
    fixed_by_category: dict[str, float]   # recurring payments and one-offs: counted once
    usual_by_category: dict[str, float]   # average of the earlier months that have data
    history_months: int

    @property
    def pace(self) -> float:
        """Multiply everyday spend-so-far by this to get the end-of-month estimate."""
        return self.days_in_month / self.days_passed if self.days_passed else 1.0

    def projected(self, category: str) -> float:
        """Recurring payments counted once; only everyday spending follows the pace.

        Otherwise a quarterly insurance paid on the 10th would be "projected"
        to double by the end of the month.
        """
        spent = self.spent_by_category.get(category, 0.0)
        fixed = self.fixed_by_category.get(category, 0.0)
        return round(fixed + (spent - fixed) * self.pace, 2)


class SpendingAdvisor:
    def __init__(
        self,
        history_months: int = 3,
        above_usual_ratio: float = 1.25,
        min_excess: float = 5000.0,
        min_days_for_pace: int = 7,
        one_off_percentile: float = 0.95,
        min_one_off: float = 30000.0,
    ) -> None:
        """
        history_months:     how many earlier months make up "usual" spending
        above_usual_ratio:  1.25 = warn when a category is 25% above usual
        min_excess:         ignore differences smaller than this (in AMD)
        min_days_for_pace:  before this day of the month a projection is too noisy
        one_off_percentile, min_one_off: a single payment above both the 95th
                            percentile of all payments and this amount is a
                            one-off (a laptop, a ticket) and isn't extended
                            at the daily pace
        """
        self.history_months = history_months
        self.above_usual_ratio = above_usual_ratio
        self.min_excess = min_excess
        self.min_days_for_pace = min_days_for_pace
        self.one_off_percentile = one_off_percentile
        self.min_one_off = min_one_off

    # month status
    def status(
        self, df: Optional[pd.DataFrame], recurring_names: Iterable[str] = ()
    ) -> Optional[MonthStatus]:
        spending = self._spending(df)
        if spending.empty:
            return None
        as_of = df["date"].max().date()
        month = pd.Period(as_of, freq="M")
        months = spending["date"].dt.to_period("M")

        current = spending[months == month]
        spent = current.groupby("category")["amount"].sum().round(2).to_dict()
        fixed_rows = current[self._counted_once(current, spending, recurring_names)]
        fixed = fixed_rows.groupby("category")["amount"].sum().round(2).to_dict()

        earlier = sorted({m for m in months.unique() if m < month}, reverse=True)
        earlier = earlier[: self.history_months]
        usual: dict[str, float] = {}
        if earlier:
            past = spending[months.isin(earlier)]
            usual = (past.groupby("category")["amount"].sum() / len(earlier)).round(2).to_dict()

        return MonthStatus(
            month=month,
            as_of=as_of,
            days_passed=as_of.day,
            days_in_month=calendar.monthrange(as_of.year, as_of.month)[1],
            spent_by_category=spent,
            fixed_by_category=fixed,
            usual_by_category=usual,
            history_months=len(earlier),
        )

    # advice
    def advise(
        self,
        df: Optional[pd.DataFrame],
        budgets: dict[str, float],
        recurring_names: Iterable[str] = (),
    ) -> list[Advice]:
        status = self.status(df, recurring_names)
        if status is None:
            return []
        use_pace = status.days_passed >= self.min_days_for_pace
        advice: list[Advice] = []
        for category in sorted(set(status.spent_by_category) | set(budgets)):
            if category in INCOME_CATEGORIES or category in NOT_SPENDING:
                continue
            spent = status.spent_by_category.get(category, 0.0)
            projected = status.projected(category) if use_pace else spent
            item = self._budget_advice(category, spent, projected, budgets.get(category, 0.0))
            if item is None:
                item = self._usual_advice(category, spent, projected, status, use_pace)
            if item is not None:
                advice.append(item)
        order = {AdviceKind.OVER_BUDGET: 0, AdviceKind.BUDGET_PACE: 1, AdviceKind.ABOVE_USUAL: 2}
        return sorted(advice, key=lambda a: (order[a.kind], -a.excess))

    def _budget_advice(
        self, category: str, spent: float, projected: float, budget: float
    ) -> Optional[Advice]:
        if budget <= 0:
            return None
        if spent > budget:
            return Advice(AdviceKind.OVER_BUDGET, category, spent, projected, budget,
                          round(spent - budget, 2))
        if projected - budget >= self.min_excess:
            return Advice(AdviceKind.BUDGET_PACE, category, spent, projected, budget,
                          round(projected - budget, 2))
        return None

    def _usual_advice(
        self, category: str, spent: float, projected: float, status: MonthStatus, use_pace: bool
    ) -> Optional[Advice]:
        usual = status.usual_by_category.get(category, 0.0)
        if usual <= 0 or not use_pace:
            return None
        excess = projected - usual
        if projected >= usual * self.above_usual_ratio and excess >= self.min_excess:
            return Advice(AdviceKind.ABOVE_USUAL, category, spent, projected, usual,
                          round(excess, 2))
        return None

    # forecast
    def forecast(
        self, df: Optional[pd.DataFrame], recurring: Optional[pd.DataFrame] = None
    ) -> Optional[MonthForecast]:
        """End-of-month estimate: spent so far + pace of everyday spending
        + recurring payments still due this month.

        Recurring payments are taken out of the pace part, otherwise a rent
        paid on the 1st would be "projected" again for every remaining day.
        """
        recurring = recurring if recurring is not None else pd.DataFrame()
        names = set(recurring["name"]) if not recurring.empty else set()
        status = self.status(df, names)
        if status is None:
            return None
        spending = self._spending(df)
        current = spending[spending["date"].dt.to_period("M") == status.month]
        everyday = current[~self._counted_once(current, spending, names)]
        days_left = status.days_in_month - status.days_passed
        per_day = everyday["amount"].sum() / status.days_passed if status.days_passed else 0.0
        variable = round(per_day * days_left, 2) if status.days_passed >= self.min_days_for_pace else 0.0

        upcoming_total, upcoming_count = 0.0, 0
        if not recurring.empty:
            month_end = status.month.end_time.date()
            for _, row in recurring.iterrows():
                due = row["next_due"]
                if status.as_of < due <= month_end and pd.notna(row["expected_amount"]):
                    upcoming_total += float(row["expected_amount"])
                    upcoming_count += 1

        return MonthForecast(
            month=status.month,
            as_of=status.as_of,
            days_passed=status.days_passed,
            days_in_month=status.days_in_month,
            spent=round(float(current["amount"].sum()), 2),
            variable_projection=variable,
            upcoming_recurring=round(upcoming_total, 2),
            upcoming_count=upcoming_count,
        )

    def _counted_once(
        self, rows: pd.DataFrame, spending: pd.DataFrame, recurring_names: Iterable[str]
    ) -> pd.Series:
        """Rows that must not be extended at the daily pace."""
        threshold = max(self.min_one_off, float(spending["amount"].quantile(self.one_off_percentile)))
        return rows["cleaned_description"].isin(set(recurring_names)) | (rows["amount"] >= threshold)

    @staticmethod
    def _spending(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "category", "amount", "cleaned_description"])
        out = df[df["transaction_type"] == TransactionType.EXPENSE.value]
        return out[~out["category"].isin(INCOME_CATEGORIES | NOT_SPENDING)]
