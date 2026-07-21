"""Bank Statement Analyzer - core processing modules."""

from .ingestion import BankStatementLoader
from .classifier import TransactionClassifier
from .analytics import (
    get_category_summary,
    get_monthly_trends,
    detect_anomalies,
    check_budget_limits,
)

__all__ = [
    "BankStatementLoader",
    "TransactionClassifier",
    "get_category_summary",
    "get_monthly_trends",
    "detect_anomalies",
    "check_budget_limits",
]
