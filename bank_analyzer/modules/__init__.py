"""Bank Statement Analyzer - core processing modules."""

from .analytics import (
    AnalyticsService,
    check_budget_limits,
    compute_kpis,
    detect_anomalies,
    get_category_summary,
    get_monthly_trends,
)
from .classifier import TransactionClassifier
from .exporters import ExportStrategyFactory
from .facade import BankAnalyzerFacade
from .ingestion import BankStatementLoader

__all__ = [
    "AnalyticsService",
    "BankAnalyzerFacade",
    "BankStatementLoader",
    "ExportStrategyFactory",
    "TransactionClassifier",
    "check_budget_limits",
    "compute_kpis",
    "detect_anomalies",
    "get_category_summary",
    "get_monthly_trends",
]
