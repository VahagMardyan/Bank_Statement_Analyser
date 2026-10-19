"""Streamlit dashboard for Bank Statement Analyzer.
UI structure:
    DashboardApp           - page setup, sidebar, KPI row, tab container
    DashboardTab (ABC)     - one tab; subclasses implement ``render``
      VisualAnalyticsTab, TransactionExplorerTab, BudgetAnomaliesTab

Adding a tab = writing one subclass and adding it to ``DashboardApp.tabs``.
All data access goes through :class:`BankAnalyzerFacade`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.constants import DEFAULT_BUDGETS
from modules.exporters import ExportStrategyFactory
from modules.facade import BankAnalyzerFacade
from modules.models import KPIMetrics

BASE_DIR = Path(__file__).resolve().parent
DATE_DISPLAY_FORMAT = "%Y-%m-%d, %H:%M"


def format_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with the date column formatted for display/export."""
    out = df.copy()
    if "date" in out.columns:
        out["date"] = out["date"].dt.strftime(DATE_DISPLAY_FORMAT)
    return out


class DashboardTab(ABC):
    """One tab of the dashboard."""

    title: str = ""

    def __init__(self, analyzer: BankAnalyzerFacade) -> None:
        self.analyzer = analyzer

    @abstractmethod
    def render(self, df: pd.DataFrame) -> None:
        """Draw the tab for the (already date-filtered) transactions."""


class VisualAnalyticsTab(DashboardTab):
    title = "Visual Analytics"

    def render(self, df: pd.DataFrame) -> None:
        col1, col2 = st.columns(2)
        with col1:
            self._render_pie(self.analyzer.category_summary(df))
        with col2:
            self._render_trend(self.analyzer.monthly_trends(df))

        st.subheader("Top 5 Largest Expenses")
        top5 = self.analyzer.top_expenses(df, 5)
        if top5.empty:
            st.info("No expenses found.")
        else:
            st.dataframe(
                format_dates(top5[["date", "description", "category", "amount"]]),
                use_container_width=True,
                hide_index=True,
            )

    @staticmethod
    def _render_pie(summary: pd.DataFrame) -> None:
        if summary.empty:
            st.info("No expense data to display.")
            return
        fig = px.pie(
            summary,
            names="category",
            values="total_spend",
            hole=0.45,
            title="Expense Distribution by Category",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_trend(trends: pd.DataFrame) -> None:
        if trends.empty:
            st.info("No trend data available.")
            return
        fig = px.bar(
            trends,
            x="period",
            y="total_spend",
            title="Monthly Spending Trends",
            labels={"total_spend": "Spend (AMD)", "period": "Month"},
        )
        st.plotly_chart(fig, use_container_width=True)


class TransactionExplorerTab(DashboardTab):
    title = "Transaction Explorer"

    COLUMNS = [
        "transaction_id",
        "date",
        "description",
        "category",
        "amount",
        "transaction_type",
        "confidence",
        "method",
    ]
    DEFAULT_COLUMNS = COLUMNS[:-2]

    def render(self, df: pd.DataFrame) -> None:
        filtered = self._filter(df)
        selected_cols = st.multiselect(
            "Select Columns to Display & Export",
            options=self.COLUMNS,
            default=self.DEFAULT_COLUMNS,
        )
        if not selected_cols:
            st.warning("Please select at least one column to display and export.")
            return

        display_df = format_dates(filtered)
        st.dataframe(display_df[selected_cols], use_container_width=True, hide_index=True)
        if not display_df.empty:
            self._render_export(display_df, selected_cols)

        st.divider()
        st.subheader("Manual Category Override")
        if not filtered.empty:
            self._render_override(filtered)

    @staticmethod
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_cat = st.selectbox("Filter by Category", categories)
        with col2:
            search = st.text_input("Search Description", "")

        filtered = df
        if selected_cat != "All":
            filtered = filtered[filtered["category"] == selected_cat]
        if search.strip():
            filtered = filtered[
                filtered["description"].str.contains(search, case=False, na=False, regex=False)
                | filtered["cleaned_description"].str.contains(
                    search, case=False, na=False, regex=False
                )
            ]
        return filtered.copy()

    @staticmethod
    def _render_export(display_df: pd.DataFrame, columns: list[str]) -> None:
        col_fmt, col_btn = st.columns([1, 2], vertical_alignment="bottom")
        with col_fmt:
            export_format = st.selectbox("Export Format", ExportStrategyFactory.labels())
        with col_btn:
            strategy = ExportStrategyFactory.create(export_format)
            st.download_button(
                label=f"📥 Download as {export_format}",
                data=strategy.export(display_df, columns=columns),
                file_name=f"transactions_export{strategy.file_extension}",
                mime=strategy.mime_type,
                use_container_width=True,
            )

    def _render_override(self, filtered: pd.DataFrame) -> None:
        labels = dict(zip(filtered["transaction_id"], filtered["description"].astype(str)))
        txn_id = st.selectbox(
            "Transaction ID",
            list(labels),
            format_func=lambda x: f"{x} - {labels[x][:50]}",
        )
        new_category = st.selectbox("New Category", self.analyzer.get_categories())
        if st.button("Apply Override"):
            st.session_state.df = self.analyzer.override_category(
                st.session_state.df, txn_id, new_category
            )
            st.success(
                f"Category updated to '{new_category}'. Rules/dataset updated for retraining."
            )
            st.rerun()


class BudgetAnomaliesTab(DashboardTab):
    title = "Budget & Anomalies"

    def render(self, df: pd.DataFrame) -> None:
        st.subheader("Budget Limits")
        self._render_budgets(df, self._budget_inputs())
        st.divider()
        st.subheader("Anomaly Detection")
        self._render_anomalies(df)

    @staticmethod
    def _budget_inputs() -> dict[str, float]:
        budgets: dict[str, float] = {}
        cols = st.columns(3)
        for idx, (cat, default) in enumerate(DEFAULT_BUDGETS.items()):
            with cols[idx % 3]:
                budgets[cat] = st.number_input(
                    f"{cat} Budget (AMD)",
                    min_value=0,
                    value=default,
                    step=1000,
                    key=f"budget_{cat}",
                )
        return budgets

    def _render_budgets(self, df: pd.DataFrame, budgets: dict[str, float]) -> None:
        report = self.analyzer.check_budget_limits(df, budgets)
        if report.empty:
            return
        for _, row in report.iterrows():
            st.write(f"**{row['category']}** — {row['actual']:,.0f} / {row['budget']:,.0f} AMD")
            st.progress(min(row["pct_used"], 100) / 100)
            if row["over_budget"]:
                st.warning(f"Over budget by {abs(row['remaining']):,.0f} AMD")
        st.dataframe(report, use_container_width=True, hide_index=True)

    def _render_anomalies(self, df: pd.DataFrame) -> None:
        threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
        anomalies = self.analyzer.detect_anomalies(df, threshold_std=threshold)
        if anomalies.empty:
            st.info("No anomalies detected at this threshold.")
            return
        st.dataframe(
            format_dates(anomalies[["date", "description", "category", "amount", "z_score"]]),
            use_container_width=True,
            hide_index=True,
        )


class DashboardApp:
    """Top-level page: sidebar, KPIs and the tab container."""

    tab_classes: tuple[type[DashboardTab], ...] = (
        VisualAnalyticsTab,
        TransactionExplorerTab,
        BudgetAnomaliesTab,
    )

    def __init__(self) -> None:
        if "analyzer" not in st.session_state:
            st.session_state.analyzer = BankAnalyzerFacade()
        if "df" not in st.session_state:
            st.session_state.df = None
        self.analyzer: BankAnalyzerFacade = st.session_state.analyzer
        self.tabs = [cls(self.analyzer) for cls in self.tab_classes]

    def run(self) -> None:
        st.set_page_config(
            page_title="Bank Statement Analyzer",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.title("🏦 Bank Statement Analyzer")
        st.caption(
            "Parse, categorize, and visualize bank statements from Armenian and "
            "international banks."
        )

        with st.sidebar:
            df = self._render_sidebar()

        if df is None or df.empty:
            self._render_welcome()
            return

        self._render_metrics(self.analyzer.compute_kpis(df))
        for tab, container in zip(self.tabs, st.tabs([t.title for t in self.tabs])):
            with container:
                tab.render(df)

    # --------------------------------------------------------------- sidebar
    def _render_sidebar(self) -> Optional[pd.DataFrame]:
        st.header("Configuration")
        uploaded = st.file_uploader(
            "Upload Bank Statement (CSV / Excel / PDF)",
            type=["csv", "xlsx", "xls", "pdf"],
        )
        bank_labels = self.analyzer.bank_labels()
        selected_bank = st.selectbox(
            "Bank",
            ["auto_detect"] + self.analyzer.list_banks(),
            format_func=lambda x: bank_labels.get(x, x),
        )
        if uploaded is not None and st.button("Load & Analyze", type="primary"):
            self._load(uploaded, selected_bank)

        df: Optional[pd.DataFrame] = st.session_state.df
        if df is None or df.empty:
            return None
        return self._date_filter(df)

    def _load(self, uploaded, bank_key: str) -> None:
        tmp_path = BASE_DIR / "data" / f"_upload{Path(uploaded.name).suffix.lower()}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())
        try:
            classified = self.analyzer.load_and_classify(tmp_path, bank_key=bank_key)
        except Exception as exc:  # show any parsing problem to the user
            st.error(f"Failed to load statement: {exc}")
            return
        st.session_state.df = classified
        st.success(f"Loaded {len(classified)} transactions.")

    def _date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        min_date, max_date = df["date"].min().date(), df["date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            if isinstance(start, date) and isinstance(end, date):
                return self.analyzer.apply_date_filter(df, start, end)
        return df

    # ------------------------------------------------------------- sections
    @staticmethod
    def _render_metrics(kpis: KPIMetrics) -> None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Income", f"{kpis.total_income:,.0f} AMD")
        c2.metric("Total Expenses", f"{kpis.total_expenses:,.0f} AMD")
        c3.metric("Net Savings", f"{kpis.net_savings:,.0f} AMD")
        c4.metric("Transactions", kpis.transaction_count)

    @staticmethod
    def _render_welcome() -> None:
        st.info("Upload a statement file (CSV, XLSX, XLS, PDF) using the sidebar to get started.")
        st.markdown(
            """
            **Supported features:**
            - Multi-bank parsing (Ameriabank, ACBA, Inecobank, Evocabank, auto-detect)
            - Supports CSV, XLSX, XLS, and PDF formats
            - Hybrid rule + ML categorization
            - Interactive charts, budget tracking, and export functionality
            """
        )


def main() -> None:
    DashboardApp().run()


if __name__ == "__main__":
    main()
