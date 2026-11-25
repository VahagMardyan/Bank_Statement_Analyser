"""The signed-in page: sidebar, KPI row and the tab container."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from modules.currency import ConversionInfo, RatesUnavailableError
from modules.facade import BankAnalyzerFacade
from modules.i18n import DEFAULT_LANGUAGE, Translator
from modules.models import KPIMetrics, User

from .account_panel import AccountPanel
from .common import BASE_DIR, LANGUAGE_STATE_KEY, money, render_language_switch
from .tabs import (
    AdvisorTab,
    AlertsTab,
    BudgetAnomaliesTab,
    DashboardTab,
    RecurringTab,
    TransactionExplorerTab,
    VisualAnalyticsTab,
)

CONVERSION_STATE_KEY = "conversion"  # ConversionInfo of the last loaded statement


class DashboardApp:
    """Top-level page: sidebar, KPIs and the tab container."""

    tab_classes: tuple[type[DashboardTab], ...] = (
        VisualAnalyticsTab,
        TransactionExplorerTab,
        BudgetAnomaliesTab,
        RecurringTab,
        AdvisorTab,
        AlertsTab,
    )

    def __init__(self, user: User, translator: Optional[Translator] = None) -> None:
        self.user = user
        analyzer = st.session_state.get("analyzer")
        # isinstance: after a code change Streamlit reloads modules, but an object
        # kept in session_state still has the old class (and its old methods).
        if (
            not isinstance(analyzer, BankAnalyzerFacade)
            or st.session_state.get("analyzer_user_id") != user.id
        ):
            st.session_state.analyzer = BankAnalyzerFacade.for_user(user.id)
            st.session_state.analyzer_user_id = user.id
            st.session_state.df = st.session_state.analyzer.load_history()
        self.analyzer: BankAnalyzerFacade = st.session_state.analyzer
        self.tr = translator or Translator.from_directory(BASE_DIR / "locales")
        self.tr.language = st.session_state.get(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE)
        self.tabs = [cls(self.analyzer, self.tr) for cls in self.tab_classes]

    def run(self) -> None:
        with st.sidebar:
            render_language_switch(self.tr)
            self._render_account()

        st.title(self.tr.t("app.title"))
        st.caption(self.tr.t("app.caption"))

        with st.sidebar:
            df = self._render_sidebar()

        if df is None or df.empty:
            self._render_welcome()
            return

        self._render_metrics(self.analyzer.compute_kpis(df))
        for tab, container in zip(self.tabs, st.tabs([t.title for t in self.tabs])):
            with container:
                # A fragment: a widget inside this tab (a filter, the slider,
                # the currency picker) reruns only this tab, not all six.
                # Actions that change data call st.rerun(), which reruns
                # the whole page.
                st.fragment(tab.render)(df)

    # sidebar
    def _render_account(self) -> None:
        AccountPanel(self.user, self.analyzer, self.tr).render()

    def _render_sidebar(self) -> Optional[pd.DataFrame]:
        st.header(self.tr.t("sidebar.header"))
        uploaded = st.file_uploader(
            self.tr.t("sidebar.upload"), type=["csv", "xlsx", "xls", "pdf"]
        )
        if uploaded is not None and st.button(self.tr.t("sidebar.load_button"), type="primary"):
            self._load(uploaded)
        self._render_conversion(st.session_state.get(CONVERSION_STATE_KEY))

        df: Optional[pd.DataFrame] = st.session_state.df
        if df is None or df.empty:
            return None
        return self._date_filter(df)

    def _load(self, uploaded) -> None:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        try:
            classified = self.analyzer.load_and_classify(tmp_path)
        except RatesUnavailableError as exc:
            st.error(self.tr.t("currency.no_rates", code=exc.currency))
            return
        except Exception as exc:  # show any parsing problem to the user
            st.error(self.tr.t("sidebar.load_error", error=exc))
            return
        finally:
            tmp_path.unlink(missing_ok=True)

        added = self.analyzer.save_transactions(classified)
        history = self.analyzer.load_history()
        st.session_state.df = history if history is not None else classified
        st.success(
            self.tr.t("sidebar.load_success", count=len(classified), added=added)
        )
        st.session_state[CONVERSION_STATE_KEY] = self.analyzer.last_conversion

    def _render_conversion(self, info: Optional[ConversionInfo]) -> None:
        """Which rates the last foreign statement was converted with."""
        if info is None or not info.converted:
            return
        dates = ", ".join(d.isoformat() for d in info.rate_dates)
        st.info(self.tr.t("currency.converted", code=info.currency, date=dates))
        if info.approximate_rows:
            st.warning(self.tr.t("currency.approximate", count=info.approximate_rows))
        if info.stale:
            st.caption(self.tr.t("currency.stale"))

    def _date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        min_date, max_date = df["date"].min().date(), df["date"].max().date()
        date_range = st.date_input(
            self.tr.t("sidebar.date_range"),
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            if isinstance(start, date) and isinstance(end, date):
                return self.analyzer.apply_date_filter(df, start, end)
        return df

    # sections
    def _render_metrics(self, kpis: KPIMetrics) -> None:
        currency = self.tr.t("meta.currency")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(self.tr.t("metrics.total_income"), f"{money(kpis.total_income)} {currency}")
        c2.metric(self.tr.t("metrics.total_expenses"), f"{money(kpis.total_expenses)} {currency}")
        c3.metric(self.tr.t("metrics.net_savings"), f"{money(kpis.net_savings)} {currency}")
        c4.metric(self.tr.t("metrics.transactions"), kpis.transaction_count)

    def _render_welcome(self) -> None:
        st.info(self.tr.t("welcome.info"))
        features = "\n".join(f"- {item}" for item in self.tr.t_list("welcome.features"))
        st.markdown(f"**{self.tr.t('welcome.features_title')}**\n{features}")
