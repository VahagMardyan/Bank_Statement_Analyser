"""Streamlit dashboard for Bank Statement Analyzer.
UI structure:
    DashboardApp           - page setup, sidebar, KPI row, tab container
    DashboardTab (ABC)     - one tab; subclasses implement ``render``
      VisualAnalyticsTab, TransactionExplorerTab, BudgetAnomaliesTab

Adding a tab = writing one subclass and adding it to ``DashboardApp.tab_classes``.
All data access goes through :class:`BankAnalyzerFacade`.

There is no UI text in this file: every label comes from ``locales/<lang>.json``
through :class:`modules.i18n.Translator`. Category keys and transaction types
stay English in the data; the translator only changes how they are shown.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.constants import DEFAULT_BUDGETS
from modules.exporters import ExportStrategyFactory
from modules.facade import BankAnalyzerFacade
from modules.i18n import DEFAULT_LANGUAGE, Translator
from modules.models import KPIMetrics

BASE_DIR = Path(__file__).resolve().parent
DATE_DISPLAY_FORMAT = "%Y-%m-%d, %H:%M"
LANGUAGE_STATE_KEY = "language"
FLAG_FONT_PATH = BASE_DIR / "assets" / "fonts" / "TwemojiCountryFlags.woff2"
# Code points of flag emoji only: the font is used for nothing else.
FLAG_UNICODE_RANGE = (
    "U+1F1E6-1F1FF, U+1F3F4, U+E0062-E0063, U+E0065, U+E0067, "
    "U+E006C, U+E006E, U+E0073-E0074, U+E0077, U+E007F"
)


def format_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with the date column formatted for display/export."""
    out = df.copy()
    if "date" in out.columns:
        out["date"] = out["date"].dt.strftime(DATE_DISPLAY_FORMAT)
    return out


def money(value: float) -> str:
    return f"{value:,.0f}"


@lru_cache(maxsize=1)
def flag_font_css() -> str:
    """CSS that makes flag emoji render on every OS.

    Chrome/Edge on Windows have no flag emoji and show "AM" / "GB" instead.
    The bundled Twemoji Country Flags font (78 KB, flags only) fixes that.
    Thanks to ``unicode-range`` the browser uses it only for flags; all other
    text keeps Streamlit's font. Embedded as base64, so it works offline too.
    """
    if not FLAG_FONT_PATH.exists():
        return ""
    data = base64.b64encode(FLAG_FONT_PATH.read_bytes()).decode("ascii")
    return f"""<style>
@font-face {{
  font-family: "Twemoji Country Flags";
  unicode-range: {FLAG_UNICODE_RANGE};
  src: url(data:font/woff2;base64,{data}) format("woff2");
  font-display: swap;
}}
[data-testid="stRadio"] label, [data-testid="stRadio"] label * {{
  font-family: "Twemoji Country Flags", "Source Sans", "Source Sans Pro", sans-serif;
}}
</style>"""


class DashboardTab(ABC):
    """One tab of the dashboard."""

    #: key of the tab title in the locale files (``tabs.<title_key>``)
    title_key: str = ""

    def __init__(self, analyzer: BankAnalyzerFacade, tr: Translator) -> None:
        self.analyzer = analyzer
        self.tr = tr

    @property
    def title(self) -> str:
        return self.tr.t(f"tabs.{self.title_key}")

    def table(self, df: pd.DataFrame) -> None:
        """Show ``df`` with dates formatted and everything translated."""
        st.dataframe(
            self.tr.localize(format_dates(df)), use_container_width=True, hide_index=True
        )

    @abstractmethod
    def render(self, df: pd.DataFrame) -> None:
        """Draw the tab for the (already date-filtered) transactions."""


class VisualAnalyticsTab(DashboardTab):
    title_key = "visual"

    def render(self, df: pd.DataFrame) -> None:
        col1, col2 = st.columns(2)
        with col1:
            self._render_pie(self.analyzer.category_summary(df))
        with col2:
            self._render_trend(self.analyzer.monthly_trends(df))

        st.subheader(self.tr.t("visual.top_expenses"))
        top5 = self.analyzer.top_expenses(df, 5)
        if top5.empty:
            st.info(self.tr.t("visual.no_expenses"))
        else:
            self.table(top5[["date", "description", "category", "amount"]])

    def _render_pie(self, summary: pd.DataFrame) -> None:
        if summary.empty:
            st.info(self.tr.t("visual.no_expense_data"))
            return
        fig = px.pie(
            self.tr.localize(summary, headers=False),
            names="category",
            values="total_spend",
            hole=0.45,
            title=self.tr.t("visual.pie_title"),
            labels={c: self.tr.column(c) for c in ("category", "total_spend")},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    def _render_trend(self, trends: pd.DataFrame) -> None:
        if trends.empty:
            st.info(self.tr.t("visual.no_trend_data"))
            return
        fig = px.bar(
            trends,
            x="period",
            y="total_spend",
            title=self.tr.t("visual.trend_title"),
            labels={
                "total_spend": self.tr.t("visual.axis_spend"),
                "period": self.tr.t("visual.axis_month"),
            },
        )
        st.plotly_chart(fig, use_container_width=True)


class TransactionExplorerTab(DashboardTab):
    title_key = "explorer"

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
    ALL = "__all__"

    def render(self, df: pd.DataFrame) -> None:
        filtered = self._filter(df)
        selected_cols = st.multiselect(
            self.tr.t("explorer.select_columns"),
            options=self.COLUMNS,
            default=self.DEFAULT_COLUMNS,
            format_func=self.tr.column,
        )
        if not selected_cols:
            st.warning(self.tr.t("explorer.no_columns"))
            return

        display_df = self.tr.localize(format_dates(filtered[selected_cols]))
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        if not display_df.empty:
            self._render_export(display_df)

        st.divider()
        st.subheader(self.tr.t("explorer.override_header"))
        if not filtered.empty:
            self._render_override(filtered)

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        categories = [self.ALL] + sorted(
            df["category"].dropna().unique().tolist(), key=self.tr.category
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_cat = st.selectbox(
                self.tr.t("explorer.filter_category"),
                categories,
                format_func=lambda c: (
                    self.tr.t("explorer.all") if c == self.ALL else self.tr.category(c)
                ),
            )
        with col2:
            search = st.text_input(self.tr.t("explorer.search"), "")

        filtered = df
        if selected_cat != self.ALL:
            filtered = filtered[filtered["category"] == selected_cat]
        if search.strip():
            filtered = filtered[
                filtered["description"].str.contains(search, case=False, na=False, regex=False)
                | filtered["cleaned_description"].str.contains(
                    search, case=False, na=False, regex=False
                )
            ]
        return filtered.copy()

    def _render_export(self, display_df: pd.DataFrame) -> None:
        """Exports exactly what is on screen, in the selected language."""
        col_fmt, col_btn = st.columns([1, 2], vertical_alignment="bottom")
        with col_fmt:
            export_format = st.selectbox(
                self.tr.t("explorer.export_format"), ExportStrategyFactory.labels()
            )
        with col_btn:
            strategy = ExportStrategyFactory.create(export_format)
            st.download_button(
                label=self.tr.t("explorer.download", format=export_format),
                data=strategy.export(display_df),
                file_name=f"{self.tr.t('explorer.export_file_name')}{strategy.file_extension}",
                mime=strategy.mime_type,
                use_container_width=True,
            )

    def _render_override(self, filtered: pd.DataFrame) -> None:
        labels = dict(zip(filtered["transaction_id"], filtered["description"].astype(str)))
        txn_id = st.selectbox(
            self.tr.t("explorer.transaction_id"),
            list(labels),
            format_func=lambda x: f"{x} - {labels[x][:50]}",
        )
        new_category = st.selectbox(
            self.tr.t("explorer.new_category"),
            sorted(self.analyzer.get_categories(), key=self.tr.category),
            format_func=self.tr.category,
        )
        if st.button(self.tr.t("explorer.apply_override")):
            st.session_state.df = self.analyzer.override_category(
                st.session_state.df, txn_id, new_category
            )
            st.success(
                self.tr.t("explorer.override_success", category=self.tr.category(new_category))
            )
            st.rerun()


class BudgetAnomaliesTab(DashboardTab):
    title_key = "budget"

    def render(self, df: pd.DataFrame) -> None:
        st.subheader(self.tr.t("budget.header"))
        self._render_budgets(df, self._budget_inputs())
        st.divider()
        st.subheader(self.tr.t("budget.anomaly_header"))
        self._render_anomalies(df)

    def _budget_inputs(self) -> dict[str, float]:
        budgets: dict[str, float] = {}
        cols = st.columns(3)
        for idx, (cat, default) in enumerate(DEFAULT_BUDGETS.items()):
            with cols[idx % 3]:
                budgets[cat] = st.number_input(
                    self.tr.t("budget.input_label", category=self.tr.category(cat)),
                    min_value=0,
                    value=default,
                    step=1000,
                    key=f"budget_{cat}",  # key stays English: survives a language switch
                )
        return budgets

    def _render_budgets(self, df: pd.DataFrame, budgets: dict[str, float]) -> None:
        report = self.analyzer.check_budget_limits(df, budgets)
        if report.empty:
            return
        for _, row in report.iterrows():
            st.write(
                self.tr.t(
                    "budget.progress",
                    category=self.tr.category(row["category"]),
                    actual=money(row["actual"]),
                    budget=money(row["budget"]),
                )
            )
            st.progress(min(row["pct_used"], 100) / 100)
            if row["over_budget"]:
                st.warning(self.tr.t("budget.over_budget", amount=money(abs(row["remaining"]))))
        self.table(report)

    def _render_anomalies(self, df: pd.DataFrame) -> None:
        threshold = st.slider(self.tr.t("budget.zscore_threshold"), 1.5, 4.0, 2.5, 0.1)
        anomalies = self.analyzer.detect_anomalies(df, threshold_std=threshold)
        if anomalies.empty:
            st.info(self.tr.t("budget.no_anomalies"))
            return
        self.table(anomalies[["date", "description", "category", "amount", "z_score"]])


class DashboardApp:
    """Top-level page: sidebar, KPIs and the tab container."""

    tab_classes: tuple[type[DashboardTab], ...] = (
        VisualAnalyticsTab,
        TransactionExplorerTab,
        BudgetAnomaliesTab,
    )

    def __init__(self, translator: Optional[Translator] = None) -> None:
        if "analyzer" not in st.session_state:
            st.session_state.analyzer = BankAnalyzerFacade()
        if "df" not in st.session_state:
            st.session_state.df = None
        self.analyzer: BankAnalyzerFacade = st.session_state.analyzer
        self.tr = translator or Translator.from_directory(BASE_DIR / "locales")
        # The language chosen on the previous run is known before any widget is
        # drawn, so even the page title is in the right language.
        self.tr.language = st.session_state.get(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE)
        self.tabs = [cls(self.analyzer, self.tr) for cls in self.tab_classes]

    def run(self) -> None:
        st.set_page_config(
            page_title=self.tr.t("app.page_title"),
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        with st.sidebar:
            self._render_language_switch()

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
                tab.render(df)

    # sidebar
    def _render_language_switch(self) -> None:
        css = flag_font_css()
        if css:
            st.markdown(css, unsafe_allow_html=True)
        languages = self.tr.languages()
        codes = list(languages)
        current = self.tr.language if self.tr.language in codes else codes[0]
        st.radio(
            self.tr.t("sidebar.language"),
            codes,
            index=codes.index(current),
            format_func=languages.get,
            horizontal=True,
            key=LANGUAGE_STATE_KEY,
        )
        self.tr.language = st.session_state[LANGUAGE_STATE_KEY]

    def _render_sidebar(self) -> Optional[pd.DataFrame]:
        st.header(self.tr.t("sidebar.header"))
        uploaded = st.file_uploader(
            self.tr.t("sidebar.upload"), type=["csv", "xlsx", "xls", "pdf"]
        )
        if uploaded is not None and st.button(self.tr.t("sidebar.load_button"), type="primary"):
            self._load(uploaded)

        df: Optional[pd.DataFrame] = st.session_state.df
        if df is None or df.empty:
            return None
        return self._date_filter(df)

    def _load(self, uploaded) -> None:
        tmp_path = BASE_DIR / "data" / f"_upload{Path(uploaded.name).suffix.lower()}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())
        try:
            classified = self.analyzer.load_and_classify(tmp_path)
        except Exception as exc:  # show any parsing problem to the user
            st.error(self.tr.t("sidebar.load_error", error=exc))
            return
        st.session_state.df = classified
        st.success(self.tr.t("sidebar.load_success", count=len(classified)))

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


def main() -> None:
    DashboardApp().run()


if __name__ == "__main__":
    main()
