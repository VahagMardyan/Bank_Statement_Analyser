"""Dashboard tabs: one class per tab, all built on ``DashboardTab``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.advisor import Advice, MonthForecast
from modules.alerts import Alert, AlertKind, alerts_frame
from modules.constants import DEFAULT_BUDGETS
from modules.currency import BASE_CURRENCY, RatesUnavailableError
from modules.exporters import ExportStrategyFactory
from modules.facade import BankAnalyzerFacade
from modules.i18n import Translator
from modules.recurring import PERIODS
from modules.report import MIME as REPORT_MIME
from modules.report import build_workbook

from .common import format_dates, money


def advice_message(tr: Translator, a: Advice) -> str:
    return tr.t(
        f"advisor.{a.kind.value}",
        category=tr.category(a.category),
        spent=money(a.spent),
        projected=money(a.projected),
        reference=money(a.reference),
        excess=money(a.excess),
        pct=int(a.pct_over),
    )


def alert_message(tr: Translator, a: Alert) -> str:
    params = dict(a.params)
    if a.kind is AlertKind.DUPLICATE_CHARGE and params.get("minutes") is None:
        return tr.t("alerts.messages.duplicate_same_day")
    for key in ("before", "threshold"):
        if key in params:
            params[key] = money(params[key])
    return tr.t(f"alerts.messages.{a.kind.value}", amount=money(a.amount), **params)


def alerts_table(tr: Translator, alerts: list[Alert]) -> pd.DataFrame:
    table = alerts_frame(alerts, lambda a: alert_message(tr, a))
    table["kind"] = table["kind"].map(lambda k: tr.t(f"alerts.kinds.{k}"))
    return table


BUDGET_SAVED_KEY = "budget_saved"  # shows "saved" after the full rerun


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
            self.tr.localize(format_dates(df)), width="stretch", hide_index=True
        )

    @abstractmethod
    def render(self, df: pd.DataFrame) -> None:
        """Draw the tab for the (already date-filtered) transactions."""


class VisualAnalyticsTab(DashboardTab):
    title_key = "visual"

    def render(self, df: pd.DataFrame) -> None:
        col1, col2 = st.columns(2)
        with col1:
            self._render_categories(self.analyzer.category_summary(df))
        with col2:
            self._render_trend(self.analyzer.monthly_trends(df))

        st.subheader(self.tr.t("visual.top_expenses"))
        top5 = self.analyzer.top_expenses(df, 5)
        if top5.empty:
            st.info(self.tr.t("visual.no_expenses"))
        else:
            self.table(top5[["date", "description", "category", "amount"]])

    def _render_categories(self, summary: pd.DataFrame) -> None:
        """Expenses by category as horizontal bars, largest on top.

        Bars instead of a pie: lengths are easier to compare than angles, and
        long category names (Armenian ones especially) fit on the axis.
        """
        if summary.empty:
            st.info(self.tr.t("visual.no_expense_data"))
            return
        # Plotly draws the first row at the bottom, so sort ascending.
        data = self.tr.localize(summary, headers=False).sort_values("total_spend")
        data["label"] = [
            f"{money(total)} ({pct:.0f}%)"
            for total, pct in zip(data["total_spend"], data["pct_of_total"])
        ]
        fig = px.bar(
            data,
            x="total_spend",
            y="category",
            orientation="h",
            text="label",
            title=self.tr.t("visual.category_title"),
            labels={"total_spend": self.tr.t("visual.axis_spend"), "category": ""},
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
        )
        fig.update_layout(
            height=max(320, 36 * len(data) + 120),
            # room on the right for the labels of the longest bar
            xaxis_range=[0, float(data["total_spend"].max()) * 1.35],
            margin={"r": 20},
        )
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")


class TransactionExplorerTab(DashboardTab):
    title_key = "explorer"

    COLUMNS = [
        "transaction_id",
        "date",
        "description",
        "category",
        "amount",
        "original_amount",
        "original_currency",
        "transaction_type",
        "confidence",
        "method",
    ]
    DEFAULT_COLUMNS = [
        "transaction_id", "date", "description", "category", "amount", "transaction_type",
    ]
    ALL = "__all__"

    def render(self, df: pd.DataFrame) -> None:
        filtered = self._filter(df)
        options = [c for c in self.COLUMNS if c in df.columns]
        if "original_currency" in df.columns and (df["original_currency"] != BASE_CURRENCY).any():
            st.caption(self.tr.t("explorer.converted_note"))
        col_cols, col_cur = st.columns([3, 1])
        with col_cols:
            selected_cols = st.multiselect(
                self.tr.t("explorer.select_columns"),
                options=options,
                default=[c for c in self.DEFAULT_COLUMNS if c in options],
                format_func=self.tr.column,
            )
        with col_cur:
            currency = st.selectbox(
                self.tr.t("explorer.currency"),
                self.analyzer.display_currencies(),
                format_func=lambda c: f"{c} ({self.tr.t(f'currency.names.{c}')})",
                key="explorer_currency",
            )
        if not selected_cols:
            st.warning(self.tr.t("explorer.no_columns"))
            return

        shown, currency = self._in_currency(filtered, currency)
        display_df = self.tr.localize(format_dates(shown[selected_cols]))
        display_df = display_df.rename(
            columns={self.tr.column("amount"): f"{self.tr.column('amount')} ({currency})"}
        )
        st.dataframe(display_df, width="stretch", hide_index=True)
        if not display_df.empty:
            self._render_export(display_df, currency)

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

    def _in_currency(self, df: pd.DataFrame, currency: str) -> tuple[pd.DataFrame, str]:
        """Amounts in the chosen currency; falls back to AMD without rates."""
        if currency == BASE_CURRENCY:
            return df, currency
        try:
            converted, info = self.analyzer.convert_for_display(df, currency)
        except RatesUnavailableError:
            st.error(self.tr.t("currency.no_rates_display", code=currency))
            return df, BASE_CURRENCY
        if info.rate_dates:
            st.caption(self.tr.t(
                "explorer.currency_note",
                code=currency,
                date=", ".join(d.isoformat() for d in info.rate_dates),
            ))
        if info.approximate_rows:
            st.caption(self.tr.t("explorer.currency_approximate", count=info.approximate_rows))
        if info.stale:
            st.caption(self.tr.t("currency.stale"))
        return converted, currency

    def _render_export(self, display_df: pd.DataFrame, currency: str) -> None:
        """Exports exactly what is on screen: language, columns and currency."""
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
                file_name=(
                    f"{self.tr.t('explorer.export_file_name')}_{currency}{strategy.file_extension}"
                ),
                mime=strategy.mime_type,
                width="stretch",
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
        month_df = self._month_selector(df)
        budgets = self._budget_form()
        self._render_budgets(month_df, budgets)
        self._render_report(df, month_df, budgets)
        st.divider()
        st.subheader(self.tr.t("budget.anomaly_header"))
        self._render_anomalies(df)

    def _month_selector(self, df: pd.DataFrame) -> pd.DataFrame:
        month = st.selectbox(
            self.tr.t("budget.month"),
            self.analyzer.months(df),
            format_func=lambda m: m.strftime("%Y-%m"),
        )
        return self.analyzer.filter_month(df, month)

    def _budget_form(self) -> dict[str, float]:
        saved = self.analyzer.get_budgets()
        values: dict[str, float] = {}
        with st.expander(self.tr.t("budget.edit")):
            with st.form("budget_form"):
                cols = st.columns(3)
                for idx, cat in enumerate(DEFAULT_BUDGETS):
                    with cols[idx % 3]:
                        values[cat] = st.number_input(
                            self.tr.t("budget.input_label", category=self.tr.category(cat)),
                            min_value=0,
                            value=int(saved[cat]),
                            step=1000,
                            key=f"budget_{cat}",  # key stays English: survives a language switch
                        )
                submitted = st.form_submit_button(self.tr.t("budget.save"), type="primary")
        if submitted:
            self.analyzer.save_budgets(values)
            # Whole page again: the Advisor tab uses these limits too.
            st.session_state[BUDGET_SAVED_KEY] = True
            st.rerun()
        if st.session_state.pop(BUDGET_SAVED_KEY, False):
            st.success(self.tr.t("budget.saved"))
        return values

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

    def _render_report(self, df: pd.DataFrame, month_df: pd.DataFrame,
                       budgets: dict[str, float]) -> None:
        month = month_df["date"].iloc[0].strftime("%Y-%m")
        st.download_button(
            self.tr.t("report.button", month=month),
            # A function: the file is only built when the button is clicked.
            data=lambda: build_workbook(self._report_sheets(df, month_df, budgets)),
            file_name=f"{self.tr.t('report.file_name')}_{month}.xlsx",
            mime=REPORT_MIME,
        )

    def _report_sheets(self, df: pd.DataFrame, month_df: pd.DataFrame,
                       budgets: dict[str, float]) -> dict[str, pd.DataFrame]:
        tr, a = self.tr, self.analyzer
        period = month_df["date"].iloc[0].to_period("M")
        kpis = a.compute_kpis(month_df)
        summary = [
            (tr.t("report.month"), period.strftime("%Y-%m")),
            (tr.t("metrics.total_income"), kpis.total_income),
            (tr.t("metrics.total_expenses"), kpis.total_expenses),
            (tr.t("metrics.net_savings"), kpis.net_savings),
            (tr.t("metrics.transactions"), kpis.transaction_count),
        ]
        forecast = a.month_forecast(df)
        advice = []
        if forecast is not None and forecast.month == period:
            summary.append((tr.t("advisor.expected"), forecast.total))
            advice = [advice_message(tr, x).replace("**", "") for x in a.spending_advice(df)]
        sheets = {
            tr.t("report.sheets.summary"): pd.DataFrame(
                summary, columns=[tr.t("report.item"), tr.t("report.value")]),
            tr.t("report.sheets.categories"): tr.localize(a.category_summary(month_df)),
            tr.t("report.sheets.budgets"): tr.localize(a.check_budget_limits(month_df, budgets)),
            tr.t("report.sheets.transactions"): tr.localize(format_dates(month_df[
                ["date", "description", "category", "amount", "transaction_type", "method"]
            ])),
        }
        payments = a.upcoming_payments(df, date.today())
        if not payments.empty:
            payments = payments.drop(columns="id").assign(
                period=payments["period"].map(lambda k: tr.t(f"recurring.periods.{k}")),
                source=payments["source"].map(lambda s: tr.t(f"recurring.sources.{s}")),
            )
            sheets[tr.t("report.sheets.recurring")] = tr.localize(payments)
        month_alerts = [x for x in a.suspicious_activity(df) if x.date.to_period("M") == period]
        if month_alerts:
            sheets[tr.t("report.sheets.alerts")] = tr.localize(format_dates(alerts_table(tr, month_alerts)))
        if advice:
            sheets[tr.t("report.sheets.advice")] = pd.DataFrame({tr.t("report.sheets.advice"): advice})
        return sheets

    def _render_anomalies(self, df: pd.DataFrame) -> None:
        threshold = st.slider(self.tr.t("budget.zscore_threshold"), 1.5, 4.0, 2.5, 0.1)
        anomalies = self.analyzer.detect_anomalies(df, threshold_std=threshold)
        if anomalies.empty:
            st.info(self.tr.t("budget.no_anomalies"))
            return
        self.table(anomalies[["date", "description", "category", "amount", "z_score"]])


class RecurringTab(DashboardTab):
    title_key = "recurring"

    def render(self, df: pd.DataFrame) -> None:
        today = date.today()
        payments = self.analyzer.upcoming_payments(df, today)
        if payments.empty:
            st.info(self.tr.t("recurring.none_found"))
        else:
            view = payments.assign(
                period=payments["period"].map(lambda k: self.tr.t(f"recurring.periods.{k}")),
                days_left=payments["next_due"].map(lambda d: (d - today).days),
                source=payments["source"].map(lambda s: self.tr.t(f"recurring.sources.{s}")),
            )
            self.table(view.drop(columns="id"))
            self._render_dismiss(payments)
        self._render_add_form(today)

    def _render_add_form(self, today: date) -> None:
        with st.expander(self.tr.t("recurring.add_header")):
            st.caption(self.tr.t("recurring.add_help"))
            with st.form("add_recurring_form", clear_on_submit=True):
                name = st.text_input(self.tr.t("recurring.add_name"))
                c1, c2 = st.columns(2)
                category = c1.selectbox(
                    self.tr.t("recurring.add_category"),
                    sorted(self.analyzer.get_categories(), key=self.tr.category),
                    format_func=self.tr.category,
                )
                amount = c2.number_input(
                    self.tr.t("recurring.add_amount"), min_value=0, step=1000, value=0
                )
                period = c1.selectbox(
                    self.tr.t("recurring.add_period"),
                    [p.key for p in PERIODS],
                    index=1,  # monthly
                    format_func=lambda k: self.tr.t(f"recurring.periods.{k}"),
                )
                next_due = c2.date_input(
                    self.tr.t("recurring.add_next_due"),
                    value=today + timedelta(days=7),
                    min_value=today,
                )
                submitted = st.form_submit_button(self.tr.t("recurring.add_button"), type="primary")
        if submitted:
            if not name.strip() or amount <= 0:
                st.error(self.tr.t("recurring.add_error"))
                return
            self.analyzer.add_recurring(name.strip(), category, float(amount), period, next_due)
            st.success(self.tr.t("recurring.add_success"))
            st.rerun()

    def _render_dismiss(self, payments: pd.DataFrame) -> None:
        if payments["id"].isna().all():  # detected on the fly, nothing stored
            return
        labels = dict(zip(payments["id"], payments["name"]))
        st.caption(self.tr.t("recurring.dismiss_help"))
        chosen = st.selectbox(
            self.tr.t("recurring.dismiss_label"), list(labels), format_func=labels.get
        )
        if st.button(self.tr.t("recurring.dismiss_button")):
            self.analyzer.dismiss_recurring(int(chosen))
            st.rerun()


class AdvisorTab(DashboardTab):
    title_key = "advisor"
    STYLE = {"over_budget": st.error, "budget_pace": st.warning, "above_usual": st.info}

    def render(self, df: pd.DataFrame) -> None:
        forecast = self.analyzer.month_forecast(df)
        if forecast is None:
            st.info(self.tr.t("advisor.no_data"))
            return
        self._render_forecast(forecast)
        st.divider()
        st.subheader(self.tr.t("advisor.header_advice"))
        if forecast.days_passed < 7:
            st.caption(self.tr.t("advisor.early", day=forecast.days_passed))
        advice = self.analyzer.spending_advice(df)
        if not advice:
            st.success(self.tr.t("advisor.all_good"))
        for item in advice:
            self.STYLE[item.kind.value](advice_message(self.tr, item))

    def _render_forecast(self, fc: MonthForecast) -> None:
        st.subheader(self.tr.t("advisor.header_forecast", month=fc.month.strftime("%Y-%m")))
        st.caption(self.tr.t(
            "advisor.as_of", date=fc.as_of.isoformat(), day=fc.days_passed, days=fc.days_in_month
        ))
        currency = self.tr.t("meta.currency")
        c1, c2, c3 = st.columns(3)
        c1.metric(self.tr.t("advisor.spent"), f"{money(fc.spent)} {currency}")
        c2.metric(
            self.tr.t("advisor.upcoming", count=fc.upcoming_count),
            f"{money(fc.upcoming_recurring)} {currency}",
        )
        c3.metric(self.tr.t("advisor.expected"), f"{money(fc.total)} {currency}")
        st.caption(self.tr.t("advisor.forecast_help"))



class AlertsTab(DashboardTab):
    title_key = "alerts"

    def render(self, df: pd.DataFrame) -> None:
        st.caption(self.tr.t("alerts.intro"))
        alerts = self.analyzer.suspicious_activity(df)
        if not alerts:
            st.success(self.tr.t("alerts.none"))
            return
        counts = pd.Series([a.kind.value for a in alerts]).value_counts()
        cols = st.columns(len(counts))
        for col, (kind, count) in zip(cols, counts.items()):
            col.metric(self.tr.t(f"alerts.kinds.{kind}"), count)
        self.table(alerts_table(self.tr, alerts))

