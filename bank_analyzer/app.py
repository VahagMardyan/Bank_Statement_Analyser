"""Streamlit dashboard for Bank Statement Analyzer.
UI structure:
    DashboardApp           - page setup, sidebar, KPI row, tab container
    DashboardTab (ABC)     - one tab; subclasses implement ``render``
      VisualAnalyticsTab, TransactionExplorerTab, BudgetAnomaliesTab, RecurringTab
    AuthPage               - sign-in / sign-up, shown when no user is in session
    AccountPanel           - sidebar: who is signed in, account settings, sign out

Adding a tab = writing one subclass and adding it to ``DashboardApp.tab_classes``.
All data access goes through :class:`BankAnalyzerFacade`.

There is no UI text in this file: every label comes from ``locales/<lang>.json``
through :class:`modules.i18n.Translator`. Category keys and transaction types
stay English in the data; the translator only changes how they are shown.
"""

from __future__ import annotations

import base64
import re
import tempfile
from abc import ABC, abstractmethod
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.constants import DEFAULT_BUDGETS
from modules.exporters import ExportStrategyFactory
from modules.facade import BankAnalyzerFacade
from modules.i18n import DEFAULT_LANGUAGE, Translator
from modules.models import KPIMetrics, User
from modules.remember import REMEMBER_DAYS, RememberMeService
from modules.users import AuthError, AuthService

USER_STATE_KEY = 'user'
FLASH_STATE_KEY = "flash"  # locale key of a success message shown after st.rerun()
REMEMBER_STATE_KEY = "remember"  # this session came from / created a remember-me cookie
COOKIE_STATE_KEY = "pending_cookie"  # cookie value to write on the next run ("" = delete)
COOKIE_CHECKED_KEY = "cookie_checked"
REMEMBER_COOKIE = "bsa_remember"
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


def md_escape(value: str) -> str:
    """User text inside st.markdown: show ``*``/``_`` literally, not as formatting."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~])", r"\\\1", value)


def show_flash(tr: Translator) -> None:
    key = st.session_state.pop(FLASH_STATE_KEY, None)
    if key:
        st.success(tr.t(key))


def reset_session(flash: Optional[str] = None) -> None:
    """Forget the user and their data; keep only the chosen language."""
    language = st.session_state.get(LANGUAGE_STATE_KEY)
    st.session_state.clear()
    if language:
        st.session_state[LANGUAGE_STATE_KEY] = language
    if flash:
        st.session_state[FLASH_STATE_KEY] = flash


def remember_cookie() -> Optional[str]:
    """The remember-me cookie as sent when the page was opened."""
    return st.context.cookies.get(REMEMBER_COOKIE)


def write_pending_cookie() -> None:
    """Set or delete the cookie in the browser.

    Python can only *read* cookies (st.context.cookies), so writing goes
    through a tiny script. It runs on the run after the one that asked for
    it, because a script added right before st.rerun() would never reach the
    browser.
    """
    value = st.session_state.pop(COOKIE_STATE_KEY, None)
    if value is None:
        return
    max_age = REMEMBER_DAYS * 24 * 3600 if value else 0
    st.html(
        "<script>document.cookie = "
        f"'{REMEMBER_COOKIE}={value}; path=/; max-age={max_age}; SameSite=Lax'"
        " + (location.protocol === 'https:' ? '; Secure' : '');</script>",
        unsafe_allow_javascript=True,
    )


def sign_out(remember: Optional[RememberMeService] = None, flash: Optional[str] = None) -> None:
    """Revoke this browser's token, clear the session, delete the cookie."""
    if remember is not None:
        remember.revoke(remember_cookie())
    reset_session(flash)
    st.session_state[COOKIE_STATE_KEY] = ""
    st.session_state[COOKIE_CHECKED_KEY] = True  # don't auto-login from the old cookie


def render_language_switch(tr: Translator) -> None:
    """Language radio, shared by the login page and the dashboard."""
    css = flag_font_css()
    if css:
        st.markdown(css, unsafe_allow_html=True)
    languages = tr.languages()
    codes = list(languages)
    options = {}
    # After sign-out the language is put back into session_state by code;
    # passing a default index as well would make Streamlit show a warning.
    if LANGUAGE_STATE_KEY not in st.session_state:
        options["index"] = codes.index(tr.language) if tr.language in codes else 0
    st.radio(
        tr.t("sidebar.language"),
        codes,
        format_func=languages.get,
        horizontal=True,
        key=LANGUAGE_STATE_KEY,
        **options,
    )
    tr.language = st.session_state[LANGUAGE_STATE_KEY]


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

class RecurringTab(DashboardTab):
    title_key = "recurring"

    def render(self, df: pd.DataFrame) -> None:
        payments = self.analyzer.recurring_payments(df)
        if payments.empty:
            st.info(self.tr.t("recurring.none_found"))
            return
        today = date.today()
        view = payments.assign(
            period=payments["period"].map(lambda k: self.tr.t(f"recurring.periods.{k}")),
            days_left=payments["next_due"].map(lambda d: (d - today).days),
        )
        self.table(view.drop(columns="id"))
        self._render_dismiss(payments)

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


class AuthPage:
    """Sign-in and sign-up forms shown before the dashboard."""
    def __init__(
        self,
        translator: Translator,
        auth: Optional[AuthService] = None,
        remember: Optional[RememberMeService] = None,
    ) -> None:
        self.tr = translator
        self.auth = auth or AuthService()
        self.remember = remember or RememberMeService()

    def render(self) -> None:
        with st.sidebar:
            render_language_switch(self.tr)
        st.title(self.tr.t('app.title'))
        show_flash(self.tr)
        login_tab, register_tab = st.tabs(
            [self.tr.t('auth.login_tab'), self.tr.t('auth.register_tab')]
        )
        with login_tab:
            self._render_login()
        with register_tab:
            self._render_register()

    def _render_login(self) -> None:
        with st.form('login_form'):
            email = st.text_input(self.tr.t("auth.email"))
            password = st.text_input(self.tr.t("auth.password"), type='password')
            keep = st.checkbox(self.tr.t("auth.remember_me"), value=True)
            submitted = st.form_submit_button(self.tr.t("auth.login_button"), type='primary')
        if submitted:
            self._sign_in(lambda: self.auth.login(email, password), keep)

    def _render_register(self) -> None:
        with st.form('register_form'):
            name = st.text_input(self.tr.t('auth.display_name'))
            email = st.text_input(self.tr.t('auth.email'))
            password = st.text_input(self.tr.t('auth.password'), type='password')
            keep = st.checkbox(self.tr.t("auth.remember_me"), value=True, key="register_keep")
            submitted = st.form_submit_button(self.tr.t('auth.register_button'), type='primary')
        if submitted:
            self._sign_in(lambda: self.auth.register(email, password, name), keep)

    def _sign_in(self, action: Callable[[], User], keep_signed_in: bool) -> None:
        try:
            user = action()
        except AuthError as exc:
            st.error(self.tr.t(exc.key))
            return
        st.session_state[USER_STATE_KEY] = user
        if keep_signed_in:
            st.session_state[COOKIE_STATE_KEY] = self.remember.issue(user.id)
            st.session_state[REMEMBER_STATE_KEY] = True
        st.rerun()

    def try_cookie_login(self) -> Optional[User]:
        """Sign in from the remember-me cookie, once per browser session."""
        if st.session_state.get(COOKIE_CHECKED_KEY):
            return None
        st.session_state[COOKIE_CHECKED_KEY] = True
        user_id = self.remember.resolve(remember_cookie())
        user = self.auth.user_by_id(user_id) if user_id is not None else None
        if user is not None:
            st.session_state[USER_STATE_KEY] = user
            st.session_state[REMEMBER_STATE_KEY] = True
        return user
        
class AccountPanel:
    """Sidebar block: current user, account settings and sign out."""

    def __init__(
        self,
        user: User,
        analyzer: BankAnalyzerFacade,
        translator: Translator,
        auth: Optional[AuthService] = None,
        remember: Optional[RememberMeService] = None,
    ) -> None:
        self.user = user
        self.analyzer = analyzer
        self.tr = translator
        self.auth = auth or AuthService()
        self.remember = remember or RememberMeService()

    def render(self) -> None:
        st.caption(self.tr.t("auth.signed_in_as"))
        st.markdown(f"**{md_escape(self.user.display_name)}**  \n{md_escape(self.user.email)}")
        show_flash(self.tr)
        if st.button(self.tr.t("auth.logout_button")):
            sign_out(self.remember)
            st.rerun()
        with st.expander(self.tr.t("account.header")):
            self._render_rename()
            st.divider()
            self._render_password()
            st.divider()
            self._render_clear_history()
            st.divider()
            self._render_delete_account()

    def _render_rename(self) -> None:
        with st.form("rename_form"):
            st.markdown(f"**{self.tr.t('account.rename_header')}**")
            name = st.text_input(self.tr.t("account.new_name"), value=self.user.display_name)
            submitted = st.form_submit_button(self.tr.t("account.rename_button"))
        if submitted:
            self._run(lambda: self._rename(name), "account.rename_success")

    def _render_password(self) -> None:
        with st.form("password_form", clear_on_submit=True):
            st.markdown(f"**{self.tr.t('account.password_header')}**")
            current = st.text_input(self.tr.t("account.current_password"), type="password")
            new = st.text_input(self.tr.t("account.new_password"), type="password")
            submitted = st.form_submit_button(self.tr.t("account.password_button"))
        if submitted:
            self._run(lambda: self._change_password(current, new), "account.password_success")

    def _render_clear_history(self) -> None:
        with st.form("clear_history_form"):
            st.markdown(f"**{self.tr.t('account.clear_header')}**")
            st.caption(self.tr.t("account.clear_help"))
            confirmed = st.checkbox(self.tr.t("account.confirm"))
            submitted = st.form_submit_button(self.tr.t("account.clear_button"))
        if submitted:
            self._run(lambda: self._clear_history(confirmed), "account.clear_success")

    def _render_delete_account(self) -> None:
        with st.form("delete_account_form", clear_on_submit=True):
            st.markdown(f"**{self.tr.t('account.delete_header')}**")
            st.caption(self.tr.t("account.delete_help"))
            password = st.text_input(self.tr.t("account.current_password"), type="password")
            confirmed = st.checkbox(self.tr.t("account.confirm"))
            submitted = st.form_submit_button(self.tr.t("account.delete_button"), type="primary")
        if submitted:
            try:
                self._require(confirmed)
                self.auth.delete_account(self.user, password)
            except AuthError as exc:
                st.error(self.tr.t(exc.key))
                return
            sign_out(flash="account.delete_success")  # tokens went with the user (CASCADE)
            st.rerun()

    # actions
    def _rename(self, name: str) -> None:
        st.session_state[USER_STATE_KEY] = self.auth.rename(self.user, name)

    def _change_password(self, current: str, new: str) -> None:
        self.auth.change_password(self.user, current, new)
        # Every "keep me signed in" device is signed out; this one gets a new token.
        self.remember.revoke_all(self.user.id)
        if st.session_state.get(REMEMBER_STATE_KEY):
            st.session_state[COOKIE_STATE_KEY] = self.remember.issue(self.user.id)

    def _clear_history(self, confirmed: bool) -> None:
        self._require(confirmed)
        self.analyzer.clear_history()
        st.session_state.df = None

    @staticmethod
    def _require(confirmed: bool) -> None:
        if not confirmed:
            raise AuthError("account.confirm_required")

    def _run(self, action: Callable[[], None], success_key: str) -> None:
        """Run an action; on success, rerun and show ``success_key``."""
        try:
            action()
        except AuthError as exc:
            st.error(self.tr.t(exc.key))
            return
        st.session_state[FLASH_STATE_KEY] = success_key
        st.rerun()


class DashboardApp:
    """Top-level page: sidebar, KPIs and the tab container."""

    tab_classes: tuple[type[DashboardTab], ...] = (
        VisualAnalyticsTab,
        TransactionExplorerTab,
        BudgetAnomaliesTab,
        RecurringTab,
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
                tab.render(df)

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
    tr = Translator.from_directory(BASE_DIR / "locales")
    tr.language = st.session_state.get(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE)
    st.set_page_config(
        page_title=tr.t("app.page_title"),
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    write_pending_cookie()
    user: Optional[User] = st.session_state.get(USER_STATE_KEY)
    if user is None:
        auth_page = AuthPage(tr)
        user = auth_page.try_cookie_login()
        if user is None:
            auth_page.render()
            return
    DashboardApp(user, tr).run()


if __name__ == "__main__":
    main()
