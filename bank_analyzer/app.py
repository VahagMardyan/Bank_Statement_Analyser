"""Streamlit entry point for Bank Statement Analyzer.

Run with ``streamlit run app.py``. This file only decides which page to show;
the pages themselves live in ``ui/``:

    ui/auth_page.py      AuthPage: sign-in / sign-up, shown when nobody is signed in
    ui/dashboard.py      DashboardApp: sidebar, KPI row, tab container
    ui/tabs.py           DashboardTab (ABC) and one subclass per tab:
                         VisualAnalyticsTab, TransactionExplorerTab,
                         BudgetAnomaliesTab, RecurringTab
    ui/account_panel.py  AccountPanel: who is signed in, account settings, sign out
    ui/common.py         session keys, cookies, flash messages, language switch

Adding a tab = writing one subclass in ``ui/tabs.py`` and adding it to
``DashboardApp.tab_classes``. All data access goes through
:class:`modules.facade.BankAnalyzerFacade`.

There is no UI text in the code: every label comes from ``locales/<lang>.json``
through :class:`modules.i18n.Translator`. Category keys and transaction types
stay English in the data; the translator only changes how they are shown.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from modules.i18n import DEFAULT_LANGUAGE, Translator
from modules.models import User
from ui.auth_page import AuthPage
from ui.common import BASE_DIR, LANGUAGE_STATE_KEY, USER_STATE_KEY, write_pending_cookie
from ui.dashboard import DashboardApp


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
