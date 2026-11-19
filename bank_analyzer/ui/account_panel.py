"""Sidebar block: current user, account settings, sign out."""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

from modules.facade import BankAnalyzerFacade
from modules.i18n import Translator
from modules.models import User
from modules.remember import RememberMeService
from modules.users import AuthError, AuthService, NotificationSettings

from .common import (COOKIE_STATE_KEY, FLASH_STATE_KEY, REMEMBER_STATE_KEY, USER_STATE_KEY,
                     md_escape, show_flash, sign_out)


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
            self._render_notifications()
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

    def _render_notifications(self) -> None:
        settings = self.auth.notification_settings(self.user)
        with st.form("notifications_form"):
            st.markdown(f"**{self.tr.t('account.notify_header')}**")
            st.caption(self.tr.t("account.notify_help", email=self.user.email))
            enabled = st.checkbox(self.tr.t("account.notify_enabled"), value=settings.enabled)
            days = st.number_input(
                self.tr.t("account.notify_days"), min_value=0, max_value=30,
                value=settings.days_before, step=1,
            )
            submitted = st.form_submit_button(self.tr.t("account.notify_button"))
        if submitted:
            # Emails go out in the language the user is looking at right now.
            new = NotificationSettings(enabled, int(days), self.tr.language)
            self._run(
                lambda: self.auth.save_notification_settings(self.user, new),
                "account.notify_success",
            )

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
