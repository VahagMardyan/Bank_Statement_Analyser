"""Sign-in / sign-up page shown when nobody is signed in."""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

from modules.i18n import Translator
from modules.models import User
from modules.remember import RememberMeService
from modules.users import AuthError, AuthService

from .common import (COOKIE_CHECKED_KEY, COOKIE_STATE_KEY, REMEMBER_STATE_KEY, USER_STATE_KEY,
                     remember_cookie, render_language_switch, show_flash)


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
