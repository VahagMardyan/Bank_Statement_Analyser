"""Shared UI helpers: session keys, cookies, flash messages, language switch."""

from __future__ import annotations

import base64
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from modules.i18n import Translator
from modules.models import User
from modules.remember import REMEMBER_DAYS, RememberMeService

USER_STATE_KEY = "user"
FLASH_STATE_KEY = "flash"  # locale key of a success message shown after st.rerun()
REMEMBER_STATE_KEY = "remember"  # this session came from / created a remember-me cookie
COOKIE_STATE_KEY = "pending_cookie"  # cookie value to write on the next run ("" = delete)
COOKIE_CHECKED_KEY = "cookie_checked"
REMEMBER_COOKIE = "bsa_remember"
BASE_DIR = Path(__file__).resolve().parent.parent  # bank_analyzer/
DATE_DISPLAY_FORMAT = "%Y-%m-%d, %H:%M"
LANGUAGE_STATE_KEY = "language"
# Set by code (sign-in, account settings) and applied by render_language_switch
# before the radio is drawn: a widget's own key can't be changed after that.
PENDING_LANGUAGE_KEY = "pending_language"
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

    The slot is created on EVERY run, even when there is nothing to write.
    Streamlit matches elements between runs by their position. If the script
    took position 0 only on some runs, everything below would shift by one
    and the old login form would stay on screen under the new one (seen when
    pressing Enter in the login form right after signing out).
    """
    slot = st.empty()
    value = st.session_state.pop(COOKIE_STATE_KEY, None)
    if value is None:
        return
    max_age = REMEMBER_DAYS * 24 * 3600 if value else 0
    slot.html(
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
    pending = st.session_state.pop(PENDING_LANGUAGE_KEY, None)
    if pending in codes:
        st.session_state[LANGUAGE_STATE_KEY] = pending
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


def use_account_language(user: User) -> None:
    """Switch the app to the user's account language on the next run."""
    if user.language:
        st.session_state[PENDING_LANGUAGE_KEY] = user.language


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
