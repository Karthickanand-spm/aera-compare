"""Session state shared by every page. The UI stores choices here; aera/ does the maths."""

import re
from datetime import date, datetime, timezone

import streamlit as st

from aera.compare import compare
from aera.config import FX_DATE, FX_RATES, FX_SOURCE
from aera.event import Event

EVENT = "event"  # aera.event.Event, or None before anything is loaded
DECISIONS = "decisions"  # {(rfx_line_id, vendor): aera.compare.buyer_decision(...)}
FX_USD = "fx_usd"  # the USD -> INR rate the buyer is using
UPLOADS_DONE = "uploads_done"  # file_ids of uploads already processed (so reruns don't re-extract)
ASK_HISTORY = "ask_history"  # aera.analyst.Answer list, newest first
API_CALLS = "api_calls"  # usage dicts (tokens + cost_usd) for every Ask and Clarify call this session
AWARD_CONFIRMED = "award_confirmed"  # aera.award.confirm_award(...) record, or None
CLARIFY_DRAFTS = "clarify_drafts"  # {vendor: aera.clarify.Draft}
CLARIFY_EXTRA = "clarify_extra"  # {vendor: [analyst missing-data text]} sent over from the Ask page
CLARIFY_FOCUS = "clarify_focus"  # vendor to show first on the Clarify page, or None
SENT_LOG = "sent_log"  # [{"Vendor", "Time", "Subject"}] for simulated sends

DEFAULT_FX_USD = FX_RATES["USD"]


def init_state() -> None:
    st.session_state.setdefault(EVENT, None)
    st.session_state.setdefault(DECISIONS, {})
    st.session_state.setdefault(FX_USD, DEFAULT_FX_USD)
    st.session_state.setdefault(UPLOADS_DONE, set())
    st.session_state.setdefault(ASK_HISTORY, [])
    st.session_state.setdefault(API_CALLS, [])
    st.session_state.setdefault(AWARD_CONFIRMED, None)
    st.session_state.setdefault(CLARIFY_DRAFTS, {})
    st.session_state.setdefault(CLARIFY_EXTRA, {})
    st.session_state.setdefault(CLARIFY_FOCUS, None)
    st.session_state.setdefault(SENT_LOG, [])


def get_event() -> Event | None:
    return st.session_state[EVENT]


def set_event(event: Event) -> None:
    """A freshly loaded or re-extracted event. Earlier review decisions and drafts no longer apply.

    The sent log is kept: it is a record of what was already sent."""
    st.session_state[EVENT] = event
    st.session_state[DECISIONS] = {}
    st.session_state[AWARD_CONFIRMED] = None
    st.session_state[CLARIFY_DRAFTS] = {}
    st.session_state[CLARIFY_EXTRA] = {}
    st.session_state[CLARIFY_FOCUS] = None


def fx_is_default() -> bool:
    return st.session_state[FX_USD] == DEFAULT_FX_USD


def fx_settings() -> tuple[dict[str, float], str]:
    """(rates, date text) to pass to compare(). The date text ends up in every FX assumption."""
    rates = {**FX_RATES, "USD": float(st.session_state[FX_USD])}
    if fx_is_default():
        return rates, f"{FX_DATE}, {FX_SOURCE.lower()}"
    return rates, f"{date.today().isoformat()}, entered by the buyer"


def comparison_tables(event: Event):
    """(comparison, vendor summary) for the event with the current FX rate and decisions."""
    rates, fx_date = fx_settings()
    return compare(event.rfx, event.last_year, event.replies, event.certificates, rates,
                   fx_date, event.texts, st.session_state[DECISIONS])


def now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|$~<>:])")


def md(text) -> str:
    """Vendor text made safe for st.markdown (no accidental bold, links, maths or emoji codes)."""
    return _MD_SPECIAL.sub(r"\\\1", "" if text is None else str(text))
