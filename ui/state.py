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

DEFAULT_FX_USD = FX_RATES["USD"]


def init_state() -> None:
    st.session_state.setdefault(EVENT, None)
    st.session_state.setdefault(DECISIONS, {})
    st.session_state.setdefault(FX_USD, DEFAULT_FX_USD)
    st.session_state.setdefault(UPLOADS_DONE, set())


def get_event() -> Event | None:
    return st.session_state[EVENT]


def set_event(event: Event) -> None:
    """A freshly loaded or re-extracted event. Earlier review decisions no longer apply."""
    st.session_state[EVENT] = event
    st.session_state[DECISIONS] = {}


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
