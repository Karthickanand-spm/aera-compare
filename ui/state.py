"""Session state shared by every page. The UI stores choices here; aera/ does the maths."""

import re
from datetime import date, datetime, timezone

import streamlit as st

from aera.compare import compare
from aera.config import FX_DATE, FX_RATES, FX_SOURCE
from aera.event import Event, load_sample_event
from aera.rfx_builder import empty_draft

EVENT = "event"  # aera.event.Event, or None before anything is loaded
DECISIONS = "decisions"  # {(rfx_line_id, vendor): aera.compare.buyer_decision(...)}
FX_USD = "fx_usd"  # the USD -> INR rate the buyer is using
REJECTED_UPLOAD = "rejected_upload"  # {"name", "data", "reason"} of the last upload, if it wasn't a quote
UPLOADS_DONE = "uploads_done"  # file_ids of uploads already processed (so reruns don't re-extract)
ASK_HISTORY = "ask_history"  # aera.analyst.Answer list, newest first
API_CALLS = "api_calls"  # usage dicts (tokens + cost_usd) for every Ask and Clarify call this session
AWARD_CONFIRMED = "award_confirmed"  # aera.award.confirm_award(...) record, or None
CLARIFY_DRAFTS = "clarify_drafts"  # {vendor: aera.clarify.Draft}
CLARIFY_EXTRA = "clarify_extra"  # {vendor: [analyst missing-data text]} sent over from the Ask page
CLARIFY_FOCUS = "clarify_focus"  # vendor to show first on the Clarify page, or None
SENT_LOG = "sent_log"  # [{"Vendor", "Time", "Subject"}] for simulated sends
RFX_CHAT = "rfx_chat"  # Create RFx chat: [{"role", "content", "notes"}]
RFX_DRAFT = "rfx_draft"  # the draft RFx being built (aera.rfx_builder.empty_draft() shape)
RFX_TABLE_VERSION = "rfx_table_version"  # bumped when Claude updates the draft, so the table resets
RFX_SENT_LOG = "rfx_sent_log"  # [{"Vendor", "Email", "Time", "Status"}] for the simulated RFx send

DEFAULT_FX_USD = FX_RATES["USD"]


def init_state() -> None:
    st.session_state.setdefault(EVENT, None)
    st.session_state.setdefault(DECISIONS, {})
    st.session_state.setdefault(FX_USD, DEFAULT_FX_USD)
    st.session_state.setdefault(UPLOADS_DONE, set())
    st.session_state.setdefault(REJECTED_UPLOAD, None)
    st.session_state.setdefault(ASK_HISTORY, [])
    st.session_state.setdefault(API_CALLS, [])
    st.session_state.setdefault(AWARD_CONFIRMED, None)
    st.session_state.setdefault(CLARIFY_DRAFTS, {})
    st.session_state.setdefault(CLARIFY_EXTRA, {})
    st.session_state.setdefault(CLARIFY_FOCUS, None)
    st.session_state.setdefault(SENT_LOG, [])
    st.session_state.setdefault(RFX_CHAT, [])
    st.session_state.setdefault(RFX_DRAFT, empty_draft())
    st.session_state.setdefault(RFX_TABLE_VERSION, 0)
    st.session_state.setdefault(RFX_SENT_LOG, [])


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


def load_sample() -> None:
    """Load the sample event (cached extractions, so no API calls)."""
    with st.spinner("Loading the sample event (cached files make no API calls)..."):
        set_event(load_sample_event())


def completed_steps(state=None) -> set[str]:
    """Workflow steps the buyer has finished this session, for the stepper at the top of each page."""
    state = st.session_state if state is None else state
    done = {
        "Create RFx": bool(state.get(RFX_SENT_LOG)),
        "Compare": state.get(EVENT) is not None,
        "Ask": bool(state.get(ASK_HISTORY)),
        "Decide": state.get(AWARD_CONFIRMED) is not None,
        "Clarify": bool(state.get(SENT_LOG)),
    }
    return {step for step, ok in done.items() if ok}


def keep_fx_values() -> None:
    """Keep the buyer's FX rates on a page that doesn't draw the FX inputs.

    Streamlit forgets a widget's value on any run where the widget isn't drawn. Writing the value
    back to session state on those runs keeps it, so the rates are still there on Compare."""
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (key == FX_USD or key.startswith((FX_EXTRA_RATE, FX_EXTRA_DATE))):
            st.session_state[key] = st.session_state[key]


def fx_is_default() -> bool:
    return st.session_state[FX_USD] == DEFAULT_FX_USD


# Rates the buyer enters for currencies not in config.FX_RATES (sidebar widgets, one pair per currency).
FX_EXTRA_RATE = "fx_rate|"  # + currency code -> float or None
FX_EXTRA_DATE = "fx_date|"  # + currency code -> datetime.date


def fx_settings() -> tuple[dict[str, float], dict[str, str]]:
    """(rates, source text per currency) to pass to compare(). Each source ends up in that FX assumption."""
    rates = {**FX_RATES, "USD": float(st.session_state[FX_USD])}
    sources = {code: f"rate dated {FX_DATE}, {FX_SOURCE.lower()}" for code in FX_RATES}
    if not fx_is_default():
        sources["USD"] = f"buyer-entered on {date.today().isoformat()}"
    for key, value in st.session_state.to_dict().items():
        if isinstance(key, str) and key.startswith(FX_EXTRA_RATE) and value:
            code = key[len(FX_EXTRA_RATE):]
            rates[code] = float(value)
            entered_on = st.session_state.get(FX_EXTRA_DATE + code) or date.today()
            sources[code] = f"buyer-entered on {entered_on.isoformat()}"
    return rates, sources


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
