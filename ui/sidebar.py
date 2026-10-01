"""Sidebar shown on every page: load the event, FX rate, add a reply, re-extract."""

import streamlit as st

from aera.compare import vendor_name
from aera.config import FX_DATE, FX_SOURCE
from aera.event import add_reply, load_sample_event, reextract_all
from aera.extract import ExtractionError
from ui.state import (
    API_CALLS, DECISIONS, DEFAULT_FX_USD, FX_USD, UPLOADS_DONE, fx_is_default, get_event, set_event,
)

UPLOAD_TYPES = ["xlsx", "docx", "pdf", "eml", "jpg", "jpeg", "png"]


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Event")
        _load_sample()
        st.divider()
        _fx_input()
        st.divider()
        _upload()
        st.divider()
        _reextract()
        st.divider()
        _api_cost()


def _api_cost() -> None:
    calls = st.session_state[API_CALLS]
    total = sum(u["cost_usd"] for u in calls)
    tokens_in = sum(u["input_tokens"] + u["cache_write_tokens"] + u["cache_read_tokens"] for u in calls)
    tokens_out = sum(u["output_tokens"] for u in calls)
    st.caption(f"Ask and Clarify API cost this session: **~${total:.4f}** "
               f"({len(calls)} calls, {tokens_in:,} tokens in, {tokens_out:,} out)")


def _load_sample() -> None:
    if st.button("Load sample event", type="primary", width="stretch"):
        with st.spinner("Loading the sample event (cached files make no API calls)..."):
            set_event(load_sample_event())

    event = get_event()
    if event is None:
        st.caption("No event loaded yet.")
        return
    st.caption(f"{event.rfx.rfx_id} · {len(event.replies)} vendor replies · "
               f"{len(event.certificates)} certificates")
    if event.errors:
        st.warning("Some files could not be used:\n\n" + "\n".join(f"- {e}" for e in event.errors))


def _fx_input() -> None:
    st.number_input("USD to INR rate", min_value=0.01, step=0.25, format="%.2f", key=FX_USD,
                    help="Used for every USD price. Changing it recomputes the comparison.")
    st.caption(f"Rate date {FX_DATE} · {FX_SOURCE}")
    if not fx_is_default():
        st.caption(f"You changed the rate (default {DEFAULT_FX_USD:.2f}). "
                   "Assumptions now say it was entered by you.")
        st.button("Reset to default rate", on_click=_reset_fx)


def _reset_fx() -> None:
    st.session_state[FX_USD] = DEFAULT_FX_USD


def _upload() -> None:
    event = get_event()
    uploaded = st.file_uploader(
        "Add a vendor reply", type=UPLOAD_TYPES, disabled=event is None,
        help="Excel, Word, PDF, email or a photo. Read live by Claude (uses API credit).",
    )
    if event is None:
        st.caption("Load an event first.")
        return
    if uploaded is None or uploaded.file_id in st.session_state[UPLOADS_DONE]:
        return

    # Mark it first: a failed upload is not retried on every click. Upload it again to retry.
    st.session_state[UPLOADS_DONE].add(uploaded.file_id)
    try:
        with st.spinner(f"Reading {uploaded.name} with Claude..."):
            added, note = add_reply(event, uploaded.name, uploaded.getvalue())
    except (ExtractionError, ValueError) as e:
        st.error(f"Could not add {uploaded.name}: {e}")
        return

    if not added:
        st.info(note)
        return
    # Old decisions for this vendor belonged to their earlier reply.
    new_vendor = vendor_name(event.replies[-1])
    st.session_state[DECISIONS] = {k: v for k, v in st.session_state[DECISIONS].items()
                                   if k[1] != new_vendor}
    st.success(note or f"Added {uploaded.name} to the comparison.")


def _reextract() -> None:
    event = get_event()
    if event is None:
        return
    st.warning(f"Re-extract all calls Claude again for all {len(event.files)} files, ignoring "
               f"the cache. It costs API credit (about ${event.extraction_cost_usd:.2f} last "
               "time) and clears your review decisions.")
    sure = st.checkbox("I understand this uses API credit")
    if st.button("Re-extract all", disabled=not sure, width="stretch"):
        with st.status("Re-extracting every file...", expanded=True) as status:
            fresh = reextract_all(event, progress=status.write)
            status.update(label="Re-extraction finished", state="complete")
        set_event(fresh)
        st.rerun()
