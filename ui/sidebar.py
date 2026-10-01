"""Sidebar shown on every page, in three groups: Event, Assumptions, Session."""

from datetime import date

import streamlit as st

from aera.compare import currencies_without_rate, vendor_name
from aera.config import FX_DATE, FX_RATES, FX_SOURCE
from aera.event import NotAQuote, add_reply, reextract_all
from aera.extract import ExtractionError
from aera.ui import render_sidebar, sidebar_sections
from ui.state import (
    API_CALLS, DECISIONS, DEFAULT_FX_USD, FX_EXTRA_DATE, FX_EXTRA_RATE, FX_USD, REJECTED_UPLOAD, UPLOADS_DONE, fx_is_default, get_event,
    keep_fx_values, load_sample, md, set_event,
)

UPLOAD_TYPES = ["xlsx", "docx", "pdf", "eml", "jpg", "jpeg", "png"]
FOOTER = "Concept prototype. Not an Aerchain product."


def render(page: str) -> None:
    """The sidebar for `page` (its title). aera/ui.py decides which sections show there."""
    if "Assumptions" not in sidebar_sections(page):
        keep_fx_values()  # the FX inputs aren't drawn on this page, so Streamlit would drop their values
    # Event comes before Assumptions, so a currency in a reply added on this run gets its rate box at once.
    render_sidebar(page, {"Event": _event, "Assumptions": _assumptions, "Session": _session})


def _event() -> None:
    _load_sample()
    _upload()
    _reextract()


def _assumptions() -> None:
    _fx_input()
    _extra_fx_inputs()


def _session() -> None:
    _api_cost()
    st.caption(FOOTER)


def _api_cost() -> None:
    calls = st.session_state[API_CALLS]
    total = sum(u["cost_usd"] for u in calls)
    tokens_in = sum(u["input_tokens"] + u["cache_write_tokens"] + u["cache_read_tokens"] for u in calls)
    tokens_out = sum(u["output_tokens"] for u in calls)
    st.caption(f"API cost so far: **~${total:.4f}**")
    st.caption(f"Create RFx, Ask and Clarify · {len(calls)} calls · {tokens_in:,} tokens in, {tokens_out:,} out")


def _load_sample() -> None:
    if st.button("Load sample event", type="primary", width="stretch"):
        load_sample()

    event = get_event()
    if event is None:
        st.caption("No event loaded yet.")
        return
    st.caption(f"{event.rfx.rfx_id} · {len(event.replies)} vendor replies · "
               f"{len(event.certificates)} certificates")
    if event.errors:
        st.warning("Some files could not be used:\n\n" + "\n".join(f"- {e}" for e in event.errors))


def _fx_input() -> None:
    """The USD rate, with its date and where it came from."""
    st.number_input("USD to INR rate", min_value=0.01, step=0.25, format="%.2f", key=FX_USD,
                    help="Used for every USD price. Changing it recomputes the comparison.")
    if fx_is_default():
        st.caption(f"{FX_SOURCE} · {_nice_date(FX_DATE)}")
    else:
        st.caption(f"Entered by you (default {DEFAULT_FX_USD:.2f}, {FX_SOURCE.lower()} dated {FX_DATE}). "
                   "Assumptions now say it was entered by you.")
        st.button("Reset to default rate", on_click=_reset_fx)


def _nice_date(iso: str) -> str:
    """'2026-09-25' -> '25 Sep 2026'."""
    d = date.fromisoformat(iso)
    return f"{d.day} {d:%b %Y}"


def _extra_fx_inputs() -> None:
    event = get_event()
    if event is None:
        return
    # Checked against config rates only, so a currency's input stays put once the buyer fills it.
    for code, needed in currencies_without_rate(event.replies, FX_RATES).items():
        _extra_fx_input(code, needed)


def _extra_fx_input(code: str, needed: dict) -> None:
    """A rate for a currency the app has no rate for. Empty by default: nothing is guessed."""
    vendors, lines = needed["vendors"], needed["lines"]
    label = (f"{code} rate (₹ per {code}), needed for {vendors} vendor{'s' if vendors != 1 else ''}, "
             f"{lines} line{'s' if lines != 1 else ''}")
    rate = st.number_input(label, min_value=0.01, step=0.25, format="%.2f", value=None,
                           key=FX_EXTRA_RATE + code, placeholder="Enter a rate",
                           help=f"The app has no {code} rate. Until you enter one, {code} prices stay "
                                "Not comparable and out of totals.")
    st.date_input("Rate date", value=date.today(), key=FX_EXTRA_DATE + code)
    st.caption("Buyer-entered rate · shown in every assumption that uses it" if rate
               else f"Buyer-entered rate · empty, so {code} prices stay Not comparable")


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
    if uploaded is not None and uploaded.file_id not in st.session_state[UPLOADS_DONE]:
        # Mark it first: a failed upload is not retried on every click. Upload it again to retry.
        st.session_state[UPLOADS_DONE].add(uploaded.file_id)
        st.session_state[REJECTED_UPLOAD] = None
        with st.spinner(f"Reading {uploaded.name} with Claude..."):
            _add(event, uploaded.name, uploaded.getvalue())
    _rejected_upload(event)


def _add(event, name: str, data: bytes, force: bool = True, allow_non_quote: bool = False) -> None:
    """Add one reply to the event and say what happened. A non-quote is held back for the buyer."""
    try:
        added, note = add_reply(event, name, data, force=force, allow_non_quote=allow_non_quote)
    except NotAQuote as e:
        st.session_state[REJECTED_UPLOAD] = {"name": name, "data": data, "reason": str(e)}
        # The warning sits under the uploader, often below the fold of the sidebar; the toast shows on screen.
        st.toast(f"{md(name)} was not added: it doesn't look like a quote for this RFx. See the sidebar.",
                 icon=":material/warning:")
        return
    except (ExtractionError, ValueError) as e:
        st.error(f"Could not add {name}: {e}")
        return

    if not added:
        st.info(note)
        return
    # Old decisions for this vendor belonged to their earlier reply.
    new_vendor = vendor_name(event.replies[-1])
    st.session_state[DECISIONS] = {k: v for k, v in st.session_state[DECISIONS].items()
                                   if k[1] != new_vendor}
    st.success(note or f"Added {name} to the comparison.")


def _rejected_upload(event) -> None:
    """Warn about an upload that isn't a quote, until the next upload.

    'Add anyway' reuses the cached extraction (no API cost)."""
    rejected = st.session_state[REJECTED_UPLOAD]
    if rejected is None:
        return
    warning = st.empty()  # cleared below if the buyer adds it, so the warning and "Added" don't both show
    warning.warning(f"This doesn't look like a quote for this RFx: {md(rejected['reason'])}. Nothing was added.")
    if st.button("Add anyway", key="add_rejected_upload", type="tertiary", icon=":material/add:"):
        warning.empty()
        st.session_state[REJECTED_UPLOAD] = None
        _add(event, rejected["name"], rejected["data"], force=False, allow_non_quote=True)


def _reextract() -> None:
    event = get_event()
    if event is None:
        return
    st.caption(f":orange[**Re-extract all**] reads all {len(event.files)} files again with Claude, ignoring "
               f"the cache. It costs API credit (about ${event.extraction_cost_usd:.2f} last "
               "time) and clears your review decisions.")
    sure = st.checkbox("I understand this uses API credit")
    if st.button("Re-extract all", disabled=not sure, icon=":material/refresh:"):
        with st.status("Re-extracting every file...", expanded=True) as status:
            fresh = reextract_all(event, progress=status.write)
            status.update(label="Re-extraction finished", state="complete")
        set_event(fresh)
        st.rerun()
