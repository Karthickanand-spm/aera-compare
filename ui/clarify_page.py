"""Clarify page: one card per vendor with its open items and a drafted clarification email.

aera/clarify.py collects the open items in code and asks Claude only to word the
email. Sending is simulated: it adds a line to the sent log, nothing is emailed.
"""

from collections import defaultdict
from datetime import date

import pandas as pd
import streamlit as st

from aera.clarify import ClarifyError, all_open_items, draft_email, ticked_by_default, vendors_mentioned
from aera.compare import display_name, vendor_name
from aera.ingest import load_reply_bytes
from ui.state import (
    API_CALLS, ASK_HISTORY, CLARIFY_DRAFTS, CLARIFY_EXTRA, CLARIFY_FOCUS, SENT_LOG, comparison_tables,
    get_event, md, now_text,
)

URL_PATH = "clarify"
SEVERITY_COLORS = {"high": "red", "medium": "orange", "low": "gray"}  # same as the Decide page


def page() -> st.Page:
    """The navigation entry. Built here so the Ask page can switch to the same page."""
    return st.Page(render, title="Clarify", url_path=URL_PATH)


def open_for(vendor: str, missing: list[str]) -> None:
    """Go to this page with `vendor` shown first and the analyst's missing items added to it."""
    extra = st.session_state[CLARIFY_EXTRA].setdefault(vendor, [])
    extra.extend(m for m in missing if m not in extra)
    st.session_state[CLARIFY_FOCUS] = vendor
    st.switch_page(page())


def render() -> None:
    st.title("Clarify")
    event = get_event()
    if event is None:
        st.info("Click **Load sample event** in the sidebar to begin.")
        return
    if not event.replies:
        st.warning("No vendor replies could be read yet. See the sidebar for details.")
        return

    st.caption("Open items are collected by code from the comparison, each vendor's reply and "
               "certificate, and the Ask page's missing data. Untick anything you don't want to ask, "
               "then draft. Claude only words the email; each draft uses API credit (see the sidebar).")
    comparison, _ = comparison_tables(event)
    items_by_vendor = all_open_items(event.rfx, comparison, event.replies, event.certificates,
                                     _analyst_missing(event))

    focus = st.session_state[CLARIFY_FOCUS]
    order = sorted(items_by_vendor, key=lambda v: v != focus)  # the vendor picked on Ask first
    replies = {vendor_name(ext): ext for ext in event.replies}
    for vendor in order:
        _card(event, replies[vendor], items_by_vendor[vendor], vendor == focus)
    _sent_log()


def _analyst_missing(event) -> dict[str, list[str]]:
    """{vendor: missing-data text} from Ask answers that name the vendor, plus items sent from Ask."""
    by_display = {display_name(vendor_name(ext)): vendor_name(ext) for ext in event.replies}
    names = list(by_display)
    out: dict[str, list[str]] = defaultdict(list)
    for answer in st.session_state[ASK_HISTORY]:
        if answer.data_sufficient:
            continue
        for m in answer.missing_data:
            for name in vendors_mentioned(m, names):
                out[by_display[name]].append(m)
    for vendor, texts in st.session_state[CLARIFY_EXTRA].items():
        out[vendor].extend(texts)
    return out


# ---------- One vendor ----------

def _card(event, ext: dict, items: list[dict], focused: bool) -> None:
    vendor = vendor_name(ext)
    with st.container(border=True):
        st.subheader(display_name(vendor))
        if focused:
            st.caption("Opened from the Ask page: the analyst's missing data is added below.")
        if not items:
            st.success("Nothing to clarify")
            return

        st.markdown("**Open items**")
        if any(not ticked_by_default(it) for it in items):
            st.caption("LOW items start unticked so the email focuses on what blocks the decision. "
                       "Tick them to mention them briefly at the end.")
        chosen = []
        for it in items:
            color = SEVERITY_COLORS[it["severity"]]
            label = f":{color}[**{it['severity'].upper()}**] {md(it['text'])}"
            if it["vendor_words"]:
                label += f" · vendor wrote: *'{md(it['vendor_words'])}'*"
            if st.checkbox(label, value=ticked_by_default(it), key=f"clarify_item_{vendor}_{it['key']}"):
                chosen.append(it)

        if st.button("Draft email", key=f"clarify_draft_{vendor}", type="primary", disabled=not chosen):
            _draft(event, ext, chosen)
        if not chosen:
            st.caption("Tick at least one item to draft an email.")
        _draft_box(vendor, chosen)


def _draft(event, ext: dict, chosen: list[dict]) -> None:
    vendor = vendor_name(ext)
    source = event.files.get(ext.get("source_file"))
    blocks, text, kind = [], None, None
    if source is not None:
        blocks = load_reply_bytes(source.name, source.data).content_blocks
        text, kind = source.text, source.kind
    with st.spinner(f"Claude is drafting the email to {display_name(vendor)}..."):
        try:
            draft = draft_email(event.rfx, vendor, chosen, blocks, text, kind, date.today())
        except ClarifyError as e:
            st.error(f"Could not draft the email: {e}")
            return
    if draft.usage:
        st.session_state[API_CALLS].append(draft.usage)
    st.session_state[CLARIFY_DRAFTS][vendor] = draft
    # The text boxes below have not been drawn yet on this run, so their contents can be set.
    st.session_state[f"clarify_subject_{vendor}"] = draft.subject
    st.session_state[f"clarify_text_{vendor}"] = draft.text
    st.rerun()  # redraw so the sidebar cost includes this draft


def _draft_box(vendor: str, chosen: list[dict]) -> None:
    draft = st.session_state[CLARIFY_DRAFTS].get(vendor)
    if draft is None:
        return
    if [i["key"] for i in chosen] != draft.item_keys:
        st.warning("The ticked items changed since this draft. Click **Draft email** again to update it.")
    subject = st.text_input("Subject", key=f"clarify_subject_{vendor}")
    text = st.text_area("Email (you can edit it)", key=f"clarify_text_{vendor}", height=320)
    for note in draft.notes:
        st.caption(md(note))

    copy, send = st.columns([1, 1])
    with copy.popover("Copy", width="stretch"):
        st.caption("Click the copy icon at the top right of the box.")
        st.code(f"Subject: {subject}\n\n{text}", language=None, wrap_lines=True)
    if send.button("Send (simulated)", key=f"clarify_send_{vendor}", width="stretch"):
        st.session_state[SENT_LOG].append(
            {"Vendor": display_name(vendor), "Time": now_text(), "Subject": subject})
        st.toast(f"Logged as sent to {display_name(vendor)}. Nothing was emailed.")


# ---------- Sent log ----------

def _sent_log() -> None:
    st.divider()
    st.subheader("Sent log")
    st.caption("Sending is simulated in this prototype: nothing is emailed, the send is only logged here.")
    log = st.session_state[SENT_LOG]
    if not log:
        st.caption("Nothing sent yet.")
        return
    st.dataframe(pd.DataFrame(log), hide_index=True)
