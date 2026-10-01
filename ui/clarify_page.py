"""Clarify page: vendor cards on the left; the picked vendor's open items and draft email on the right.

aera/clarify.py collects the open items in code and asks Claude only to word the
email. Sending is simulated: it adds a line to the sent log, nothing is emailed.
"""

from collections import defaultdict
from datetime import date

import streamlit as st

from aera.clarify import (
    ClarifyError, all_open_items, by_severity, draft_email, ticked_by_default, vendors_mentioned,
)
from aera.compare import display_name, vendor_name
from aera.ingest import load_reply_bytes
from aera.ui import card, empty_state, page_header, severity_counts_text, status_badge
from ui.state import (
    API_CALLS, ASK_HISTORY, CLARIFY_DRAFTS, CLARIFY_EXTRA, CLARIFY_FOCUS, SENT_LOG, comparison_tables,
    get_event, load_sample, md, now_text,
)

URL_PATH = "clarify"
SELECTED = "clarify_selected"  # the vendor whose items show on the right (display state only)


def page() -> st.Page:
    """The navigation entry. Built here so the Ask page can switch to the same page."""
    return st.Page(render, title="Clarify", url_path=URL_PATH, icon=":material/mail:")


def open_for(vendor: str, missing: list[str]) -> None:
    """Go to this page with `vendor` shown first and the analyst's missing items added to it."""
    extra = st.session_state[CLARIFY_EXTRA].setdefault(vendor, [])
    extra.extend(m for m in missing if m not in extra)
    st.session_state[CLARIFY_FOCUS] = vendor
    st.session_state[SELECTED] = vendor
    st.switch_page(page())


def render() -> None:
    page_header("Clarify", "Draft one email per vendor asking about anything that is still unclear in their quote.")
    event = get_event()
    if event is None:
        empty_state("No event loaded yet. Load the sample event to see what each vendor still needs to clarify.", "Load sample event", load_sample)
        return
    if not event.replies:
        st.warning("No vendor replies could be read yet. See the sidebar for details.")
        return

    st.caption("Open items are collected by code from the comparison, each vendor's reply and "
               "certificate, and the Ask page's missing data. Untick anything you don't want to ask, "
               "then draft. Claude only words the email; each draft uses API credit (see the sidebar).")
    comparison, summary = comparison_tables(event)
    items_by_vendor = all_open_items(event.rfx, comparison, event.replies, event.certificates,
                                     _analyst_missing(event))
    quality = dict(zip(summary["vendor"], summary["quality_status"]))

    focus = st.session_state[CLARIFY_FOCUS]
    order = sorted(items_by_vendor, key=lambda v: v != focus)  # the vendor picked on Ask first
    selected = st.session_state.get(SELECTED)
    if selected not in items_by_vendor:  # first visit, or a different event was loaded
        selected = next((v for v in order if items_by_vendor[v]), order[0])
        st.session_state[SELECTED] = selected
    replies = {vendor_name(ext): ext for ext in event.replies}

    left, right = st.columns([1, 2.2], gap="large", vertical_alignment="top")
    with left:
        st.markdown("**Vendors**")
        for i, vendor in enumerate(order):
            _vendor_card(i, vendor, quality.get(vendor), items_by_vendor[vendor], vendor == selected)
    with right:
        _vendor_detail(event, replies[selected], items_by_vendor[selected], selected == focus)
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


# ---------- Vendor list ----------

def _select(vendor: str) -> None:
    st.session_state[SELECTED] = vendor


def _vendor_card(i: int, vendor: str, quality: str | None, items: list[dict], selected: bool) -> None:
    """Compact card: name (click to pick), quality badge, and open items counted by severity."""
    with card(key=f"aera-picked-{i}" if selected else f"aera-vendor-{i}"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.button(display_name(vendor), key=f"clarify_pick_{i}", type="tertiary", width="content",
                      on_click=_select, args=(vendor,),
                      help=None if selected else "Show this vendor's open items")
            if quality:
                status_badge(f"Quality {quality}", quality)
        st.caption(severity_counts_text(items))


# ---------- The selected vendor ----------

def _vendor_detail(event, ext: dict, items: list[dict], focused: bool) -> None:
    vendor = vendor_name(ext)
    with card():
        st.subheader(display_name(vendor), anchor=False)
        if focused:
            st.caption("Opened from the Ask page: the analyst's missing data is added below.")
        if not items:
            st.success("Nothing to clarify")
            return

        st.markdown("**Open items**")
        if any(not ticked_by_default(it) for it in items):
            st.caption("LOW items start unticked so the email focuses on what blocks the decision. "
                       "Tick them to mention them briefly at the end.")
        chosen = [it for it in by_severity(items) if _item_row(vendor, it)]

        if st.button("Draft email", key=f"clarify_draft_{vendor}", type="primary", disabled=not chosen,
                     icon=":material/edit:"):
            _draft(event, ext, chosen)
        if not chosen:
            st.caption("Tick at least one item to draft an email.")
        _draft_box(vendor, chosen)


def _item_row(vendor: str, it: dict) -> bool:
    """One checklist row: tick box, severity badge, item text. Returns whether it is ticked."""
    sev = it["severity"].upper()
    text = md(it["text"])
    if it["vendor_words"]:
        text += f" · vendor wrote: *'{md(it['vendor_words'])}'*"
    with st.container(horizontal=True, vertical_alignment="top", gap="small"):
        # The label is hidden on screen (the badge and text show it) but still read out by screen readers.
        ticked = st.checkbox(f"{sev} {text}", value=ticked_by_default(it), label_visibility="collapsed",
                             key=f"clarify_item_{vendor}_{it['key']}", width="content")
        status_badge(sev)
        st.markdown(text, width="stretch")
    return ticked


def _draft(event, ext: dict, chosen: list[dict]) -> None:
    vendor = vendor_name(ext)
    source = event.files.get(ext.get("source_file"))
    blocks, text, kind = [], None, None
    if source is not None:
        blocks = load_reply_bytes(source.name, source.data).content_blocks
        text, kind = source.text, source.kind
    with st.spinner("Drafting the email..."):
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
    with copy.popover("Copy", width="stretch", icon=":material/content_copy:"):
        st.caption("Click the copy icon at the top right of the box.")
        st.code(f"Subject: {subject}\n\n{text}", language=None, wrap_lines=True)
    if send.button("Send (simulated)", key=f"clarify_send_{vendor}", width="stretch", icon=":material/send:"):
        st.session_state[SENT_LOG].append(
            {"Vendor": display_name(vendor), "Time": now_text(), "Subject": subject})
        st.toast(f"Logged as sent to {display_name(vendor)}. Nothing was emailed.")


# ---------- Sent log ----------

def _sent_log() -> None:
    st.divider()
    st.markdown("**Sent log**")
    log = st.session_state[SENT_LOG]
    if not log:
        st.caption("Nothing sent yet. Sending is simulated: nothing is emailed, the send is only logged here.")
        return
    # Newest first, one line per send: a small timeline.
    st.markdown("  \n".join(f":green[●] **{md(e['Vendor'])}** · {md(e['Subject'])} · :gray[{e['Time']}]"
                            for e in reversed(log)))
    st.caption("Sending is simulated: nothing was emailed.")
