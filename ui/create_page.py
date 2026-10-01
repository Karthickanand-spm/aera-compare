"""Create RFx page: chat with Claude on the left, the draft RFx as a live document on the right.

aera/rfx_builder.py makes the Claude call and checks every number in the draft against
what the buyer typed. Sending is simulated: nothing is emailed, the send is only logged.
"""

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from aera.analyst import AnalystError
from aera.ui import card, info_strip, metric_row, page_header
from ui import compare_page
from aera.rfx_builder import (
    GREETING, LINE_FIELDS, _blank, checklist, is_complete, load_vendor_list, next_turn, quality_bar_set,
    rows_to_lines, to_rfx_json,
)
from ui.state import (
    API_CALLS, RFX_CHAT, RFX_DRAFT, RFX_SENT_LOG, RFX_TABLE_VERSION, get_event, load_sample, md, now_text,
)

VENDORS_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "vendors.json"

STARTERS = ("Shipper cartons for small appliances", "Heavy-duty 7-ply cartons", "Layer pads and partitions")

# What each line must have before it can go out, in the order the chat asks for it.
REQUIRED = (("annual_qty", "qty", "How many {item} a year?"),
            ("size", "size", "Inside dimensions for {item} (L x W x H mm)?"),
            ("board_spec", "board spec", "Board grade for {item}, or type 'vendors to propose'"))

MISSING_COL = "missing"
COLUMN_CONFIG = {
    "description": st.column_config.TextColumn("Description", width="medium"),
    "ply": st.column_config.TextColumn("Ply"),
    "board_spec": st.column_config.TextColumn("Board spec", width="medium"),
    "size": st.column_config.TextColumn("Size (mm)", help="Inside dimensions, L x W x H mm"),
    "print": st.column_config.TextColumn("Print"),
    # Text, not a number column: number columns can't show Indian grouping (1,10,000).
    "annual_qty": st.column_config.TextColumn("Annual qty", help="Whole number, e.g. 1,10,000",
                                              validate=r"^[\d,\s]*$"),
    "uom": st.column_config.TextColumn("UoM"),
    "nominal_weight_g": st.column_config.NumberColumn("Weight", min_value=0, format="%g g"),
    MISSING_COL: st.column_config.TextColumn("Missing", disabled=True),
}


def render() -> None:
    page_header("Create RFx", "Describe what you need and Claude drafts the request for quotes with you.",
                "send it to vendors, then load the sample event to see how replies are compared.")
    if info_strip("Want to see what happens when replies come back? A sample event with 5 vendor "
                  "replies is ready.", "Open the sample event", key="sample_strip"):
        _open_sample()
    st.caption("Claude asks follow-up questions and fills in the draft; code blanks any number you haven't "
               "stated. Each message uses API credit (see the sidebar).")

    draft = st.session_state[RFX_DRAFT]
    chat_col, doc_col = st.columns([45, 55], gap="large")
    with doc_col:
        _document(draft)
    with chat_col:
        _chat(draft)


def _open_sample() -> None:
    """Load the sample event and go to Compare. An event already loaded is kept, with its review decisions."""
    if get_event() is None:
        load_sample()
    st.switch_page(compare_page.page())


# ---------- Chat ----------

def placeholder(draft: dict) -> str:
    """Chat box hint: the next thing the draft needs, worked out in code from the draft."""
    lines = draft["lines"]
    if not lines:
        return "Describe what you need, e.g. 3-ply cartons for kettles, 60,000 a year, delivered to Chakan"
    for field, _, hint in REQUIRED:
        for i, ln in enumerate(lines, start=1):
            if _blank(ln.get(field)):
                return hint.format(item=_item_name(ln, i))
    if _blank(draft["terms"].get("delivery")):
        return "Where should it be delivered?"
    if not quality_bar_set(draft["quality_bar"]):
        return "Any quality requirements, e.g. ISO 9001 or a maximum defect rate?"
    return "Anything to change? Otherwise, click Send to vendors"


def _item_name(line: dict, number: int) -> str:
    desc = (line.get("description") or "").strip()
    if not desc:
        return f"line {number}"
    return desc if desc[:2].isupper() else desc[0].lower() + desc[1:]  # keep acronyms like "OTG"


def _chat(draft: dict) -> None:
    history = st.session_state[RFX_CHAT]
    starter = None
    with st.container(height=560):
        with st.chat_message("assistant"):
            st.markdown(GREETING)
            if not history:
                st.caption("Or start from one of these:")
                with st.container(horizontal=True):
                    for i, text in enumerate(STARTERS):
                        if st.button(text, key=f"rfx_starter_{i}", icon=":material/add:"):
                            starter = text
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(md(msg["content"]))
                for note in msg.get("notes", []):
                    st.caption(f"Check: {md(note)}")

    prompt = st.chat_input(placeholder(draft)) or starter
    if prompt:
        _send(prompt, draft)


def _send(prompt: str, draft: dict) -> None:
    turn = st.session_state[RFX_CHAT] + [{"role": "user", "content": prompt}]
    try:
        with st.spinner("Claude is updating the draft..."):
            reply, new_draft, notes, usage = next_turn(turn, draft)
    except AnalystError as e:
        st.error(f"{e} Your message was not sent; please try again.")
        return
    st.session_state[API_CALLS].append(usage)
    st.session_state[RFX_CHAT] = turn + [{"role": "assistant", "content": reply, "notes": notes}]
    st.session_state[RFX_DRAFT] = new_draft
    st.session_state[RFX_TABLE_VERSION] += 1  # fresh table for the new lines
    st.rerun()


# ---------- The RFx document ----------

def _document(draft: dict) -> None:
    items = checklist(draft)
    done = sum(ok for _, ok, _ in items)
    complete = done == len(items)
    rfx_id = to_rfx_json(draft, date.today())["rfx_id"]

    with card():
        head, pill = st.columns([3, 2], vertical_alignment="center")
        with head:
            st.markdown(f"#### {md(draft.get('title') or 'Untitled RFx')}")
            st.caption(f"{rfx_id} · Draft started {date.today():%d %b %Y}")
        with pill, st.container(horizontal_alignment="right"):
            if complete:
                st.badge("Ready to send", icon=":material/check_circle:", color="green")
            else:
                st.badge(f"Draft · {done} of {len(items)} checks done", color="gray")
        st.progress(done / len(items))

        st.markdown("**Line items**")
        _lines_table(draft)
        _line_metrics(draft["lines"])

        _details(draft)

        st.markdown("**Ready to send?**")
        st.markdown("  \n".join((":green[✓] " if ok else ":gray[○] ") + item + (f" :gray[({detail})]" if detail else "")
                                for item, ok, detail in items))

        st.divider()
        _actions(draft, rfx_id, complete)


def _indian(n: float) -> str:
    """12,34,567 style grouping."""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head] + groups + [tail])


def _table_rows(lines: list[dict]) -> list[dict]:
    rows = []
    for ln in lines:
        row = {f: ln.get(f) for f in LINE_FIELDS}
        row["annual_qty"] = None if ln.get("annual_qty") is None else _indian(ln["annual_qty"])
        gaps = [label for field, label, _ in REQUIRED if _blank(ln.get(field))]
        row[MISSING_COL] = ", ".join(gaps) if gaps else "✓"
        rows.append(row)
    return rows


def _lines_table(draft: dict) -> None:
    key = f"rfx_lines_{st.session_state[RFX_TABLE_VERSION]}"
    table = pd.DataFrame(_table_rows(draft["lines"]), columns=[*LINE_FIELDS, MISSING_COL])
    table["nominal_weight_g"] = pd.to_numeric(table["nominal_weight_g"], errors="coerce")
    st.data_editor(table, key=key, column_config=COLUMN_CONFIG, num_rows="dynamic", hide_index=True,
                   width="stretch", on_change=_commit_table_edits, args=(key,))


def _commit_table_edits(key: str) -> None:
    """Save the buyer's table edits into the draft straight away, so 'Missing' and the checks update."""
    changes = st.session_state[key]
    rows = _table_rows(st.session_state[RFX_DRAFT]["lines"])
    for idx, edit in changes["edited_rows"].items():
        rows[int(idx)].update(edit)
    deleted = {int(i) for i in changes["deleted_rows"]}
    rows = [r for i, r in enumerate(rows) if i not in deleted] + list(changes["added_rows"])
    for r in rows:
        qty = re.sub(r"[,\s]", "", str(r.get("annual_qty") or ""))
        r["annual_qty"] = int(qty) if qty.isdigit() else None
    st.session_state[RFX_DRAFT] = {**st.session_state[RFX_DRAFT], "lines": rows_to_lines(rows)}
    st.session_state[RFX_TABLE_VERSION] += 1


def _line_metrics(lines: list[dict]) -> None:
    if not lines:
        st.caption("No lines yet. They appear here as you describe what you need.")
        return
    qtys = [ln["annual_qty"] for ln in lines if ln.get("annual_qty") is not None]
    no_qty = len(lines) - len(qtys)
    items = [
        ("Lines", len(lines)),
        ("Total annual qty", _indian(sum(qtys)) if qtys else "—",
         f"{no_qty} line(s) have no quantity yet and are not counted." if no_qty else None),
    ]
    if all(ln.get("annual_qty") is not None and ln.get("nominal_weight_g") is not None for ln in lines):
        tonnes = sum(ln["annual_qty"] * ln["nominal_weight_g"] for ln in lines) / 1_000_000
        items.append(("Est. tonnage / year", f"{tonnes:,.1f} t", "Annual qty x nominal weight, summed over all lines."))
    else:
        items.append(("Est. tonnage / year", "—", "Needs a quantity and a weight on every line."))
    metric_row(items)


def _details(draft: dict) -> None:
    def tick(ok: bool) -> str:
        return "✓" if ok else "○"

    questions = draft["questionnaire"]
    with st.expander(f"{tick(bool(questions))} Questionnaire ({len(questions)})"):
        _bullets(questions, "No questions yet.")
    terms = draft["terms"]
    with st.expander(f"{tick(not _blank(terms.get('delivery')))} Terms"):
        _bullets([f"**{name.capitalize()}:** " + (md(v) if v else ":gray[not set]") for name, v in terms.items()],
                 "", escape=False)
    qb = draft["quality_bar"]
    with st.expander(f"{tick(quality_bar_set(qb))} Quality bar"):
        _bullets(_quality_bar_text(qb), "Not set yet.")


def _bullets(items: list[str], empty: str, escape: bool = True) -> None:
    if items:
        st.markdown("\n".join(f"- {md(i) if escape else i}" for i in items))
    elif empty:
        st.caption(empty)


def _quality_bar_text(qb: dict) -> list[str]:
    out = []
    if qb.get("iso9001_required"):
        out.append("Valid ISO 9001 certificate required")
    if qb.get("max_defect_rate_pct") is not None:
        out.append(f"Defect rate at most {qb['max_defect_rate_pct']:g}%")
    if qb.get("test_report_per_batch"):
        out.append("Test report with every batch")
    return out + list(qb.get("other_requirements") or [])


# ---------- Save and send ----------

def _subject(draft: dict, rfx_id: str) -> str:
    return f"Request for quotation: {draft.get('title') or 'Untitled RFx'} ({rfx_id})"


def _actions(draft: dict, rfx_id: str, complete: bool) -> None:
    today = date.today()
    with st.container(horizontal=True):
        st.download_button("Save RFx", json.dumps(to_rfx_json(draft, today), indent=2),
                           file_name=f"rfx_draft_{today:%Y%m%d}.json", mime="application/json",
                           icon=":material/download:")
        if st.button("Send to vendors", type="primary", disabled=not complete, icon=":material/send:",
                     help=None if complete else "Enabled once every check above is done."):
            _send_dialog(draft, rfx_id)

    log = st.session_state[RFX_SENT_LOG]
    if log:
        st.markdown("**Sent**")
        st.markdown("  \n".join(f":green[●] **{md(e['Vendor'])}** · {md(e['Email'])} · :gray[{e['Time']}]"
                                for e in reversed(log)))
        st.caption("Sending is simulated: nothing was emailed. Replies arrive over the next few days.")
    st.caption("The sample event on the Compare page shows what happens when replies come back.")


@st.dialog("Send to vendors")
def _send_dialog(draft: dict, rfx_id: str) -> None:
    vendors = load_vendor_list(VENDORS_PATH)
    by_label = {f"{v['name']} <{v['email']}>": v for v in vendors}
    picked = st.multiselect("Vendors to invite", list(by_label), placeholder="Pick vendors")
    st.text_input("Email subject", _subject(draft, rfx_id), disabled=True)
    st.caption("Simulated: nothing is emailed, the send is only logged.")
    if st.button("Confirm", type="primary", disabled=not picked):
        sent_at = now_text()
        st.session_state[RFX_SENT_LOG].extend(
            {"Vendor": by_label[p]["name"], "Email": by_label[p]["email"], "Time": sent_at,
             "Status": "Replies arrive over the next few days"} for p in picked)
        st.rerun()
