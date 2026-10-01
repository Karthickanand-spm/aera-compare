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
from aera.ui import card, metric_row, page_header, rfx_status, status_badge
from ui import compare_page
from aera.rfx_builder import (
    GREETING, LINE_FIELDS, _blank, checklist, is_complete, load_vendor_list, next_turn, quality_bar_set,
    reply_text, rows_to_lines, tagged_answers, to_rfx_json,
)
from ui.state import (
    API_CALLS, RFX_CHAT, RFX_DRAFT, RFX_PENDING, RFX_PROCESSED, RFX_SENT_LOG, RFX_TABLE_VERSION, get_event, load_sample, md,
    now_text,
)

ROOT = Path(__file__).resolve().parent.parent
VENDORS_PATH = ROOT / "data" / "sample" / "vendors.json"
ASSISTANT_AVATAR = str(ROOT / "assets" / "icon.svg")  # the AC mark
BUYER_AVATAR = ":material/person:"
ASSISTANT_NAME = "Aera guide"
OPEN_QUESTIONS_HINT = "Answer the questions above, or type anything else"

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
    page_header("Create RFx", "Describe what you need and Aera guide drafts the request for quotes with you.")
    draft = st.session_state[RFX_DRAFT]
    # Both columns start with a bordered box, so their tops line up.
    chat_col, doc_col = st.columns([45, 55], gap="large", vertical_alignment="top")
    with doc_col:
        _document(draft)
    with chat_col:
        _chat(draft)


def open_sample() -> None:
    """Load the sample event and go to Compare. Also used by the sidebar shortcut. An event already loaded is kept, with its review decisions."""
    if get_event() is None:
        load_sample()
    st.switch_page(compare_page.page())


# ---------- Chat ----------

def placeholder(draft: dict, questions_open: bool = False) -> str:
    """Chat box hint: the next thing the draft needs, worked out in code from the draft."""
    if questions_open:
        return OPEN_QUESTIONS_HINT
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


def open_questions(history: list[dict]) -> list[dict]:
    """Questions on the latest assistant message; none once the buyer has replied to it."""
    if history and history[-1]["role"] == "assistant":
        return history[-1].get("questions") or []
    return []


def claim_submission(processed: set[str], submission_id: str) -> bool:
    """True the first time a submission id is seen, False after that, so nothing is sent twice."""
    if submission_id in processed:
        return False
    processed.add(submission_id)
    return True


def _item_name(line: dict, number: int) -> str:
    desc = (line.get("description") or "").strip()
    if not desc:
        return f"line {number}"
    return desc if desc[:2].isupper() else desc[0].lower() + desc[1:]  # keep acronyms like "OTG"


def _chat(draft: dict) -> None:
    history = st.session_state[RFX_CHAT]
    pending = st.session_state[RFX_PENDING]
    busy = pending is not None and pending["status"] != "failed"
    settled = history[:-1] if pending else history  # the chat as of the last reply
    questions = open_questions(settled)
    sent = None
    with st.container(height=560):
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            _assistant_label()
            st.markdown(GREETING)
            if not history:
                st.caption("Or start from one of these:")
                with st.container(horizontal=True):
                    for i, text in enumerate(STARTERS):
                        if st.button(text, key=f"rfx_starter_{i}", icon=":material/add:"):
                            sent = text
        for n, msg in enumerate(history):
            if msg["role"] == "user":
                with st.chat_message("user", avatar=BUYER_AVATAR):
                    st.markdown(md(msg["content"]))
                continue
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                _assistant_label()
                st.markdown(md(msg.get("summary", msg["content"])))
                for note in msg.get("notes", []):
                    st.caption(f"Check: {md(note)}")
                if questions and n == len(settled) - 1:
                    sent = _answer_form(questions, key=f"rfx_answers_{n}", busy=busy) or sent
        if pending:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                _assistant_label()
                if busy:
                    _run_pending(draft)
                else:
                    st.markdown(f"Sorry, that didn't go through. {md(pending['error'])}")
                    if st.button("Try again", key="rfx_retry", icon=":material/refresh:") and \
                            claim_submission(st.session_state[RFX_PROCESSED], f"turn-{len(settled)}"):
                        st.session_state[RFX_PENDING] = {"status": "queued"}
                        st.rerun()

    hint = f"{ASSISTANT_NAME} is drafting your RFx..." if busy else placeholder(draft, bool(questions))
    sent = st.chat_input(hint, disabled=busy) or sent
    st.caption(f"{ASSISTANT_NAME} asks follow-up questions and fills in the draft; code blanks any number you "
               "haven't stated. Each message uses API credit (see the sidebar).")
    # Widgets return a value only on the run they were submitted, so the API is called only then.
    # The id ties a submission to the chat turn it answers, so a stray rerun can't send it again.
    if sent and claim_submission(st.session_state[RFX_PROCESSED], f"turn-{len(settled)}"):
        _queue(sent)


def _assistant_label() -> None:
    st.markdown(f":small[**{ASSISTANT_NAME}**]")


def _answer_form(questions: list[dict], key: str, busy: bool = False) -> str | None:
    """One input per open question, plus a 'Vendors to propose' tick where it makes sense.

    Greyed out while a reply is being drafted. Returns the tagged answer message on submit, else None."""
    answers, propose = {}, {}
    with st.form(key, border=False):
        for i, q in enumerate(questions, start=1):
            st.markdown(f"**{i}. {md(q['text'])}**")
            answers[q["id"]] = st.text_input(q["text"], placeholder=q["example"], key=f"{key}_{q['id']}",
                                             label_visibility="collapsed", disabled=busy)
            if q["allow_vendors_propose"]:
                propose[q["id"]] = st.checkbox("Vendors to propose", key=f"{key}_{q['id']}_propose",
                                               disabled=busy)
        if st.form_submit_button("Drafting..." if busy else "Send answers", type="primary",
                                 icon=":material/send:", disabled=busy):
            return tagged_answers(questions, answers, propose)
    return None


def _queue(prompt: str) -> None:
    """Show the buyer's message straight away; the next run makes the API call under it."""
    history = st.session_state[RFX_CHAT]
    if st.session_state[RFX_PENDING] is not None:  # a failed message is replaced by the new one
        history = history[:-1]
    st.session_state[RFX_CHAT] = history + [{"role": "user", "content": prompt}]
    st.session_state[RFX_PENDING] = {"status": "queued"}
    st.rerun()


def _fail(error: str) -> None:
    st.session_state[RFX_PENDING] = {"status": "failed", "error": error}
    # The buyer may send this turn again (Try again, or a new message).
    st.session_state[RFX_PROCESSED].discard(f"turn-{len(st.session_state[RFX_CHAT]) - 1}")
    st.rerun()


def _run_pending(draft: dict) -> None:
    """The API call for the buyer's newest message, with a spinner in the reply bubble."""
    pending = st.session_state[RFX_PENDING]
    if pending["status"] == "running":
        # An earlier run started this call and was cut off (e.g. the page was left), so don't resend by itself.
        _fail("The request was interrupted before the reply arrived.")
    pending["status"] = "running"
    turn = st.session_state[RFX_CHAT]
    try:
        with st.spinner(f"{ASSISTANT_NAME} is drafting your RFx..."):
            summary, questions, new_draft, notes, usage = next_turn(turn, draft)
    except AnalystError as e:
        _fail(str(e))
    st.session_state[API_CALLS].append(usage)
    st.session_state[RFX_CHAT] = turn + [{"role": "assistant", "content": reply_text(summary, questions),
                                          "summary": summary, "questions": questions, "notes": notes}]
    st.session_state[RFX_PENDING] = None
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
            status_badge(*rfx_status(done, len(items)))
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
        st.caption("Sending is simulated: nothing was emailed.")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption("Replies usually take a few days. See what happens when five replies come back.",
                       width="stretch")
            if st.button("Open the sample event", key="rfx_open_sample", width="content"):
                open_sample()


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
