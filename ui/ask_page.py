"""Ask page: plain-English questions about the comparison.

Claude writes pandas code; aera/analyst.py runs it and words the answer from the
result. This page only displays, and shows the code under every answer.
"""

import altair as alt
import streamlit as st

from aera.analyst import (
    AnalystError, Answer, analyst_data, ask, display_table, format_money, money_header, money_kind,
    to_excel_bytes,
)
from aera.clarify import vendors_mentioned
from aera.compare import display_name, vendor_name
from ui import clarify_page
from ui.state import API_CALLS, ASK_HISTORY, comparison_tables, fx_settings, get_event, md

# Starting points only: each one is sent to Claude like any typed question.
EXAMPLES = [
    "Which vendor is cheapest on the lines every vendor quoted?",
    "Total annual cost per vendor, only for vendors that cleared quality",
    "Chart each vendor's savings against last year's prices",
    "Which vendors have high-severity open risks, and what are they?",
]
PENDING = "ask_pending"  # question waiting to be answered on this run
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def render() -> None:
    st.title("Ask")
    event = get_event()
    if event is None:
        st.info("Click **Load sample event** in the sidebar to begin.")
        return
    if not event.replies:
        st.warning("No vendor replies could be read yet. See the sidebar for details.")
        return

    st.caption("Ask about the comparison in plain English. Claude writes pandas code, the code "
               "does all the maths, and the code is shown under every answer. Each question "
               "uses API credit (see the sidebar).")
    _question_box()
    question = st.session_state.pop(PENDING, None)
    if question:
        _answer(event, question)
    _history(event)


def _question_box() -> None:
    st.markdown("**Try one of these**")
    cols = st.columns(len(EXAMPLES))
    for i, (col, q) in enumerate(zip(cols, EXAMPLES)):
        col.button(q, key=f"example_{i}", on_click=_set_pending, args=(q,), width="stretch")
    with st.form("ask_form", clear_on_submit=True):
        typed = st.text_input("Your question",
                              placeholder="e.g. How much would we save on line 3 by switching vendor?")
        if st.form_submit_button("Ask", type="primary") and typed.strip():
            st.session_state[PENDING] = typed.strip()


def _set_pending(question: str) -> None:
    st.session_state[PENDING] = question


def _answer(event, question: str) -> None:
    comparison, summary = comparison_tables(event)
    rates, fx_date = fx_settings()
    data = analyst_data(comparison, summary, event.rfx, event.last_year, rates, fx_date)
    with st.spinner("Claude is working out the analysis..."):
        try:
            answer = ask(question, data)
        except AnalystError as e:
            st.error(f"Could not answer \"{question}\": {e}")
            return
    st.session_state[API_CALLS].extend(answer.usages)
    st.session_state[ASK_HISTORY].insert(0, answer)
    st.rerun()  # redraw so the sidebar cost includes this answer


# ---------- History ----------

def _history(event) -> None:
    history = st.session_state[ASK_HISTORY]
    if not history:
        return
    vendors = {display_name(vendor_name(ext)): vendor_name(ext) for ext in event.replies}
    st.divider()
    st.subheader("Answers (newest first)")
    for answer in history:
        with st.container(border=True):
            _show(answer, vendors)


def _show(a: Answer, vendors: dict[str, str]) -> None:
    st.markdown(f"**Q: {md(a.question)}**")
    st.caption(f"{a.asked_at} · {len(a.usages)} Claude calls · ~${a.cost_usd:.4f}")

    if a.error:
        st.warning(a.error)
    else:
        st.markdown(md(a.text))
        if a.unchecked_numbers:
            st.warning("Check these numbers: they appear in the sentences above but not in the "
                       "calculated result: " + ", ".join(a.unchecked_numbers))
        if a.answer_type == "chart":
            _chart(a)
        elif a.answer_type == "table" and a.table is not None:
            st.dataframe(display_table(a.table), hide_index=True)
        for sv in a.sensitivity:
            _freight_table(sv)

    if a.caveats and not a.error:
        st.warning("**Caveats**\n\n" + "\n".join(f"- {md(c)}" for c in a.caveats))

    if not a.data_sufficient:
        st.markdown("**The data can't fully answer this. Missing:**")
        st.markdown("\n".join(f"- {md(m)}" for m in a.missing_data) or "- (not stated)")
        if a.missing_data:
            _clarify_buttons(a, vendors)

    if a.table is not None and not a.error:
        st.download_button("Download as Excel", to_excel_bytes(a), file_name=f"aera_answer_{a.id}.xlsx",
                           mime=EXCEL_MIME, key=f"xlsx_{a.id}", on_click="ignore")

    with st.expander("Show the working"):
        if a.explanation:
            st.markdown(md(a.explanation))
        if a.code:
            st.code(a.code, language="python")
        else:
            st.caption("No code was run.")
        for err in a.code_errors:
            st.caption(f"Error: {err}")
        for note in a.wording_notes:
            st.caption(md(note))
        if a.facts:
            st.markdown("**Facts the answer was written from** (formatted by code)")
            st.json(a.facts, expanded=False)
        if a.result is not None and a.answer_type != "table":
            st.markdown("**Result the answer was written from**")
            if a.table is not None:
                st.dataframe(display_table(a.table), hide_index=True)
            else:
                st.code(str(a.result), language=None)


def _clarify_buttons(a: Answer, vendors: dict[str, str]) -> None:
    """One button per vendor the missing data names; a vendor picker if it names none.

    `vendors` maps display name -> vendor. The button opens the Clarify page with that
    vendor first and the missing items added to its list.
    """
    names = list(vendors)
    per_vendor = {n: [m for m in a.missing_data if n in vendors_mentioned(m, names)] for n in names}
    named = [n for n in names if per_vendor[n]]
    for n in named:
        if st.button(f"Draft clarification to {n}", key=f"clarify_{a.id}_{n}"):
            clarify_page.open_for(vendors[n], per_vendor[n])
    if named:
        return
    pick = st.selectbox("Vendor to ask", names, key=f"clarify_pick_{a.id}",
                        help="The missing data doesn't name a vendor. Pick who should be asked.")
    if st.button("Draft clarification to vendor", key=f"clarify_{a.id}"):
        clarify_page.open_for(vendors[pick], a.missing_data)


def _freight_table(sv: dict) -> None:
    st.markdown(f"**Freight sensitivity: {md(sv['vendor'])}**")
    gone = ("not gone within the range searched" if sv["zero_rate"] is None
            else f"gone at about {format_money(sv['zero_rate'], 'kg')}")
    note = (f"Freight added to {sv['vendor']}'s prices at each rate (₹ per kg x box weight), and the award "
            f"re-run so lines move to the next cheapest vendor. Saving is against last year on the "
            f"{sv['lines']} lines that have a last-year price; it is {gone}.")
    if sv["lines_without_last_year"]:
        note += f" Lines with no last-year price, left out: {sv['lines_without_last_year']}."
    st.caption(md(note))
    st.dataframe(display_table(sv["table"]), hide_index=True)


def _chart(a: Answer) -> None:
    x, y = a.chart_spec["x"], a.chart_spec["y"]
    data = a.table[[x, y]].copy()
    kind = money_kind(y)
    # Bars use the plain number; the tooltip shows it in ₹ / lakh / crore.
    data["shown"] = [format_money(v, kind) for v in data[y]] if kind else data[y].astype(str)
    horizontal = a.chart_spec.get("kind") == "horizontal_bar"
    cat = alt.X(field=x, type="nominal", sort="-y", title=x) if not horizontal else \
        alt.Y(field=x, type="nominal", sort="-x", title=x)
    val = alt.Y(field=y, type="quantitative", title=money_header(y)) if not horizontal else \
        alt.X(field=y, type="quantitative", title=money_header(y))
    chart = alt.Chart(data).mark_bar().encode(
        cat, val, tooltip=[alt.Tooltip(field=x, type="nominal"),
                           alt.Tooltip(field="shown", type="nominal", title=money_header(y))]
    )
    st.altair_chart(chart, width="stretch")
