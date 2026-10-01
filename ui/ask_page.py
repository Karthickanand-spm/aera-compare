"""Ask page: plain-English questions about the comparison.

aera/analyst.py classifies the question, works out the answer in code (aera/analyses.py)
and has Claude word it from the facts. This page only displays, and shows the code under
every answer.
"""

import altair as alt
import streamlit as st

from aera.analyst import (
    AnalystError, Answer, analyst_data, ask, display_table, format_money, money_header, money_kind,
    to_excel_bytes,
)
from aera.clarify import vendors_mentioned
from aera.compare import display_name, vendor_name
from aera.ui import card, empty_state, has_high_risk, page_header, split_headline, watch_box
from ui import clarify_page
from ui.state import API_CALLS, ASK_HISTORY, comparison_tables, fx_settings, get_event, load_sample, md

# Starting points only: each one is sent to Claude like any typed question.
SUGGESTED = [
    "Split the award among vendors who cleared quality. Total vs last year?",
    "Is Ganesh cheapest once freight is in?",
    "Which vendor is best?",
    "Cap the award at two vendors",
    "Which lines did Indus not quote?",
    "Show total annual cost by vendor",
]
PENDING = "ask_pending"  # question waiting to be answered on this run
CHIPS = "ask_chips"  # the suggested-question pills
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def render() -> None:
    page_header("Ask", "Ask about the comparison in plain English. "
                "The app works out the answer in code and shows its working.")
    event = get_event()
    if event is None:
        empty_state("No event loaded yet. Load the sample event to ask questions about its quotes.", "Load sample event", load_sample)
        return
    if not event.replies:
        st.warning("No vendor replies could be read yet. See the sidebar for details.")
        return

    _question_box()
    question = st.session_state.pop(PENDING, None)
    if question:
        _answer(event, question)
    _history(event)


def _question_box() -> None:
    st.pills("Try one of these", SUGGESTED, key=CHIPS, on_change=_chip_picked)
    with st.form("ask_form", clear_on_submit=True, enter_to_submit=True):
        typed = st.text_input("Your question",
                              placeholder="e.g. How much would we save on line 3 by switching vendor?")
        if st.form_submit_button("Ask", type="primary") and typed.strip():
            st.session_state[PENDING] = typed.strip()


def _chip_picked() -> None:
    """Ask the clicked chip's question, and clear the chip so it can be clicked again."""
    question = st.session_state[CHIPS]
    st.session_state[CHIPS] = None
    if question:
        st.session_state[PENDING] = question


def _answer(event, question: str) -> None:
    comparison, summary = comparison_tables(event)
    rates, fx_date = fx_settings()
    data = analyst_data(comparison, summary, event.rfx, event.last_year, rates, fx_date)
    with st.spinner("Working it out..."):
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
    newest, *older = history
    with card():
        st.caption(f"You asked: {md(newest.question)}")
        _show(newest, vendors)
    if older:
        st.subheader("Earlier answers", anchor=False)
    for answer in older:
        with st.expander(md(answer.question), key=f"ask_old_{answer.id}"):
            _show(answer, vendors)


def _show(a: Answer, vendors: dict[str, str]) -> None:
    """One answer: headline, the rest of the text, table or chart, what to watch, the working, the tag."""
    if a.error:
        st.warning(a.error)
    else:
        head, rest = split_headline(a.text)
        st.markdown(f"**{md(head)}**")
        if rest:
            st.markdown(md(rest))
        if a.unchecked_numbers:
            st.warning("Check these numbers: they appear in the sentences above but not in the "
                       "calculated result: " + ", ".join(a.unchecked_numbers))
        if a.answer_type == "chart":
            _chart(a)
        elif a.answer_type == "table" and a.table is not None:
            st.dataframe(display_table(a.table), hide_index=True)
        for title, extra in a.extra_tables:
            st.markdown(f"**{md(title)}**")
            st.dataframe(display_table(extra), hide_index=True)
        for sv in a.sensitivity:
            _freight_table(sv)

    if not a.data_sufficient:
        st.markdown("**The data can't fully answer this. Missing:**")
        st.markdown("\n".join(f"- {md(m)}" for m in a.missing_data) or "- (not stated)")
        if a.missing_data:
            _clarify_buttons(a, vendors)

    if a.table is not None and not a.error:
        st.download_button("Download as Excel", to_excel_bytes(a), file_name=f"aera_answer_{a.id}.xlsx",
                           mime=EXCEL_MIME, key=f"xlsx_{a.id}", on_click="ignore")

    if a.caveats and not a.error:
        with watch_box(a.id, expanded=has_high_risk(a.caveats)):
            st.markdown("\n".join(f"- {md(c)}" for c in a.caveats))

    with st.expander("Show the working", key=f"working_{a.id}"):
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

    tag = f"Answered as: {md(a.tag)} · " if a.tag else ""
    st.caption(f"{tag}{a.asked_at} · {len(a.usages)} Claude calls · ~${a.cost_usd:.4f}")


def _clarify_buttons(a: Answer, vendors: dict[str, str]) -> None:
    """One button per vendor the missing data names; a vendor picker if it names none.

    `vendors` maps display name -> vendor. The button opens the Clarify page with that
    vendor first and the missing items added to its list.
    """
    names = list(vendors)
    per_vendor = {n: [m for m in a.missing_data if n in vendors_mentioned(m, names)] for n in names}
    named = [n for n in names if per_vendor[n]]
    for n in named:
        if st.button(f"Draft clarification to {n}", key=f"clarify_{a.id}_{n}", icon=":material/mail:"):
            clarify_page.open_for(vendors[n], per_vendor[n])
    if named:
        return
    pick = st.selectbox("Vendor to ask", names, key=f"clarify_pick_{a.id}",
                        help="The missing data doesn't name a vendor. Pick who should be asked.")
    if st.button("Draft clarification", key=f"clarify_{a.id}", icon=":material/mail:"):
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
