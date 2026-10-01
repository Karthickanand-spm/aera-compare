"""Decide page: choose an award approach, see the result and its risks, confirm, export.

This page only displays and collects the buyer's choices. Every number comes from
aera/award.py, which reuses the analyst's award logic.
"""

import pandas as pd
import streamlit as st

from aera.analyst import analyst_data, describe_change, display_table, format_inr
from aera.award import (
    APPROACHES, CAPPED, CHEAPEST, MAX_CAP, NOT_AWARDED, SINGLE, AwardSettings, award_signature, award_to_excel,
    build_award, confirm_award, confirm_blockers, vendor_choices, vendor_discounts,
)
from aera.compare import FAIL
from aera.normalize import describe_rates
from ui.state import AWARD_CONFIRMED, comparison_tables, fx_settings, get_event, md, now_text

BEST_SINGLE = "Best available vendor"
SEVERITY_COLORS = {"high": "red", "medium": "orange", "low": "gray"}
REVIEW_MARK = "⚠"
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def render() -> None:
    st.title("Decide")
    event = get_event()
    if event is None:
        st.info("Click **Load sample event** in the sidebar to begin.")
        return
    if not event.replies:
        st.warning("No vendor replies could be read yet. See the sidebar for details.")
        return

    comparison, summary = comparison_tables(event)
    rates, fx_date = fx_settings()
    data = analyst_data(comparison, summary, event.rfx, event.last_year, rates, fx_date)
    st.caption("The app recommends; you confirm the award. Prices still waiting for review on the "
               "Compare page must be confirmed there before the award can be confirmed.")

    settings = _controls(summary)
    award = build_award(data, summary, settings, vendor_discounts(event.replies))
    st.divider()
    if award.is_empty:
        st.warning("No allowed vendor has a counted price on any line. Allow more vendors above.")
        return

    _tiles(award)
    _line_table(award)
    _subtotals(award)
    st.divider()
    _risks(award)
    _assumptions(award)
    st.divider()
    confirmation = _confirm(award)
    fx_text = describe_rates(rates, fx_date)
    st.download_button(
        "Export award (Excel)", award_to_excel(award, event.rfx.rfx_id, fx_text, confirmation, now_text()),
        file_name=f"aera_award_{event.rfx.rfx_id.replace('/', '-')}.xlsx", mime=EXCEL_MIME, on_click="ignore")


# ---------- Controls ----------

def _controls(summary: pd.DataFrame) -> AwardSettings:
    choices = vendor_choices(summary)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Award approach")
        approach = st.radio("Approach", APPROACHES, index=APPROACHES.index(CHEAPEST), key="award_approach",
                            label_visibility="collapsed")
        cap, single = 2, None
        if approach == CAPPED:
            cap = st.slider("Most vendors (N)", 1, MAX_CAP, 2, key="award_cap",
                            help="Every combination of N allowed vendors is tried; the cheapest one "
                                 "covering the most lines wins.")
        include_nc = st.toggle("Include 'Not comparable' rows", value=False, key="award_include_nc",
                  help="Prices for a different spec. Off: they are left out of the award.")
        apply_discounts = st.toggle("Apply recorded discounts if their condition is met", value=False, key="award_discounts",
                  help="Each condition is checked by code against this award's order value. A condition "
                       "code can't read is never applied.")
    with right:
        st.subheader("Allowed vendors")
        st.caption("Only vendors that passed quality are ticked to start with.")
        allowed = []
        for c in choices.itertuples(index=False):
            box, why = st.columns([2, 3])
            if box.checkbox(c.display_name, value=bool(c.default_allowed), key=f"award_allow_{c.display_name}"):
                allowed.append(c.display_name)
            if c.quality_status != "PASS":
                color = "red" if c.quality_status == FAIL else "orange"
                why.markdown(f":{color}[**{c.quality_status}**] {md(c.reason)}")
        if approach == SINGLE:
            pick = st.selectbox("Vendor for the single award", [BEST_SINGLE] + allowed, key="award_single")
            single = None if pick == BEST_SINGLE else pick
    return AwardSettings(approach=approach, allowed=allowed, cap=cap, single_vendor=single,
                         include_not_comparable=include_nc, apply_discounts=apply_discounts)


# ---------- Results ----------

def _tiles(award) -> None:
    st.subheader("Recommended award")
    a, b, c, d = st.columns(4)
    a.metric("Annual total", format_inr(award.total_inr))
    if award.saving_inr is None:
        b.metric("Vs last year", "—", help="No awarded line has a last-year price.")
    else:
        change = describe_change(award.saving_inr, award.saving_pct)
        b.metric("Vs last year", change[0].upper() + change[1:],
                 help=f"On the {len(award.saving_lines)} awarded lines with a last-year price.")
    c.metric("Vendors used", len(award.vendors_used))
    d.metric("Review lines unconfirmed", len(award.unconfirmed))
    st.caption("Awarded to: " + md(", ".join(award.vendors_used)))
    if award.unawarded_lines:
        st.warning(f"Not awarded (no allowed vendor priced them): lines {award.unawarded_lines}. "
                   "They are not in the total.")


def _line_table(award) -> None:
    st.markdown("**Per line**")
    rows = []
    for r in award.lines.itertuples(index=False):
        rows.append({
            "Line": r.rfx_line_id, "Description": r.description, "Qty": f"{r.annual_qty:,}",
            "Awarded vendor": r.display_name or NOT_AWARDED,
            "₹/piece": format_inr(r.price_inr_per_piece, per_unit=True),
            "Annual ₹": format_inr(r.annual_inr),
            "Last year ₹/piece": format_inr(r.last_year_inr_per_piece, per_unit=True),
            "Label": r.label, "Confidence": r.confidence or "—",
            "Review": REVIEW_MARK if r.unconfirmed else "",
        })
    table = pd.DataFrame(rows)
    if award.lines["discount_pct"].notna().any():
        table.insert(5, "Discount", [f"{p:g}% off" if pd.notna(p) else "" for p in award.lines["discount_pct"]])
    st.dataframe(table, hide_index=True)
    if not award.unconfirmed.empty:
        st.caption(f"{REVIEW_MARK} = price needs your review on the Compare page and is not confirmed yet.")


def _subtotals(award) -> None:
    st.markdown("**Per vendor**")
    t = award.vendor_subtotals
    table = pd.DataFrame({
        "Vendor": t["display_name"], "Lines won": t["lines_won"],
        "Quoted annual ₹": [format_inr(v) for v in t["quoted_annual_inr"]],
        "Discount ₹": [format_inr(v) if v else "—" for v in t["discount_inr"]],
        "Annual ₹": [format_inr(v) for v in t["annual_inr"]],
        "Unconfirmed lines": t["unconfirmed_lines"],
    })
    st.dataframe(table, hide_index=True)


def _risks(award) -> None:
    st.subheader("Open risks")
    st.caption("For the vendors in this award only, most severe first.")
    if not award.risks:
        st.success("No open risks for the vendors in this award.")
    for r in award.risks:
        color = SEVERITY_COLORS.get(r["severity"], "gray")
        who = f"**{md(r['vendor'])}**: " if r["vendor"] else ""
        st.markdown(f":{color}[**{r['severity'].upper()}**] {who}{md(r['text'])}")
    for s in award.sensitivity:
        st.markdown(f"**Freight sensitivity: {md(s['vendor'])}**")
        if s.get("error"):
            st.caption(md(f"Can't be worked out: {s['error']}"))
            continue
        gone = ("not gone within the range searched" if s["zero_rate"] is None
                else f"gone at about {format_inr(s['zero_rate'], per_unit=True)}/kg")
        note = (f"Freight added to {s['vendor']}'s prices at each rate (₹ per kg x box weight), and the award "
                f"re-run among {', '.join(s['pool'])}. Saving is against last year on the {s['lines']} lines "
                f"with a last-year price, before any discount; it is {gone}.")
        st.caption(md(note))
        st.dataframe(display_table(s["table"]), hide_index=True)


def _assumptions(award) -> None:
    st.markdown("**Assumptions used in this award**")
    st.markdown("\n".join(f"- {md(a)}" for a in award.assumptions))


# ---------- Confirmation ----------

def _confirm(award) -> dict | None:
    """Show the confirm controls. Returns the confirmation if it matches the award shown."""
    st.subheader("Confirm award")
    record = st.session_state[AWARD_CONFIRMED]
    current = record if record and record["signature"] == award_signature(award) else None
    if current:
        st.success(f"Award confirmed by buyer at {current['confirmed_at']}.")
        if current["note"]:
            st.caption("Note: " + md(current["note"]))
    elif record:
        st.warning(f"An award was confirmed at {record['confirmed_at']} ({record['approach']}, "
                   f"{format_inr(record['total_inr'])}), but the award shown now is different. "
                   "Confirm again to replace it.")

    blockers = confirm_blockers(award)
    if blockers:
        st.warning("Can't confirm yet. Review these awarded prices on the Compare page first: "
                   + md(", ".join(blockers)) + ".")
    note = st.text_area("Note for the record (optional)", key="award_note",
                        placeholder="e.g. Approved by category head; freight to be confirmed with vendor E")
    if st.button("Confirm award", type="primary", disabled=bool(blockers) or current is not None):
        try:
            st.session_state[AWARD_CONFIRMED] = confirm_award(award, note, now_text())
        except ValueError as e:
            st.error(str(e))
            return current
        st.rerun()
    return current
