"""Decide page: choose an award approach, see the result and its risks, confirm, export.

This page only displays and collects the buyer's choices. Every number comes from
aera/award.py, which reuses the analyst's award logic.
"""

import html

import pandas as pd
import streamlit as st

from aera.analyst import analyst_data, describe_change, display_table, format_inr
from aera.award import (
    APPROACHES, CAPPED, CHEAPEST, MAX_CAP, NOT_AWARDED, SINGLE, AwardSettings, award_signature, award_to_excel,
    build_award, confirm_award, vendor_choices, vendor_discounts,
)
from aera.compare import PASS
from aera.normalize import describe_rates
from aera.ui import card, empty_state, metric_row, page_header, status_badge, tint
from ui import compare_page
from ui.state import AWARD_CONFIRMED, comparison_tables, fx_settings, get_event, load_sample, md, now_text

BEST_SINGLE = "Best available vendor"
REVIEW_MARK = "⚠"
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
FREIGHT_RISK = "Freight extra"  # how aera/award.py spots the freight risk it adds the sensitivity to

# Page-only styles. Tile values wrap instead of being cut off with "…" (the saving is a phrase,
# not a number), and the award table follows the theme's text colour in light and dark mode.
_STYLES = """
<style>
.st-key-decide_tiles [data-testid="stMetricValue"],
.st-key-decide_tiles [data-testid="stMetricValue"] * { white-space: normal; overflow: visible;
  text-overflow: clip; font-size: 1.6rem; line-height: 1.25; }
.aera-award-wrap { overflow-x: auto; }
table.aera-award { width: 100%; border-collapse: collapse; font-size: 0.9rem; color: inherit; }
table.aera-award th, table.aera-award td { padding: 0.35rem 0.6rem; text-align: left;
  border-bottom: 1px solid rgba(128, 128, 128, 0.2); vertical-align: top; }
table.aera-award th { font-weight: 600; font-size: 0.8rem; opacity: 0.75; white-space: nowrap; }
table.aera-award td.num, table.aera-award th.num { text-align: right; white-space: nowrap; }
table.aera-award tr.group td { background: rgba(128, 128, 128, 0.1); padding-top: 0.5rem; }
table.aera-award tr.group .meta { opacity: 0.8; margin-left: 0.5rem; }
table.aera-award td.label { white-space: nowrap; }
</style>
"""


def render() -> None:
    page_header("Decide", "Pick an award approach, check the cost and open risks, then confirm.")
    event = get_event()
    if event is None:
        empty_state("No event loaded yet. Load the sample event to build an award from its quotes.", "Load sample event", load_sample)
        return
    if not event.replies:
        st.warning("No vendor replies could be read yet. See the sidebar for details.")
        return
    st.html(_STYLES)

    comparison, summary = comparison_tables(event)
    rates, fx_date = fx_settings()
    data = analyst_data(comparison, summary, event.rfx, event.last_year, rates, fx_date)

    settings = _controls(summary)
    award = build_award(data, summary, settings, vendor_discounts(event.replies))
    if award.is_empty:
        if settings.approach == SINGLE and settings.single_vendor and settings.single_vendor not in settings.allowed:
            st.warning(f"{md(settings.single_vendor)} is picked for the single award but isn't ticked under "
                       "Allowed vendors. Tick it, or pick another vendor, then Apply.")
        else:
            st.warning("No allowed vendor has a counted price on any line. Allow more vendors above, then Apply.")
        return

    _tiles(award)
    _award_table(award)
    _risks(award)
    fx_text = describe_rates(rates, fx_date)
    _confirm(award, lambda confirmation: award_to_excel(award, event.rfx.rfx_id, fx_text, confirmation, now_text()),
             f"aera_award_{event.rfx.rfx_id.replace('/', '-')}.xlsx")


# ---------- Controls ----------

def _controls(summary: pd.DataFrame) -> AwardSettings:
    """All award choices in one form, so nothing recomputes until the buyer clicks Apply."""
    choices = vendor_choices(summary)
    with st.form("award_controls", border=True, enter_to_submit=False):
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("**Award approach**")
            approach = st.radio("Approach", APPROACHES, index=APPROACHES.index(CHEAPEST), key="award_approach",
                                label_visibility="collapsed")
            # A form doesn't rerun until Apply, so these two are always shown and used only by their approach.
            cap = st.slider("N for the capped approach", 1, MAX_CAP, 2, key="award_cap",
                            help="Used only by 'capped at N vendors'. Every combination of N allowed vendors "
                                 "is tried; the cheapest one covering the most lines wins.")
            pick = st.selectbox("Vendor for a single-vendor award", [BEST_SINGLE] + list(choices["display_name"]),
                                key="award_single", help="Used only by 'Single vendor'.")
            include_nc = st.toggle("Include Not comparable rows", value=False, key="award_include_nc",
                                   help="Prices for a different spec. Off: they are left out of the award.")
            apply_discounts = st.toggle("Apply recorded discounts if their condition is met", value=False,
                                        key="award_discounts",
                                        help="Each condition is checked by code against this award's order "
                                             "value. A condition code can't read is never applied.")
        with right:
            st.markdown("**Allowed vendors**")
            st.caption("Only vendors that passed quality are ticked to start with.")
            allowed = []
            for c in choices.itertuples(index=False):
                with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                    if st.checkbox(c.display_name, value=bool(c.default_allowed), key=f"award_allow_{c.display_name}",
                                   width="content"):
                        allowed.append(c.display_name)
                    status_badge(c.quality_status)
                    if c.quality_status != PASS and c.reason:
                        st.caption(md(c.reason), width="stretch")
        st.form_submit_button("Apply", type="primary")
    return AwardSettings(approach=approach, allowed=allowed, cap=cap,
                         single_vendor=None if pick == BEST_SINGLE else pick,
                         include_not_comparable=include_nc, apply_discounts=apply_discounts)


# ---------- Results ----------

def _tiles(award) -> None:
    if award.saving_inr is None:
        vs_last_year = ("Saving vs last year", "—", "No awarded line has a last-year price.")
    else:
        change = describe_change(award.saving_inr, award.saving_pct)
        vs_last_year = ("Saving vs last year", change[0].upper() + change[1:],
                        f"On the {len(award.saving_lines)} awarded lines with a last-year price.")
    with st.container(key="decide_tiles"):
        metric_row([
            ("Annual total", format_inr(award.total_inr), award.settings.describe()),
            vs_last_year,
            ("Vendors used", len(award.vendors_used), ", ".join(award.vendors_used)),
            ("Open review items", len(award.unconfirmed),
             "Awarded prices that still need your review on the Compare page."),
        ])
    if award.unawarded_lines:
        st.warning(f"Not awarded (no allowed vendor priced them): lines {award.unawarded_lines}. "
                   "They are not in the total.")


def _award_table(award) -> None:
    with card():
        st.subheader("Award by vendor", anchor=False)
        st.html(award_table_html(award))
        if not award.unconfirmed.empty:
            st.caption(f"{REVIEW_MARK} = price needs your review on the Compare page and is not confirmed yet.")


def award_table_html(award) -> str:
    """One table: a header row per awarded vendor (name, line count, subtotal), then its lines.
    Lines nobody was awarded come last under their own header."""
    lines = award.lines
    show_discount = lines["discount_pct"].notna().any()
    heads = [("Line", ""), ("Description", ""), ("Qty", "num"), ("₹/piece", "num"), ("Annual ₹", "num"),
             ("Last year ₹/piece", "num"), ("Label", ""), ("Review", "")]
    if show_discount:
        heads.insert(4, ("Discount", ""))
    head = "".join(f'<th class="{cls}" scope="col">{name}</th>' for name, cls in heads)

    subtotals = {r.display_name: r for r in award.vendor_subtotals.itertuples(index=False)}
    groups = [(name, lines[lines["display_name"] == name]) for name in award.vendors_used]
    rest = lines[~lines["display_name"].isin(award.vendors_used)]
    if not rest.empty:
        groups.append((None, rest))

    body = []
    for name, rows in groups:
        n = len(rows)
        meta = f"{n} line{'s' if n != 1 else ''}"
        if name is None:
            title = NOT_AWARDED
        else:
            title, sub = name, subtotals.get(name)
            if sub is not None:
                meta += f" · subtotal {format_inr(sub.annual_inr)}"
                if sub.discount_inr:
                    meta += f" (after {format_inr(sub.discount_inr)} discount)"
        body.append(f'<tr class="group"><td colspan="{len(heads)}"><strong>{html.escape(title)}</strong>'
                    f'<span class="meta">{html.escape(meta)}</span></td></tr>')
        for r in rows.itertuples(index=False):
            cells = [(str(r.rfx_line_id), ""), (html.escape(str(r.description)), ""),
                     (f"{r.annual_qty:,}", "num"),
                     (format_inr(r.price_inr_per_piece, per_unit=True), "num"),
                     (format_inr(r.annual_inr), "num"),
                     (format_inr(r.last_year_inr_per_piece, per_unit=True), "num")]
            if show_discount:
                cells.insert(3, (f"{r.discount_pct:g}% off" if pd.notna(r.discount_pct) else "", ""))
            row = "".join(f'<td class="{cls}">{text}</td>' for text, cls in cells)
            row += f'<td class="label" style="{tint(r.label)}">{html.escape(str(r.label))}</td>'
            review = (f'<span title="Needs review on the Compare page">{REVIEW_MARK}</span>'
                      if r.unconfirmed else "")
            body.append(f"<tr>{row}<td>{review}</td></tr>")
    return (f'<div class="aera-award-wrap"><table class="aera-award"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _risks(award) -> None:
    with card():
        st.subheader("Open risks", anchor=False)
        st.caption("For the vendors in this award only, most severe first.")
        if not award.risks:
            st.success("No open risks for the vendors in this award.")
        sens = {s["vendor"]: s for s in award.sensitivity}
        shown = set()
        for r in award.risks:
            with st.container(horizontal=True, vertical_alignment="top", gap="small"):
                status_badge(r["severity"].upper())
                who = f"**{md(r['vendor'])}**: " if r["vendor"] else ""
                st.markdown(who + md(r["text"]), width="stretch")
            if r["vendor"] in sens and r["text"].startswith(FREIGHT_RISK) and r["vendor"] not in shown:
                shown.add(r["vendor"])
                _sensitivity(sens[r["vendor"]])
        for name, s in sens.items():  # a sensitivity with no freight risk item to sit under
            if name not in shown:
                _sensitivity(s)
        with st.expander("Assumptions used in this award"):
            st.markdown("\n".join(f"- {md(a)}" for a in award.assumptions))


def _sensitivity(s: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**Freight sensitivity: {md(s['vendor'])}**")
        if s.get("error"):
            st.caption(md(f"Can't be worked out: {s['error']}"))
            return
        gone = ("not gone within the range searched" if s["zero_rate"] is None
                else f"gone at about {format_inr(s['zero_rate'], per_unit=True)}/kg")
        note = (f"Freight added to {s['vendor']}'s prices at each rate (₹ per kg x box weight), and the award "
                f"re-run among {', '.join(s['pool'])}. Saving is against last year on the {s['lines']} lines "
                f"with a last-year price, before any discount; it is {gone}.")
        st.caption(md(note))
        st.dataframe(display_table(s["table"]), hide_index=True)


# ---------- Confirmation ----------

def blocked_lines_text(award) -> str:
    """'Ganesh 28, 29, 30; Kumar 5': unconfirmed awarded lines, grouped by vendor."""
    parts = []
    for name, g in award.unconfirmed.groupby("display_name", sort=False):
        parts.append(f"{name} " + ", ".join(str(i) for i in g["rfx_line_id"]))
    return "; ".join(parts)


def _confirm(award, excel, file_name: str) -> None:
    """Confirm controls and the export button. `excel(confirmation)` builds the workbook."""
    with card():
        st.subheader("Confirm award", anchor=False)
        record = st.session_state[AWARD_CONFIRMED]
        current = record if record and record["signature"] == award_signature(award) else None
        blocked = not award.unconfirmed.empty

        if current:
            st.success(f"Award confirmed by buyer at {current['confirmed_at']}."
                       + (f"  \nNote: {md(current['note'])}" if current["note"] else ""))
        elif record:
            st.warning(f"An award was confirmed at {record['confirmed_at']} ({record['approach']}, "
                       f"{format_inr(record['total_inr'])}), but the award shown now is different. "
                       "Confirm again to replace it.")
        if blocked and not current:
            st.warning("Resolve these lines before confirming: " + md(blocked_lines_text(award)))
        note = ""
        if not blocked and not current:
            note = st.text_input("Note for the record (optional)", key="award_note",
                                 placeholder="e.g. Approved by category head; freight to be confirmed with vendor E")

        with st.container(horizontal=True, gap="small"):
            if not current:
                if blocked:
                    if st.button("Go to Compare to review", type="primary"):
                        st.switch_page(compare_page.page())
                elif st.button("Confirm award", type="primary"):
                    try:
                        st.session_state[AWARD_CONFIRMED] = confirm_award(award, note, now_text())
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.rerun()
            st.download_button("Export award (Excel)", excel(current), file_name=file_name, mime=EXCEL_MIME,
                               on_click="ignore", type="primary" if current else "secondary")
