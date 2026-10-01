"""Compare page: vendor cards, the price grid, line inspector and the review panel.

This page only displays. Prices, labels, confidence and buyer decisions are all
worked out in aera/compare.py.
"""

import html

import pandas as pd
import streamlit as st

from aera.compare import (
    BUYER_EDIT, COMPARABLE, NOT_COMPARABLE, NOT_QUOTED, PASS, UNCLEAR, USE_ALTERNATIVE,
    USE_EXTRACTED, WITH_ASSUMPTION, buyer_decision, find_snippet_span,
)
from aera.event import Event
from aera.normalize import describe_rates
from aera.ui import badge_md, card, empty_state, metric_row, page_header, status_badge, tint
from ui.state import DECISIONS, comparison_tables, fx_settings, get_event, load_sample, md, now_text

FORMAT_NAMES = {"excel": "Excel", "word": "Word", "email": "Email", "pdf": "PDF", "image": "Photo"}
LABEL_MARKS = {COMPARABLE: "", WITH_ASSUMPTION: " ≈", NOT_COMPARABLE: " ≠"}
REVIEW_MARK, CONFIRMED_MARK = "⚠", "✓"
CARDS_PER_ROW = 3
CONTEXT_LINES = 4  # lines shown either side of a highlighted snippet


def page() -> st.Page:
    """The navigation entry. Built here so other pages can switch to the same page."""
    return st.Page(render, title="Compare", url_path="compare", default=True)


def render() -> None:
    page_header("Compare", "Every vendor's price side by side, with a clear mark on any number that "
                "can't be compared like for like.",
                "confirm the values flagged for review below, then ask questions on Ask.")
    event = get_event()
    if event is None:
        empty_state("No event loaded yet. Load the sample event to see five vendor replies compared.",
                    "Load sample event", load_sample)
        return

    rfx = event.rfx
    st.markdown(f"**{md(rfx.title)}**")
    st.caption(f"{rfx.rfx_id} · {rfx.buyer} · issued {rfx.issued} · due {rfx.due}")
    if not event.replies:
        st.warning("No vendor replies could be read yet. See the sidebar for details.")
        return

    comparison, summary = comparison_tables(event)
    _headline(event, comparison)
    _vendor_cards(event, summary)
    st.divider()
    _grid(event, comparison)
    st.divider()
    _inspect(event, comparison)
    st.divider()
    _review_panel(comparison)


def _headline(event: Event, comparison: pd.DataFrame) -> None:
    review = comparison[comparison["needs_review"].astype(bool)]
    metric_row([
        ("Vendor replies", len(event.replies)),
        ("RFx lines", len(event.rfx.lines)),
        ("Not quoted", int((comparison["label"] == NOT_QUOTED).sum()),
         "Line and vendor pairs the vendor skipped. Never counted as zero."),
        ("Waiting for your review", int((~review["buyer_confirmed"].astype(bool)).sum()),
         "Low-confidence values. They count only after you confirm them in the review panel below."),
    ])


# ---------- Vendor cards ----------

def _vendor_cards(event: Event, summary: pd.DataFrame) -> None:
    st.subheader("Vendors")
    n_lines = len(event.rfx.lines)
    rows = list(summary.itertuples(index=False))
    for start in range(0, len(rows), CARDS_PER_ROW):
        cols = st.columns(CARDS_PER_ROW)
        for col, s in zip(cols, rows[start:start + CARDS_PER_ROW]):
            with col, card():
                _vendor_card(event, s, n_lines)


def _vendor_card(event: Event, s, n_lines: int) -> None:
    f = event.files.get(s.source_file)
    fmt = FORMAT_NAMES.get(f.kind, f.kind) if f else "Unknown"
    st.markdown(f"**{md(s.display_name)}**")
    st.caption(f"{fmt} reply · {s.source_file}")
    st.markdown(f"Lines quoted **{s.lines_quoted}** / {n_lines} · comparable **{s.lines_comparable}**")
    freight = (s.freight or "not stated").capitalize()
    payment = "Not stated" if _missing(s.payment_days) else f"{s.payment_days:g} days"
    st.markdown(f"Freight: **{freight}** · Payment: **{payment}**")

    status_badge(f"Quality {s.quality_status}", s.quality_status)
    with st.expander("Quality reasons"):
        st.markdown("\n".join(f"- {md(r)}" for r in s.quality_reasons) or "No checks recorded.")

    if s.open_risks:
        st.markdown("**Open risks**")
        st.markdown("\n\n".join(
            f"{badge_md(r['severity'].upper())} {md(r['text'])}"
            for r in s.open_risks))
    else:
        st.caption("No open risks.")
    if s.discounts:
        st.markdown("**Discounts**")
        st.markdown("\n".join(f"- {md(d)}" for d in s.discounts))
    if s.lines_needing_review:
        st.markdown(f"{REVIEW_MARK} {s.lines_needing_review} value(s) waiting for your review")


# ---------- Grid ----------

def _grid(event: Event, comparison: pd.DataFrame) -> None:
    st.subheader("Price comparison")
    rates, fx_date = fx_settings()
    gaps = sorted({c for c in comparison["missing_fx_currency"] if isinstance(c, str)})
    no_rate = (f" No rate yet for {', '.join(gaps)}: enter it in the sidebar to convert those prices."
               if gaps else "")
    st.caption(f"INR per piece. FX: {describe_rates(rates, fx_date)}. "
               f"Not comparable and Not quoted cells are left out of totals.{no_rate}")
    st.markdown(
        f"{badge_md(COMPARABLE)} `12.34` · {badge_md(WITH_ASSUMPTION)} `12.34 ≈` · "
        f"{badge_md(NOT_COMPARABLE)} `12.34 ≠` · {badge_md(NOT_QUOTED)} vendor skipped the line (never 0) · "
        f"`{REVIEW_MARK}` needs your review · `{CONFIRMED_MARK}` confirmed by you"
    )

    names = _column_names(comparison)
    cells = comparison.assign(cell=comparison.apply(_cell_text, axis=1))
    wide = cells.pivot(index="rfx_line_id", columns="vendor", values="cell")
    table = pd.DataFrame({
        "Line": [ln.line_id for ln in event.rfx.lines],
        "Description": [ln.description for ln in event.rfx.lines],
        "Annual qty": [ln.annual_qty for ln in event.rfx.lines],
    })
    for vendor, name in names.items():
        table[name] = [wide.at[ln.line_id, vendor] for ln in event.rfx.lines]

    styled = table.style.map(_cell_style, subset=list(names.values()))
    column_config = {
        "Line": st.column_config.NumberColumn(width="small", pinned=True),
        "Description": st.column_config.TextColumn(width="medium", pinned=True),
        "Annual qty": st.column_config.NumberColumn(format="localized", width="small"),
    }
    st.dataframe(styled, hide_index=True, column_config=column_config,
                 height=min(38 + 35 * len(table), 1100))


def _column_names(comparison: pd.DataFrame) -> dict[str, str]:
    """vendor -> column header (display name, made unique if two vendors share one)."""
    names: dict[str, str] = {}
    for vendor, display in comparison[["vendor", "display_name"]].drop_duplicates().itertuples(index=False):
        name, n = display, 2
        while name in names.values():
            name, n = f"{display} ({n})", n + 1
        names[vendor] = name
    return names


def _cell_text(r) -> str:
    if r["label"] == NOT_QUOTED:
        text = "Not quoted"
    elif _missing(r["price_inr_per_piece"]):
        fx_gap = r["missing_fx_currency"]
        text = (f"No {fx_gap} rate" if isinstance(fx_gap, str) else "No price") + LABEL_MARKS.get(r["label"], "")
    else:
        text = f"{r['price_inr_per_piece']:,.2f}" + LABEL_MARKS.get(r["label"], "")
    if r["buyer_confirmed"]:
        text += f" {CONFIRMED_MARK}"
    elif r["needs_review"]:
        text += f" {REVIEW_MARK}"
    return text


def _cell_style(text: str) -> str:
    """Tints in the badge colours. Text keeps the theme colour so it stays readable in light and dark mode."""
    if text.endswith(REVIEW_MARK):
        return tint(UNCLEAR)
    if text.startswith("Not quoted"):
        return tint(NOT_QUOTED) + "; font-style: italic"
    if "≠" in text:
        return tint(NOT_COMPARABLE)
    if text.endswith(CONFIRMED_MARK):
        return tint(PASS)
    return ""


# ---------- Inspect one line ----------

def _inspect(event: Event, comparison: pd.DataFrame) -> None:
    st.subheader("Inspect line")
    rfx = event.rfx
    line_id = st.selectbox("Inspect line", [ln.line_id for ln in rfx.lines], key="inspect_line",
                           format_func=lambda i: f"{i} · {rfx.line(i).description}",
                           label_visibility="collapsed")
    ln = rfx.line(line_id)
    st.caption(f"RFx spec: {ln.ply} · {ln.board_spec} · {ln.size} · {ln.print} · "
               f"{ln.annual_qty:,} {ln.uom} a year · nominal weight {ln.nominal_weight_g:g} g")

    rows = comparison[comparison["rfx_line_id"] == line_id]
    names = _column_names(comparison)
    tabs = st.tabs([f"{names[r['vendor']]} · {_cell_text(r)}" for _, r in rows.iterrows()])
    for tab, (_, r) in zip(tabs, rows.iterrows()):
        with tab:
            _cell_detail(event, r)


def _cell_detail(event: Event, r) -> None:
    r = _clean(r)
    price = r["price_inr_per_piece"]
    status_badge(r["label"])
    metric_row([
        ("INR per piece", "Not quoted" if r["label"] == NOT_QUOTED
         else "No price" if _missing(price) else f"{price:,.2f}"),
        ("Confidence", r["confidence"].capitalize()),
        ("Assumptions", len(r["assumptions"])),
    ])

    if r["buyer_decision"]:
        st.success(f"{CONFIRMED_MARK} {r['buyer_decision']}")
    elif r["needs_review"]:
        st.warning(f"{REVIEW_MARK} Needs your review before it counts. See the review panel below.")

    st.markdown(f"**Vendor wrote:** {md(r['raw_price_text']) if r['raw_price_text'] else '—'}")
    if r["quoted_spec"]:
        st.markdown(f"**Different spec quoted:** {md(r['quoted_spec'])}")
    if r["notes"]:
        st.markdown(f"**How it was read:** {md(r['notes'])}")

    st.markdown("**Assumptions**")
    st.markdown("\n".join(f"- {md(a)}" for a in r["assumptions"]) or "None.")
    st.markdown("**Why this confidence**")
    st.markdown("\n".join(f"- {md(x)}" for x in r["confidence_reasons"]) or "No checks recorded.")

    page = f", page {int(r['page'])}" if not _missing(r["page"]) else ""
    st.markdown(f"**Source:** {md(r['source_file'])}{md(page)}")
    if r["source_snippet"]:
        st.code(r["source_snippet"], language=None, wrap_lines=True)
    else:
        st.caption("No source snippet (the vendor did not mention this line).")

    if r["alternatives"]:
        st.markdown("**Other possible readings**")
        for alt in r["alternatives"]:
            alt_price = alt["price_inr_per_piece"]
            shown = "no price" if alt_price is None else f"INR {alt_price:,.2f} per piece"
            extra = f" ({md('; '.join(alt['assumptions']))})" if alt["assumptions"] else ""
            st.markdown(f"- '{md(alt['raw_price_text'])}' → {shown}{extra}")

    with st.expander("View source file"):
        _source_view(event, r)


def _source_view(event: Event, r) -> None:
    f = event.files.get(r["source_file"])
    if f is None:
        st.info("The original file is not available in this session.")
        return
    key = f"{r['rfx_line_id']}|{r['vendor']}"
    if f.kind == "image":
        st.image(f.data, caption=f.name)
    elif f.kind == "pdf":
        page = f" The price is on page {int(r['page'])}." if not _missing(r["page"]) else ""
        st.caption(f"PDFs can't be shown inline here. Download it to check the snippet.{page}")
        st.download_button(f"Download {f.name}", f.data, file_name=f.name, mime="application/pdf",
                           key=f"download|{key}", on_click="ignore")
    elif f.text:
        _highlighted_text(f.text, r["source_snippet"], key)
    else:
        st.info("No text was extracted from this file.")


def _highlighted_text(text: str, snippet: str | None, key: str) -> None:
    span = find_snippet_span(snippet, text)
    if span is None:
        if snippet:
            st.warning("The snippet was not found in the extracted text. Showing the full text.")
        st.html(_text_box(text, None))
        return
    start, end = span
    lines = text.split("\n")
    first, last = text.count("\n", 0, start), text.count("\n", 0, end)
    lo, hi = max(0, first - CONTEXT_LINES), min(len(lines), last + CONTEXT_LINES + 1)
    offset = sum(len(x) + 1 for x in lines[:lo])
    excerpt = "\n".join(lines[lo:hi])
    st.caption("Snippet highlighted, with the lines around it:")
    st.html(_text_box(excerpt, (start - offset, end - offset)))
    if st.toggle("Show full text", key=f"fulltext|{key}"):
        st.html(_text_box(text, span))


def _text_box(text: str, span: tuple[int, int] | None) -> str:
    if span:
        s, e = span
        body = (html.escape(text[:s]) + "<mark>" + html.escape(text[s:e]) + "</mark>"
                + html.escape(text[e:]))
    else:
        body = html.escape(text)
    return ('<div style="white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; '
            'max-height: 420px; overflow-y: auto; padding: 0.75rem; '
            'border: 1px solid rgba(128,128,128,0.3); border-radius: 0.5rem;">' + body + "</div>")


# ---------- Review panel ----------

def _review_panel(comparison: pd.DataFrame) -> None:
    review = comparison[comparison["needs_review"].astype(bool)]
    waiting = int((~review["buyer_confirmed"].astype(bool)).sum())
    st.subheader("Review")
    if review.empty:
        st.success("Nothing needs review. Every value passed the checks.")
        return
    st.caption(f"{waiting} waiting · {len(review) - waiting} confirmed. Low-confidence values "
               "need your decision before they count.")
    for _, r in review.iterrows():
        with card():
            _review_row(r)


def _review_row(r) -> None:
    r = _clean(r)
    key = (int(r["rfx_line_id"]), r["vendor"])
    wkey = f"{key[0]}|{key[1]}"
    mark = CONFIRMED_MARK if r["buyer_confirmed"] else REVIEW_MARK
    st.markdown(f"{mark} **{md(r['display_name'])}** · line {key[0]} {md(r['description'])}")
    price = r["price_inr_per_piece"]
    shown = ("Not quoted" if r["label"] == NOT_QUOTED
             else "no price" if _missing(price) else f"INR {price:,.2f} per piece")
    st.markdown(f"Vendor wrote: {md(r['raw_price_text']) if r['raw_price_text'] else '—'} → "
                f"**{shown}** · {badge_md(r['label'])} · confidence {r['confidence']}")
    with st.expander("Why it needs review"):
        st.markdown("\n".join(f"- {md(x)}" for x in r["confidence_reasons"]))

    if r["buyer_confirmed"]:
        st.success(r["buyer_decision"])
        st.button("Undo decision", key=f"undo|{wkey}", on_click=_undo, args=(key,))
        return

    if r["missing_fx_currency"]:
        _fx_gap_actions(r, key, wkey)
        return

    use_label = "Keep as Not quoted" if r["label"] == NOT_QUOTED else "Use this price"
    alts = [(i, a) for i, a in enumerate(r["alternatives"]) if a["price_inr_per_piece"] is not None]
    cols = st.columns(2 + len(alts))
    cols[0].button(use_label, key=f"use|{wkey}", on_click=_decide,
                   args=(key, USE_EXTRACTED), width="stretch")
    for col, (i, alt) in zip(cols[1:], alts):
        col.button(f"Use alternative: '{alt['raw_price_text']}' → INR {alt['price_inr_per_piece']:,.2f}",
                   key=f"alt{i}|{wkey}", on_click=_decide, args=(key, USE_ALTERNATIVE),
                   kwargs={"alternative_index": i}, width="stretch")
    with cols[-1].popover("Edit", width="stretch"):
        _manual_price(price, key, wkey)


def _manual_price(price, key, wkey) -> None:
    value = st.number_input("INR per piece", min_value=0.01, step=0.01, format="%.2f",
                            value=None if _missing(price) else float(price),
                            key=f"edit_value|{wkey}")
    st.button("Save price", key=f"save|{wkey}", disabled=value is None, on_click=_decide,
              args=(key, BUYER_EDIT), kwargs={"value_key": f"edit_value|{wkey}"})


def _fx_gap_actions(r: dict, key, wkey) -> None:
    """A price in a currency with no rate: the fix is a rate in the sidebar, not a typed price."""
    code = r["missing_fx_currency"]
    st.info(f"Not comparable: no FX rate for {code}. Enter a **{code} rate** in the sidebar and this "
            f"line converts automatically, with the rate shown as an assumption.", icon=":material/currency_exchange:")
    with st.popover("Type an INR price instead (last resort)"):
        st.caption(f"Only if you can't get a {code} rate. This skips the conversion, so no FX "
                   "assumption is recorded for this line.")
        _manual_price(None, key, wkey)


def _decide(key, choice, alternative_index=None, value_key=None) -> None:
    value = st.session_state.get(value_key) if value_key else None
    st.session_state[DECISIONS][key] = buyer_decision(
        choice, now_text(), alternative_index=alternative_index, value_inr_per_piece=value)


def _undo(key) -> None:
    st.session_state[DECISIONS].pop(key, None)


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _clean(r) -> dict:
    """A comparison row with blank (NaN) cells read back as None, so `if r[...]` is safe."""
    return {k: None if _missing(v) else v for k, v in r.items()}
