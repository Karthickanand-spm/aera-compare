"""Compare page display choices: which cells to highlight, which lines have issues, headline counts."""

import pandas as pd

from aera.compare import (
    COMPARABLE, FAIL, HIGH, LOW, MEDIUM, NOT_COMPARABLE, NOT_QUOTED, PASS, WITH_ASSUMPTION,
)
from ui.compare_page import cheapest_cells, high_risk_count, lines_with_issues, quote_md, unconfirmed


def _cell(line, vendor, price, label=COMPARABLE, review=False, confirmed=False):
    return {"rfx_line_id": line, "vendor": vendor, "price_inr_per_piece": price, "label": label,
            "included_in_totals": label in (COMPARABLE, WITH_ASSUMPTION),
            "needs_review": review, "buyer_confirmed": confirmed}


def _summary(**status):
    return pd.DataFrame([{"vendor": v, "quality_status": s, "open_risks": []} for v, s in status.items()])


def test_cheapest_is_lowest_counted_price_per_line():
    comp = pd.DataFrame([_cell(1, "a", 10.0), _cell(1, "b", 9.0, WITH_ASSUMPTION), _cell(2, "a", 5.0),
                         _cell(2, "b", 6.0)])
    assert cheapest_cells(comp, _summary(a=PASS, b=PASS)) == {(1, "b"), (2, "a")}


def test_cheapest_skips_vendors_that_did_not_pass_quality():
    comp = pd.DataFrame([_cell(1, "a", 10.0), _cell(1, "b", 9.0)])
    assert cheapest_cells(comp, _summary(a=PASS, b=FAIL)) == {(1, "a")}


def test_cheapest_skips_not_comparable_not_quoted_and_unconfirmed():
    comp = pd.DataFrame([
        _cell(1, "a", 10.0),
        _cell(1, "b", 1.0, NOT_COMPARABLE),
        _cell(1, "c", None, NOT_QUOTED),
        _cell(1, "d", 2.0, review=True),
    ])
    assert cheapest_cells(comp, _summary(a=PASS, b=PASS, c=PASS, d=PASS)) == {(1, "a")}


def test_confirmed_review_value_can_be_cheapest_and_ties_all_count():
    comp = pd.DataFrame([_cell(1, "a", 8.0), _cell(1, "b", 8.0, review=True, confirmed=True)])
    assert cheapest_cells(comp, _summary(a=PASS, b=PASS)) == {(1, "a"), (1, "b")}


def test_no_passing_vendor_means_no_highlight():
    comp = pd.DataFrame([_cell(1, "a", 8.0)])
    assert cheapest_cells(comp, _summary(a=FAIL)) == set()


def test_lines_with_issues():
    comp = pd.DataFrame([
        _cell(1, "a", 10.0), _cell(1, "b", 9.0, WITH_ASSUMPTION),  # clean
        _cell(2, "a", 10.0), _cell(2, "b", None, NOT_QUOTED),
        _cell(3, "a", 10.0), _cell(3, "b", 9.0, NOT_COMPARABLE),
        _cell(4, "a", 10.0), _cell(4, "b", 9.0, review=True),
        _cell(5, "a", 10.0), _cell(5, "b", 9.0, review=True, confirmed=True),  # confirmed: clean
    ])
    assert lines_with_issues(comp) == {2, 3, 4}


def test_unconfirmed_counts_only_waiting_values():
    comp = pd.DataFrame([_cell(1, "a", 1.0, review=True), _cell(1, "b", 1.0, review=True, confirmed=True),
                         _cell(1, "c", 1.0)])
    assert unconfirmed(comp).tolist() == [True, False, False]


def test_high_risk_count():
    summary = pd.DataFrame([
        {"vendor": "a", "open_risks": [{"severity": HIGH, "text": "x"}, {"severity": LOW, "text": "y"}]},
        {"vendor": "b", "open_risks": [{"severity": HIGH, "text": "z"}, {"severity": MEDIUM, "text": "w"}]},
        {"vendor": "c", "open_risks": []},
    ])
    assert high_risk_count(summary) == 2


def test_quote_block_quotes_every_line_and_escapes_markdown():
    assert quote_md("Box *A*: 12.50\n\nper 100") == "> Box \\*A\\*\\: 12\\.50\n>\n> per 100"
