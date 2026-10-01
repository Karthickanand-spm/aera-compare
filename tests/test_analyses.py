"""Tests for analyses.py: one function per intent, plus the vendor scope. Pure code, no API calls."""

import json

import numpy as np
import pandas as pd
import pytest

from aera import analyses
from aera.analyses import (
    AnalystData, award_capped, award_split, fx_sensitivity, landed_cost, line_lookup, quality_risk,
    recommendation, vendor_scope, vendor_totals,
)
from aera.money import display_table, format_inr

DECCAN, INDUS, SAHYADRI = "Deccan Corrupack Pvt Ltd", "Indus Packaging Solutions", "Sahyadri Boxes & Cartons"
HARBOURLINE, GANESH = "Harbourline Packaging (EOU)", "Ganesh Packaging Industries"


@pytest.fixture(scope="module")
def sample() -> AnalystData:
    from aera.compare import compare
    from aera.config import FX_DATE, FX_RATES
    from aera.event import load_sample_event
    event = load_sample_event()
    comparison, summary = compare(event.rfx, event.last_year, event.replies, event.certificates,
                                  FX_RATES, FX_DATE, event.texts, {})
    return analyses.analyst_data(comparison, summary, event.rfx, event.last_year, FX_RATES, FX_DATE)


def _facts_text(a) -> str:
    return json.dumps(a.facts, ensure_ascii=False)


# ---------- Vendor scope ----------

def test_default_scope_is_quality_pass_only_with_reasons(sample):
    scope = vendor_scope(sample)
    assert scope.names == [DECCAN, INDUS, GANESH]
    assert scope.excluded == [{"display_name": SAHYADRI, "reason": "failed quality"},
                              {"display_name": HARBOURLINE, "reason": "quality unclear"}]
    assert scope.flagged == []


def test_include_failed_brings_them_in_flagged(sample):
    scope = vendor_scope(sample, include_failed=True)
    assert len(scope.names) == 5 and scope.excluded == []
    assert scope.flagged == [SAHYADRI, HARBOURLINE]


def test_a_named_failed_vendor_still_needs_include_failed(sample):
    scope = vendor_scope(sample, ["sahyadri", "Deccan Corrupack Pvt Ltd"])
    assert scope.names == [DECCAN]
    assert scope.excluded == [{"display_name": SAHYADRI, "reason": "failed quality"}]


def test_vendor_names_resolve_by_case_short_name_and_unknowns_are_kept(sample):
    names, unknown = analyses.resolve_vendors(["ganesh", "INDUS PACKAGING SOLUTIONS", "Acme Boxes"], sample)
    assert names == [GANESH, INDUS]
    assert unknown == ["Acme Boxes"]
    assert any("Acme Boxes" in c for c in award_split(sample, vendor_scope(sample, ["Acme Boxes"])).caveats)


def test_quality_warning_is_one_sentence_with_the_reasons(sample):
    warning = analyses.quality_warning([SAHYADRI], sample)
    assert warning.startswith(f"Includes {SAHYADRI} (failed quality: ISO 9001 certificate expired 2026-03-15")
    assert warning.endswith("not a valid award option as things stand.")
    assert ".pdf" not in warning


# ---------- award_split ----------

def test_award_split_on_the_sample(sample):
    a = award_split(sample, vendor_scope(sample))
    assert a.values["total_inr"] == pytest.approx(36_400_000, abs=100_000)
    assert format_inr(a.values["saving_inr"]) == "₹12.03 lakh"
    assert a.facts["total_annual_cost"] == "₹3.64 crore"
    assert a.facts["saving_vs_last_year"] == "saves ₹12.03 lakh (3.2%)"
    assert a.values["lines_won"] == {GANESH: 21, INDUS: 7, DECCAN: 2}
    shown = display_table(a.table).set_index("vendor")
    assert shown.loc["Total split award", "saving (₹)"] == "saves ₹12.03 lakh (3.2%)"
    assert a.facts["excluded_vendors"] == [f"{SAHYADRI} (failed quality)", f"{HARBOURLINE} (quality unclear)"]


def test_award_split_carries_the_award_freight_sentence(sample):
    a = award_split(sample, vendor_scope(sample))
    assert a.facts["headline_before_freight"] == "Saves ₹12.03 lakh (3.2%) before freight"
    assert a.facts["freight_vendors"] == [GANESH]
    assert a.facts["freight_must_include"] == ["₹3.50/kg"]
    assert [sv["vendor"] for sv in a.sensitivity] == [GANESH]
    assert a.missing_data == [f"{GANESH}'s freight charge in ₹ per kg"]


def test_award_split_facts_hold_only_formatted_strings(sample):
    text = _facts_text(award_split(sample, vendor_scope(sample)))
    assert "36375440" not in text and "1202720" not in text


def test_award_split_with_include_failed_warns_first(sample):
    a = award_split(sample, vendor_scope(sample, [SAHYADRI, DECCAN], include_failed=True))
    assert a.facts["quality_warning"].startswith(f"Includes {SAHYADRI} (failed quality")
    assert "excluded_vendors" not in a.facts


# ---------- award_capped ----------

def test_award_capped_at_two_picks_indus_and_ganesh(sample):
    a = award_capped(sample, vendor_scope(sample), 2)
    assert set(a.values["chosen"]) == {INDUS, GANESH}
    assert a.facts["chosen_vendors"] == [GANESH, INDUS]
    assert a.tag == "award capped at 2 vendors"
    assert a.facts["combinations_compared"] == "3"
    # Capping can only cost more than the free split.
    assert a.values["total_inr"] >= a.values["split_total_inr"]
    assert a.facts["against_uncapped_split"].startswith("costs ₹")


def test_award_capped_has_no_award_freight_sentence_but_a_caveat(sample):
    a = award_capped(sample, vendor_scope(sample), 2)
    assert "before freight" not in _facts_text(a) and "headline_before_freight" not in a.facts
    assert a.sensitivity == []
    assert f"{GANESH} hasn't quoted freight, so this total is before freight." in a.caveats


def test_award_capped_at_one_is_the_best_single_vendor(sample):
    a = award_capped(sample, vendor_scope(sample), 1)
    assert a.values["chosen"] == [GANESH]  # covers all 30 lines, and Indus covers only 25


# ---------- vendor_totals ----------

def test_vendor_totals_main_view_uses_only_the_common_lines(sample):
    a = vendor_totals(sample, vendor_scope(sample))
    assert a.facts["common_lines_count"] == "25"
    assert a.tag == "vendor totals, like for like on 25 lines"
    assert set(a.table["lines_in_total"]) == {25}
    assert list(a.table["vendor"]) == [GANESH, INDUS, DECCAN]
    (title, full), = a.extra_tables
    assert "don't compare" in title
    assert full.set_index("vendor")["lines_in_total"].to_dict() == {DECCAN: 30, GANESH: 30, INDUS: 25}


def test_vendor_totals_has_no_award_or_freight_facts(sample):
    text = _facts_text(vendor_totals(sample, vendor_scope(sample)))
    assert "freight" not in text and "saving" not in text


def test_vendor_totals_chart_spec(sample):
    a = vendor_totals(sample, vendor_scope(sample), chart=True)
    assert a.answer_type == "chart"
    assert a.chart_spec == {"x": "vendor", "y": "like_for_like_total_inr", "kind": "bar"}


def _two_vendors_no_common_line() -> AnalystData:
    """A quotes only line 1, B only line 2: no line is common."""
    rows = [{"rfx_line_id": 1, "vendor": "a", "display_name": "A", "price_inr_per_piece": 5.0},
            {"rfx_line_id": 2, "vendor": "a", "display_name": "A", "price_inr_per_piece": None},
            {"rfx_line_id": 1, "vendor": "b", "display_name": "B", "price_inr_per_piece": None},
            {"rfx_line_id": 2, "vendor": "b", "display_name": "B", "price_inr_per_piece": 4.0}]
    for r in rows:
        priced = r["price_inr_per_piece"] is not None
        r.update(label="Comparable" if priced else "Not quoted", included_in_totals=priced, needs_review=False,
                 buyer_confirmed=False, assumptions=[])
    vendors = pd.DataFrame([{"vendor": v, "display_name": v.upper(), "quality_status": "PASS", "freight": "included",
                             "quality_reasons": [], "open_risks": []} for v in "ab"])
    lines = pd.DataFrame([{"line_id": i, "description": f"Box {i}", "annual_qty": 1000, "uom": "pcs",
                           "nominal_weight_g": 500.0} for i in (1, 2)])
    last_year = pd.DataFrame([{"line_id": 1, "price_inr_per_piece": 6.0}])
    return AnalystData(pd.DataFrame(rows), vendors, lines, last_year, {"USD": 94.5}, "2026-09-25")


def test_vendor_totals_without_a_common_line_says_so_and_shows_full_totals():
    data = _two_vendors_no_common_line()
    a = vendor_totals(data, vendor_scope(data), chart=True)
    assert "common_lines_count" not in a.facts and "no like-for-like total" in a.facts["no_common_lines"]
    assert a.answer_type == "table" and "total_on_own_lines_inr" in a.table.columns


# ---------- recommendation ----------

def test_recommendation_returns_three_views_each_with_a_main_caveat(sample):
    a = recommendation(sample, vendor_scope(sample))
    assert list(a.table["view"]) == ["Cheapest single vendor", "Lowest risk", "Cheapest split"]
    assert a.values["cheapest"] == GANESH and a.values["safest"] == INDUS
    assert all(a.table["main_caveat"].str.len() > 0)
    views = a.table.set_index("view")
    assert "hasn't quoted freight" in views.loc["Cheapest single vendor", "main_caveat"]
    assert views.loc["Lowest risk", "main_caveat"] == f"Costs ₹4.31 lakh more than {GANESH} on the 25 common lines."
    assert display_table(a.table).set_index("view").loc["Cheapest split", "annual_cost (₹)"] == "₹3.64 crore"
    assert SAHYADRI not in " ".join(a.table["vendors"])
    assert len(a.facts["views"]) == 3


# ---------- landed_cost ----------

def test_landed_cost_reruns_the_award_with_freight(sample):
    a = landed_cost(sample, vendor_scope(sample), rate_inr_per_kg=2)
    assert a.facts["freight_vendors"] == [GANESH]
    assert a.facts["headline_before_freight"] == "Saves ₹12.03 lakh (3.2%) before freight"
    assert a.values["zero_rates"] == {GANESH: 3.5}
    assert a.facts["at_the_rate_asked"][0].startswith(f"With {GANESH}'s freight at ₹2.00/kg the award saves")
    rates = list(a.sensitivity[0]["table"]["rate_inr_per_kg"])
    assert rates == sorted(rates) and 2.0 in rates


def test_landed_cost_with_freight_included_everywhere_adds_nothing():
    data = _two_vendors_no_common_line()
    a = landed_cost(data, vendor_scope(data))
    assert "no_freight_to_add" in a.facts and a.sensitivity == []


# ---------- line_lookup ----------

def test_line_lookup_gives_prices_labels_and_receipts(sample):
    a = line_lookup(sample, vendor_scope(sample), [3, 14])
    assert a.tag == "line lookup, lines 3 and 14"
    assert set(a.table["line_id"]) == {3, 14}
    assert a.table["source_snippet"].notna().all()  # every number has a receipt
    line_14 = a.table[a.table["line_id"] == 14].set_index("vendor")
    assert line_14.loc[INDUS, "label"] == "Not comparable"
    assert a.facts["line_ids_asked"] == ["3", "14"]
    assert a.facts["lines"][1].startswith("Line 14 (")
    assert f"Cheapest counted price: {GANESH}" in a.facts["lines"][0]


def test_line_lookup_keeps_not_quoted_as_missing_never_zero(sample):
    indus_missing = sample.df[(sample.df["display_name"] == INDUS) & (sample.df["label"] == "Not quoted")]
    line = int(indus_missing["rfx_line_id"].iloc[0])
    a = line_lookup(sample, vendor_scope(sample), [line])
    row = a.table.set_index("vendor").loc[INDUS]
    assert pd.isna(row["price_inr_per_piece"])
    assert f"{INDUS} not quoted" in a.facts["lines"][0]


def test_line_lookup_over_all_lines_lists_what_each_vendor_did_not_quote(sample):
    df = sample.df
    expected = sorted(int(i) for i in df[(df["display_name"] == INDUS) & (df["label"] == "Not quoted")]["rfx_line_id"])
    a = line_lookup(sample, vendor_scope(sample, [INDUS]))
    assert a.facts["not_quoted"] == [f"{INDUS} did not quote {len(expected)} lines: "
                                     + analyses.join_names([str(i) for i in expected])]
    assert a.facts["not_quoted_line_ids"] == [str(i) for i in expected]


def test_line_lookup_notes_assumptions_and_freight_next_to_a_price(sample):
    a = line_lookup(sample, vendor_scope(sample, [GANESH]), [11])
    assert "per kg" in a.facts["lines"][0] and "freight extra" in a.facts["lines"][0]
    assert any("freight extra" in n for n in a.facts["price_notes"])


def test_line_lookup_reports_a_line_that_does_not_exist(sample):
    a = line_lookup(sample, vendor_scope(sample), [3, 99])
    assert any("no RFx line 99" in c for c in a.caveats)


# ---------- fx_sensitivity ----------

def test_fx_up_3_pct_raises_harbourline_prices_3_pct(sample):
    a = fx_sensitivity(sample, vendor_scope(sample, include_failed=True), 3)
    prices = a.values["prices"]
    assert set(prices["vendor"]) == {HARBOURLINE}  # the only vendor priced in USD
    assert len(prices) == 30
    assert np.allclose(prices["price_after_inr_per_piece"] / prices["price_before_inr_per_piece"], 1.03)
    assert a.facts["rate"] == "₹94.50 per USD becomes ₹97.34"
    assert a.facts["lines_changing_hands_count"] == str(len(a.values["changes"]))
    assert a.tag == "FX sensitivity, USD +3%"
    assert "quality_warning" in a.facts  # Harbourline is UNCLEAR, included only because asked


def test_fx_lists_lines_that_change_hands(sample):
    a = fx_sensitivity(sample, vendor_scope(sample, include_failed=True), 3)
    for c in a.values["changes"]:
        assert c["winner_before"] == HARBOURLINE and c["winner_after"] != HARBOURLINE
        assert c["price_after_inr_per_piece"] > c["price_before_inr_per_piece"]
    assert list(a.table["line_id"]) == [c["line_id"] for c in a.values["changes"]]


def test_fx_with_default_scope_has_no_usd_vendor_and_says_who_was_left_out(sample):
    a = fx_sensitivity(sample, vendor_scope(sample), 3)
    assert a.facts["vendors_priced_in_currency"] == "none"
    assert a.facts["currency_vendors_not_considered"] == [f"{HARBOURLINE} (quality unclear)"]
    assert a.facts["lines_changing_hands_count"] == "0"
    assert a.table is None and a.answer_type == "text"


def test_fx_leaves_the_original_comparison_untouched(sample):
    before = sample.df["price_inr_per_piece"].copy()
    fx_sensitivity(sample, vendor_scope(sample, include_failed=True), 10)
    pd.testing.assert_series_equal(sample.df["price_inr_per_piece"], before)


# ---------- quality_risk ----------

def test_quality_risk_covers_every_vendor_with_claims_against_certificates(sample):
    a = quality_risk(sample)
    assert a.values["statuses"] == {DECCAN: "PASS", INDUS: "PASS", SAHYADRI: "FAIL", HARBOURLINE: "UNCLEAR",
                                    GANESH: "PASS"}
    assert a.facts["not_passed"] == [SAHYADRI, HARBOURLINE]
    cert = dict(x.split(": ", 1) for x in a.facts["certificate_vs_claims"])
    assert "expired 2026-03-15" in cert[SAHYADRI] and "vendor claims ISO 9001" in cert[SAHYADRI]
    assert a.facts["vendors_with_high_risks"] == [GANESH]
    (_, risks), = a.extra_tables
    assert list(risks["severity"]) == sorted(risks["severity"], key=["high", "medium", "low"].index)


def test_quality_risk_for_named_vendors_only(sample):
    a = quality_risk(sample, ["harbourline"])
    assert list(a.table["vendor"]) == [HARBOURLINE]


# ---------- Freight sensitivity on a small award with a known breakeven ----------

def _two_line_data() -> AnalystData:
    """2 lines, 1,000 pieces each, 500 g a box, last year ₹10.00 on both.
    A (freight extra): ₹8.00 and ₹9.00.   B (freight included): ₹11.00 and ₹10.50.
    Before freight A wins both and saves ₹3,000; the saving is gone at ₹3.00/kg."""
    rows = []
    for vendor, prices in (("A", (8.0, 9.0)), ("B", (11.0, 10.5))):
        for line_id, price in zip((1, 2), prices):
            rows.append({"rfx_line_id": line_id, "vendor": vendor, "display_name": vendor,
                         "price_inr_per_piece": price, "label": "Comparable", "included_in_totals": True,
                         "needs_review": False, "buyer_confirmed": False, "assumptions": []})
    vendors = pd.DataFrame([
        {"vendor": "A", "display_name": "A", "quality_status": "PASS", "freight": "extra"},
        {"vendor": "B", "display_name": "B", "quality_status": "PASS", "freight": "included"},
    ])
    rfx_lines = pd.DataFrame([{"line_id": i, "description": f"Box {i}", "annual_qty": 1000, "uom": "pcs",
                               "nominal_weight_g": 500.0} for i in (1, 2)])
    last_year = pd.DataFrame([{"line_id": 1, "price_inr_per_piece": 10.0},
                              {"line_id": 2, "price_inr_per_piece": 10.0}])
    return AnalystData(pd.DataFrame(rows), vendors, rfx_lines, last_year, {"USD": 94.5}, "2026-09-25")


def test_freight_sensitivity_on_a_two_line_award_with_a_known_breakeven():
    s = analyses.freight_sensitivity(_two_line_data(), "A", ["A", "B"])
    assert s["zero_rate"] == 3.0
    table = s["table"].set_index("rate_inr_per_kg")
    assert table.loc[0, "saving_inr"] == 3000.0 and table.loc[0, "saving_pct"] == 15.0
    # The saving shrinks from the first rupee of freight, not only above some rate.
    assert table.loc[0.5, "saving_inr"] == 2500.0
    assert table.loc[2, "saving_inr"] == 1000.0
    # At ₹4/kg line 2 has moved to B (10.50 < 11.00); A keeps line 1 at 10.00 (< B's 11.00).
    assert table.loc[4, "lines_won"] == 1
    assert table.loc[4, "total_inr"] == 20500.0 and table.loc[4, "saving_inr"] == -500.0


def test_freight_effect_wording():
    e = analyses.freight_effect(_two_line_data(), "A", ["A", "B"])
    assert e["headline"] == "Saves ₹3,000 (15.0%) before freight"
    assert e["text"] == ("A wins 2 lines but hasn't quoted freight; the saving shrinks as freight rises "
                         "and is gone at about ₹3.00/kg (table below)")
    assert e["must_include"] == ["₹3.00/kg"]


def test_missing_box_weight_is_reported_not_treated_as_zero():
    data = _two_line_data()
    data.rfx_lines.loc[0, "nominal_weight_g"] = None
    e = analyses.freight_effect(data, "A", ["A", "B"])
    assert e["sensitivity"] is None and e["headline"] is None
    assert "no box weight for line(s) [1]" in e["text"]


def test_award_split_on_the_small_award_and_lines_nobody_priced():
    data = _two_line_data()
    b_2 = (data.df["vendor"] == "B") & (data.df["rfx_line_id"] == 2)
    a_2 = (data.df["vendor"] == "A") & (data.df["rfx_line_id"] == 2)
    data.df.loc[a_2 | b_2, ["price_inr_per_piece", "label", "included_in_totals"]] = [None, "Not quoted", False]
    a = award_split(data, vendor_scope(data))
    assert a.values["total_inr"] == 8000.0  # line 2 has no price: left out, never counted as zero
    assert a.facts["lines_awarded"] == "1 of 2 lines"
    assert "line 2" not in a.facts["lines_not_awarded"] and "(2)" in a.facts["lines_not_awarded"]
