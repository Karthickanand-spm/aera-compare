import pandas as pd
import pytest

from aera.compare import (
    AMBIGUOUS_ASSUMPTION,
    COMPARABLE,
    FAIL,
    LAST_YEAR_ASSUMPTION,
    NOT_COMPARABLE,
    NOT_QUOTED,
    PASS,
    UNCLEAR,
    WITH_ASSUMPTION,
    add_months,
    compare,
    display_name,
    match_certificate,
    parse_date,
)
from aera.rfx import RFx, RfxLine

FX = {"USD": 94.50}


def make_rfx(n_lines: int = 3) -> RFx:
    lines = [
        RfxLine(line_id=i, description=f"Carton {i}", ply="3-ply", board_spec="150 GSM",
                size="100 x 100 x 100 mm", print="Plain", annual_qty=1000, uom="pcs",
                nominal_weight_g=100.0 * i)
        for i in range(1, n_lines + 1)
    ]
    return RFx(
        rfx_id="T-1", buyer="Buyer", title="Test", issued="2026-09-16", due="2026-09-25",
        terms={"validity": "12 months"},
        quality_bar={"iso9001_valid_on": "2026-09-16", "max_defect_rate_pct": 2.0,
                     "test_report_per_batch": True},
        questionnaire=[], lines=lines,
    )


def line(line_id, price, unit_basis="per_piece", currency="INR", raw=None, snippet=None, **extra):
    raw = raw if raw is not None else f"{price}"
    return {
        "rfx_line_id": line_id, "raw_price_text": raw, "price": price, "currency": currency,
        "unit_basis": unit_basis, "pack_size": None, "vendor_qty_text": None,
        "vendor_line_total_text": None, "quoted_spec_if_different": None,
        "source_snippet": snippet if snippet is not None else raw, "page": None,
        "interpretation_note": None, "is_ambiguous": False, "candidate_group": None,
        "ambiguity_reason": None, **extra,
    }


def good_questionnaire(**overrides):
    q = {
        "iso9001_claimed": {"value": True, "source_snippet": "ISO yes"},
        "cert_valid_until_claimed": {"value": None, "iso_date": None, "source_snippet": None},
        "defect_rate_pct": {"value": 1.0, "raw_text": "1%", "source_snippet": "1%"},
        "test_report_per_batch": {"value": "yes", "source_snippet": "yes"},
        "capacity_t_month": {"value": 500.0, "raw_text": "500 T", "source_snippet": "500 T"},
        "lead_time_days": {"value": 10.0, "raw_text": "10 days", "source_snippet": "10 days"},
    }
    q.update(overrides)
    return q


def reply(lines, vendor="Acme Boxes", not_quoted=None, freight="included", questionnaire=None,
          file_kind="email", source_file="acme.eml", extra_runs=None, payment_days=30.0):
    run = {
        "vendor": vendor,
        "lines": lines,
        "not_quoted": not_quoted or [],
        "commercial_terms": {
            "freight": {"value": freight, "source_snippet": None},
            "payment_days": {"value": payment_days, "raw_text": None, "source_snippet": None},
            "validity": {"value": None, "source_snippet": None},
            "discounts": [], "other_conditions": [],
        },
        "questionnaire": questionnaire or good_questionnaire(),
    }
    return {"source_file": source_file, "file_kind": file_kind, "runs": [run] + (extra_runs or [])}


def cert(holder="Acme Boxes", valid_until="2028-01-01"):
    return {
        "source_file": "acme_cert.pdf",
        "runs": [{"holder": holder, "standard": "ISO 9001:2015",
                  "valid_until": {"value": valid_until, "iso_date": valid_until,
                                  "source_snippet": f"Valid until: {valid_until}"}}],
    }


def run_compare(ext, certificates=None, last_year=None, text=None, n_lines=3):
    texts = {ext["source_file"]: text} if text is not None else {}
    return compare(make_rfx(n_lines), last_year or {}, [ext], certificates or [], FX,
                   "2026-09-25", texts)


def row_for(df, line_id):
    return df[df["rfx_line_id"] == line_id].iloc[0]


# ---------- Prices and labels ----------

def test_per_kg_uses_rfx_box_weight():
    df, _ = run_compare(reply([line(2, 38.0, "per_kg", raw="38/kg")]), text="38/kg")
    r = row_for(df, 2)
    assert r["price_inr_per_piece"] == pytest.approx(7.6)  # 38 x 200 g / 1000
    assert r["label"] == WITH_ASSUMPTION
    assert any("200 g" in a for a in r["assumptions"])
    assert r["confidence"] == "medium"


def test_same_as_last_year_resolves_from_contract():
    entry = line(1, None, "reference_last_year", currency=None, raw="same as last year")
    df, _ = run_compare(reply([entry]), last_year={1: 5.56}, text="same as last year")
    r = row_for(df, 1)
    assert r["price_inr_per_piece"] == pytest.approx(5.56)
    assert r["label"] == WITH_ASSUMPTION
    assert r["assumptions"][0].startswith(LAST_YEAR_ASSUMPTION)


def test_same_as_last_year_without_contract_price_is_not_zero():
    entry = line(1, None, "reference_last_year", currency=None, raw="same as last year")
    df, _ = run_compare(reply([entry]), last_year={}, text="same as last year")
    r = row_for(df, 1)
    assert pd.isna(r["price_inr_per_piece"])
    assert r["label"] == NOT_COMPARABLE
    assert r["needs_review"]


def test_not_quoted_stays_none_never_zero():
    df, summary = run_compare(reply([line(1, 5.0)]), text="5.0")
    r = row_for(df, 3)
    assert r["label"] == NOT_QUOTED
    assert pd.isna(r["price_inr_per_piece"])
    assert r["price_inr_per_piece"] != 0
    assert not r["included_in_totals"]
    assert summary.iloc[0]["lines_quoted"] == 1


def test_explicit_decline_is_not_quoted_with_its_snippet():
    nq = [{"rfx_line_id": 2, "reason": "not in range", "source_snippet": "we don't make 7-ply"}]
    df, _ = run_compare(reply([line(1, 5.0)], not_quoted=nq), text="5.0")
    r = row_for(df, 2)
    assert r["label"] == NOT_QUOTED
    assert r["source_snippet"] == "we don't make 7-ply"


def test_spec_deviation_is_not_comparable_and_excluded_from_totals():
    entry = line(1, 4.5, quoted_spec_if_different="150 GSM top liner instead of 180")
    df, _ = run_compare(reply([entry]), text="4.5")
    r = row_for(df, 1)
    assert r["label"] == NOT_COMPARABLE
    assert r["price_inr_per_piece"] == pytest.approx(4.5)  # still shown
    assert not r["included_in_totals"]


def test_inr_per_piece_with_snippet_found_is_comparable_high():
    df, _ = run_compare(reply([line(1, 5.93, raw="5.93", snippet="Steam iron 5.93")]),
                        text="Item 1  Steam   iron 5.93\n")
    r = row_for(df, 1)
    assert r["label"] == COMPARABLE
    assert r["confidence"] == "high"
    assert r["included_in_totals"]
    assert not r["buyer_confirmed"]


def test_snippet_missing_from_text_is_low():
    df, _ = run_compare(reply([line(1, 5.93, snippet="made up text")]), text="something else")
    r = row_for(df, 1)
    assert r["confidence"] == "low"
    assert r["needs_review"]


def test_per_100_divides_exactly_once():
    # Even if extraction wrongly fills pack_size for a per_100 price, divide by 100 only once.
    entry = line(1, 514.0, "per_100", raw="Rs 514", pack_size=100)
    df, _ = run_compare(reply([entry]), text="Rs 514")
    r = row_for(df, 1)
    assert r["price_inr_per_piece"] == pytest.approx(5.14)
    assert r["assumptions"] == ["Quoted per 100 pieces; divided by 100"]


def test_candidate_group_picks_higher_price_and_flags_review():
    per_kg = line(1, 38.0, "per_kg", raw="38 for the 3-ply", candidate_group="line-1",
                  is_ambiguous=True)  # 38 x 100 g = 3.80
    last_year = line(1, None, "reference_last_year", currency=None, raw="rest same as last year",
                     candidate_group="line-1", is_ambiguous=True)  # 3.23
    df, _ = run_compare(reply([per_kg, last_year]), last_year={1: 3.23},
                        text="38 for the 3-ply, rest same as last year")
    rows = df[df["rfx_line_id"] == 1]
    assert len(rows) == 1  # one row per (line, vendor)
    r = rows.iloc[0]
    assert r["price_inr_per_piece"] == pytest.approx(3.80)
    assert r["raw_price_text"] == "38 for the 3-ply"
    assert r["confidence"] == "low"
    assert r["needs_review"]
    assert AMBIGUOUS_ASSUMPTION in r["assumptions"]
    assert len(r["alternatives"]) == 1
    assert r["alternatives"][0]["price_inr_per_piece"] == pytest.approx(3.23)


def test_usd_price_converts_and_is_medium():
    df, _ = run_compare(reply([line(1, 0.057, currency="USD")]), text="0.057")
    r = row_for(df, 1)
    assert r["price_inr_per_piece"] == pytest.approx(0.057 * 94.5)
    assert r["confidence"] == "medium"


def test_photo_readings_disagree_is_low():
    run2 = {"vendor": "Acme Boxes", "lines": [line(1, 0.75)], "not_quoted": []}
    ext = reply([line(1, 0.057)], file_kind="image", source_file="acme.jpg", extra_runs=[run2])
    df, _ = run_compare(ext)
    r = row_for(df, 1)
    assert r["confidence"] == "low"
    assert r["needs_review"]


def test_photo_readings_agree_is_medium():
    run2 = {"vendor": "Acme Boxes", "lines": [line(1, 0.057)], "not_quoted": []}
    ext = reply([line(1, 0.057)], file_kind="image", source_file="acme.jpg", extra_runs=[run2])
    df, _ = run_compare(ext)
    assert row_for(df, 1)["confidence"] == "medium"


def test_vendor_line_total_mismatch_is_low():
    entry = line(1, 5.0, vendor_qty_text="1,000", vendor_line_total_text="6,000")
    df, _ = run_compare(reply([entry]), text="5.0")
    assert row_for(df, 1)["confidence"] == "low"


# ---------- Quality ----------

def quality_of(questionnaire=None, certificates=None, freight="included", payment_days=30.0):
    ext = reply([line(1, 5.0)], questionnaire=questionnaire, freight=freight,
                payment_days=payment_days)
    _, summary = run_compare(ext, certificates=certificates, text="5.0")
    return summary.iloc[0]


def risk_texts(summary_row) -> list[str]:
    return [r["text"] for r in summary_row["open_risks"]]


def severity_of(summary_row, starts_with: str) -> str:
    return next(r["severity"] for r in summary_row["open_risks"] if r["text"].startswith(starts_with))


def test_valid_certificate_and_answers_pass():
    s = quality_of(certificates=[cert(valid_until="2028-01-01")])
    assert s["quality_status"] == PASS
    assert s["open_risks"] == []


def test_expired_certificate_fails_even_if_vendor_claims_iso():
    q = good_questionnaire(cert_valid_until_claimed={"value": "31-Dec-2027", "iso_date": "2027-12-31",
                                                     "source_snippet": "valid till 31-Dec-2027"})
    s = quality_of(questionnaire=q, certificates=[cert(valid_until="2026-03-15")])
    assert s["quality_status"] == FAIL
    assert any("expired 2026-03-15" in r for r in s["quality_reasons"])


def test_certificate_expiring_mid_contract_passes_with_open_risk():
    q = good_questionnaire(cert_valid_until_claimed={"value": "Dec 26", "iso_date": None,
                                                     "source_snippet": "ISO valid till Dec 26"})
    s = quality_of(questionnaire=q, certificates=[cert(valid_until="2026-12-31")])
    assert s["quality_status"] == PASS
    assert ("ISO certificate expires 2026-12-31, 3 months into the contract; "
            "ask for renewal evidence") in risk_texts(s)
    # Vague questionnaire date: the certificate's full date is used.
    assert any("'Dec 26'" in r and "2026-12-31" in r for r in s["quality_reasons"])


def test_test_report_on_request_is_unclear():
    q = good_questionnaire(test_report_per_batch={"value": "on_request", "source_snippet": "on request"})
    s = quality_of(questionnaire=q, certificates=[cert()])
    assert s["quality_status"] == UNCLEAR
    assert any("on request" in r for r in s["quality_reasons"])


def test_missing_defect_rate_is_unclear_not_pass():
    q = good_questionnaire(defect_rate_pct={"value": None, "raw_text": None, "source_snippet": None})
    s = quality_of(questionnaire=q, certificates=[cert()])
    assert s["quality_status"] == UNCLEAR
    assert "UNCLEAR: defect rate not provided" in s["quality_reasons"]


def test_defect_rate_above_limit_fails():
    q = good_questionnaire(defect_rate_pct={"value": 2.8, "raw_text": "2.8%", "source_snippet": "2.8%"})
    s = quality_of(questionnaire=q, certificates=[cert()])
    assert s["quality_status"] == FAIL


def test_iso_claim_without_certificate_is_unclear():
    s = quality_of(certificates=[])
    assert s["quality_status"] == UNCLEAR


def test_freight_extra_is_high_risk():
    s = quality_of(certificates=[cert()], freight="extra")
    assert severity_of(s, "Freight extra") == "high"


def test_freight_unclear_is_medium_risk():
    s = quality_of(certificates=[cert()], freight="unclear")
    assert severity_of(s, "Freight terms unclear") == "medium"


# ---------- Risk severity ----------
# RFx issued 2026-09-16, 12-month contract, so it ends 2027-09-16 and "within 6 months" is
# before 2027-03-16.

def test_certificate_expiring_within_6_months_is_high_risk():
    s = quality_of(certificates=[cert(valid_until="2026-12-31")])
    assert severity_of(s, "ISO certificate expires") == "high"


def test_certificate_expiring_after_6_months_is_low_risk():
    s = quality_of(certificates=[cert(valid_until="2027-08-31")])
    assert s["quality_status"] == PASS
    assert severity_of(s, "ISO certificate expires") == "low"


def test_certificate_expiring_exactly_6_months_in_is_low_risk():
    s = quality_of(certificates=[cert(valid_until="2027-03-16")])
    assert severity_of(s, "ISO certificate expires") == "low"


def test_certificate_outlasting_the_contract_is_no_risk():
    s = quality_of(certificates=[cert(valid_until="2027-09-16")])
    assert not any(t.startswith("ISO certificate") for t in risk_texts(s))


def test_risks_sorted_high_first():
    s = quality_of(certificates=[cert(valid_until="2027-08-31")], freight="unclear",
                   questionnaire=good_questionnaire(capacity_t_month={"value": None}))
    severities = [r["severity"] for r in s["open_risks"]]
    assert severities == sorted(severities, key=["high", "medium", "low"].index)
    assert severities[-1] == "low"


# ---------- Missing non-quality-bar answers ----------

def test_missing_capacity_and_lead_time_is_one_medium_risk():
    q = good_questionnaire(capacity_t_month={"value": None, "raw_text": None, "source_snippet": None},
                           lead_time_days={"value": None, "raw_text": None, "source_snippet": None})
    s = quality_of(questionnaire=q, certificates=[cert()])
    assert {"severity": "medium", "text": "Not provided: capacity, lead time"} in s["open_risks"]
    assert s["quality_status"] == PASS  # not part of the quality bar


def test_missing_payment_terms_is_listed():
    s = quality_of(certificates=[cert()], payment_days=None)
    assert "Not provided: payment terms" in risk_texts(s)


def test_all_answers_given_means_no_not_provided_risk():
    s = quality_of(certificates=[cert()])
    assert not any(t.startswith("Not provided") for t in risk_texts(s))


# ---------- Helpers ----------

def test_match_certificate_ignores_case_and_company_suffixes():
    certs = [cert(holder="Other Packaging"), cert(holder="Acme Boxes Pvt Ltd")]
    assert match_certificate("ACME BOXES PVT. LTD.", certs)["runs"][0]["holder"] == "Acme Boxes Pvt Ltd"
    assert match_certificate("Totally Different Co", certs) is None


@pytest.mark.parametrize("raw, expected", [
    ("DECCAN CORRUPACK PVT LTD", "Deccan Corrupack Pvt Ltd"),
    ("HARBOURLINE PACKAGING (EOU)", "Harbourline Packaging (EOU)"),
    ("SAHYADRI BOXES & CARTONS", "Sahyadri Boxes & Cartons"),
    ("Ganesh Packaging Industries", "Ganesh Packaging Industries"),  # already mixed case: kept
    ("ACME  BOXES LLP", "Acme Boxes LLP"),
])
def test_display_name(raw, expected):
    assert display_name(raw) == expected


def test_display_name_is_a_column_in_both_tables():
    df, summary = run_compare(reply([line(1, 5.0)], vendor="ACME BOXES PVT LTD"), text="5.0")
    assert set(df["display_name"]) == {"Acme Boxes Pvt Ltd"}
    assert summary.iloc[0]["display_name"] == "Acme Boxes Pvt Ltd"
    assert summary.iloc[0]["vendor"] == "ACME BOXES PVT LTD"  # raw name kept as the key


def test_parse_date_rejects_vague_dates():
    assert parse_date("Dec 26") is None
    assert str(parse_date("31-Aug-2027")) == "2027-08-31"


def test_add_months_clamps_month_end():
    assert str(add_months(parse_date("2026-01-31"), 1)) == "2026-02-28"
    assert str(add_months(parse_date("2026-09-16"), 12)) == "2027-09-16"
